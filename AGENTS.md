# AGENTS.md

Architecture and implementation conventions for the jumpto-worker codebase.

## Transcript provider model

There are exactly **4 registered transcript providers**, all first-class and
interchangeable at the registry level (`app/providers/registry.py`):

| name             | class                                     | module                          |
| ---------------- | ----------------------------------------- | ------------------------------- |
| `yt-dlp`         | `YtDlpTranscriptProvider`                 | `app/providers/ytdlp.py`        |
| `transcriptfetch` | `TranscriptFetchTranscriptProvider`       | `app/providers/transcriptfetch.py` |
| `supadata`       | `SupadataTranscriptProvider`              | `app/providers/supadata.py`     |
| `vidwords`       | `VidWordsTranscriptProvider`              | `app/providers/vidwords.py`     |

- Every provider implements `TranscriptProviderStrategy` (`app/providers/base.py`):
  it has a stable `name`, and a `fetch()` returning
  `VideoTranscriptResult | None` (``None`` = soft miss, so the pipeline tries the
  next candidate) or raising `ExternalServiceError`.
- **All four are providers.** Do not model "cloud vs local" as a first-class
  concept; whether a provider needs API keys/network access is an internal
  detail. Providers are kept independently switchable via `DEFAULT_VIDEO_PROVIDER`
  or the ordered `TRANSCRIPT_PROVIDER_CHAIN` list.
- A provider doing work in the background (e.g. Assembly) reports in-progress
  via `TranscriptJobPending`, so the pipeline can resume the same job
  (`resume_token`/`resume_provider`) instead of re-downloading.
- `supports_resume` marks strategies that can honour a `resume_token` (Supadata,
  yt-dlp); the pipeline only routes resumes to them. `fetch()` returns `None`
  on a soft miss for cloud strategies; the terminal yt-dlp strategy never
  returns `None` — it produces a result or raises.
- Deployment metadata that is not transcript behavior (e.g. `uses_cloud` —
  whether the strategy needs live external API calls) lives on the **registry
  spec** (`TranscriptProviderSpec`), not on the strategy interface.

### Self-registration (closed registry)

The registry is **closed**: it never imports concrete provider modules at the
top level. Each provider module owns its own `TranscriptProviderSpec` and calls
`register_provider()` at import time. Lookup functions (`provider_spec`,
`ordered_specs`, `build_provider`) lazily trigger `_ensure_registered()` which
imports the four provider modules once. **Adding a provider** means writing a
new module with a spec, and adding one import line in `_ensure_registered()`.

## The yt-dlp provider is a composite

`YtDlpTranscriptProvider` orchestrates several internal services:

1. Fetch media metadata once (title/duration) — via `app/providers/media.py`.
2. Prefer existing captions — `YouTubeCaptionTranscriptService`
   (`app/providers/transcript.py`); no captions is a normal outcome, falls through.
3. Only if captions are unavailable, fall back to **Assembly audio
   transcription** — `AssemblyTranscriptService` (`app/providers/assembly.py`).

`media.py` and Assembly are **internal services of the yt-dlp provider**, not
registry-level providers. They must stay reusable as services and never be
duplicated into other providers.

The registry's `spec.build` injects `settings` into the composite; its leaf
collaborators are constructor-injectable (`captions_service`,
`assembly_provider`, `cache`) with lazy concrete defaults so a plain
`YtDlpTranscriptProvider()` keeps working. Inject collaborators in tests rather
than monkeypatching module globals.

Provider modules are split by concern:

- `app/providers/base.py` — the two abstract interfaces:
  `TranscriptProviderStrategy` (registry-level strategies) and
  `TranscriptService` (internal leaf fetchers). `TranscriptService.fetch`
  declares only `fetch(url)`; implementation-specific parameters (`info`,
  `resume_token`) are added by each concrete service.
- `exceptions.py` taxonomy: `ExternalServiceError` (transient soft miss, moves
  to next provider), `PermanentExternalServiceError` (fails the job),
  `TranscriptJobPending` (async job still processing, may resume), and
  `BackendCommunicationError` (producer API failures) — don't add finer
  subclasses unless a consumer needs them.
- `app/providers/models.py` — shared domain types: `TranscriptData`,
  `TranscriptWordData`, and the `TranscriptJobPending` signal.
- `app/providers/transcript.py` — the YouTube caption service + VTT parsing.
- `app/providers/assembly.py` — the Assembly audio service plus the
  `get_transcript_provider()` factory that builds it from settings.

Internal leaf fetchers (captions, Assembly) are `*TranscriptService` subclasses
of `TranscriptService` (`app/providers/base.py`); registry strategies are
`*TranscriptProvider` subclasses of `TranscriptProviderStrategy`
(`app/providers/base.py`).

## Single source of truth for yt-dlp configuration

`build_ydlp_options()` in `app/integrations/ytdlp.py` is the **only** place that
knows how to drive yt-dlp (cookie file, POT/bot-check, proxy, extractor args,
format selection). Providers pass provider-specific overrides as
keyword arguments; they must never reimplement option building. Drift between
call sites previously caused a 403 regression — keep it centralized.

## Pipeline flow

`app/tasks/transcription.py` runs in this order:

1. `registry.candidates(settings)` resolves the ordered chain of providers from
   settings: either the explicit `provider_chain` list
   (`TRANSCRIPT_PROVIDER_CHAIN` env var, comma-separated), or the default chain
   — `DEFAULT_VIDEO_PROVIDER` first, then `yt-dlp` as the always-available
   fallback. Unknown or unconfigured providers are silently skipped; cloud
   providers are skipped when live calls are off.
2. `_try_provider()` calls each candidate's `fetch()`, routing resumes by
   provider `name`, and returns the first non-`None` result.
3. The result is submitted via `_build_result_submission()`.

`_live_pipeline_enabled()` gates whether live external calls are allowed
(`JUMPTO_LIVE_EXTERNAL_CALLS`); treat the gate as a runtime switch, not a
provider-category property. When disabled, live data is never fabricated —
jobs fail with `ExternalServiceError`.

## Celery application name

`celery_app_name` (default `"jumpto"`) is the Celery app identity from
`app/core/config.py`. It prefixes the registered task name
(`<app_name>.transcribe_video`), so it must match every producer that enqueues
jobs over the broker. Change it via `CELERY_APP_NAME` only in coordination
with producers; the default must stay stable.

## SOLID principles

Every new feature and every new piece of code must follow SOLID:

- **S — Single Responsibility**: one module, one class, one function — one
  reason to change. If a function mixes orchestration with mapping, split it.
- **O — Open/Closed**: extend via new modules/classes, not by editing existing
  ones. Provider self-registration, env-driven chains, spec metadata — these
  keep the pipeline open to new providers without code edits.
- **L — Liskov Substitution**: subclasses must honour the base-class contract.
  Providers return `VideoTranscriptResult | None`; a soft miss (`None`) is
  never confused with a terminal failure (`ExternalServiceError`).
- **I — Interface Segregation**: keep interfaces minimal. The strategy exposes
  `name`, `supports_resume`, `fetch()`. Deployment metadata (`uses_cloud`) lives
  on the registry spec, not the strategy.
- **D — Dependency Inversion**: depend on abstractions, not concretions. High-level
  pipeline code depends on the registry/spec, not concrete providers. Inject
  collaborators (settings, services, factories) rather than importing concrete
  classes at call time. `get_settings()` singletons are acceptable for
  process-wide config but should not leak into leaf services that need
  testability.