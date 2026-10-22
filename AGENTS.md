# AGENTS.md

Architecture and implementation conventions for the jumpto-worker codebase.

## Transcript provider model

There are exactly **2 registered transcript strategies**, both first-class at
the registry level (`app/providers/registry.py`):

| name      | class                       | module                     |
| --------- | --------------------------- | -------------------------- |
| `youtube` | `YouTubeTranscriptProvider` | `app/providers/youtube.py` |
| `drive`   | `DriveTranscriptProvider`   | `app/providers/drive.py`   |

- Every strategy implements `TranscriptProviderStrategy` (`app/providers/base.py`):
  it has a stable `name`, `supports_resume` and `supports_webhook` flags, and a
  `fetch()` returning `VideoTranscriptResult | None` (``None`` = soft miss, so
  the pipeline fails the job) or raising `ExternalServiceError`.
  `fetch()` accepts an optional keyword-only `webhook_url` (an opaque public
  callback URL, consumed only by the audio leaves, which declare
  ``supports_webhook``).
- **Selection is source-driven, not configured.** A captionless source (Drive)
  resolves to its own same-named strategy; a caption-bearing source (YouTube)
  resolves to `youtube`. There is no fallback chain — an unavailable strategy
  fails the job loudly. (A `VIDEO_PROVIDER` knob used to route YouTube jobs to
  paid cloud transcript APIs; those strategies were removed, so the knob is
  gone.)
- A strategy doing work in the background reports in-progress via
  `TranscriptJobPending`, so the pipeline can resume the same job
  (`resume_token`/`resume_provider`) instead of re-downloading.
- `supports_resume` marks strategies that can honour a `resume_token`; the
  pipeline only routes resumes to them. The local strategies never return
  `None` — they produce a result or raise; only audio leaves can soft-miss.
  `supports_webhook` marks strategies that consume a completion callback URL;
  the pipeline only forwards one to strategies that declare it.
- Deployment metadata that is not transcript behavior (e.g. `uses_cloud` —
  whether the strategy needs live external API calls, or
  `cache_config_fields` — which settings knobs change a provider's output and
  therefore version its cache namespace) lives on the **registry spec**
  (`TranscriptProviderSpec`), not on the strategy interface.

### Self-registration (closed registry)

The registry is **closed**: it never imports concrete provider modules at the
top level. Each provider module owns its own `TranscriptProviderSpec` and calls
`register_provider()` at import time. Lookup functions (`provider_spec`,
`ordered_specs`, `build_provider`) lazily trigger `_ensure_registered()` which
imports the provider modules once. **Adding a provider** means writing a
new module with a spec, and adding one import line in `_ensure_registered()`.

## The YouTube strategy is a composite

`YouTubeTranscriptProvider` orchestrates several internal services:

1. Fetch media metadata once (title/duration) — via `app/providers/media.py`.
2. Prefer existing captions — `YouTubeCaptionTranscriptService`
   (`app/providers/transcript.py`); no captions is a normal outcome, falls through.
3. Only if captions are unavailable, fall back to the configured **audio
   transcription leaf** — `SPEECH_TO_TEXT_PROVIDER` (`deepgram` by default, or
   `assembly`); see `app/speech_to_text/registry.py`.

`media.py` and the speech-to-text leaves are **internal services of the local
strategies**, not registry-level providers. They must stay reusable as services
and never be duplicated into other providers.

The registry's `spec.build` injects `settings` into the composite; its leaf
collaborators are constructor-injectable (`captions_service`,
`speech_to_text_provider`, `cache`) with lazy concrete defaults so a plain
`YouTubeTranscriptProvider()` keeps working. Inject collaborators in tests
rather than monkeypatching module globals.

Provider modules are split by concern:

- `app/providers/base.py` — the registry-level strategy interface,
  `TranscriptProviderStrategy`. The internal leaf-fetcher interface lives in
  `app/speech_to_text/base.py` (`SpeechToTextProvider`); its `fetch(url)`
  declares only the URL, and implementation-specific parameters (`info`,
  `resume_token`) are added by each concrete service.
- `exceptions.py` taxonomy: `ExternalServiceError` (transient soft miss, moves
  to next provider), `PermanentExternalServiceError` (fails the job),
  `TranscriptJobPending` (async job still processing, may resume), and
  `BackendCommunicationError` (producer API failures) — don't add finer
  subclasses unless a consumer needs them. `PermanentExternalServiceError` and
  `TranscriptJobPending` are **siblings** of `ExternalServiceError` (both derive
  from `DomainError`), never subclasses: their meaning is the opposite of a soft
  miss, so `except ExternalServiceError` must never swallow them.
- `app/providers/models.py` — shared domain types: `TranscriptData`,
  `TranscriptWordData`, and the `TranscriptJobPending` signal.
- `app/providers/transcript.py` — the YouTube caption service + VTT parsing.
- `app/speech_to_text/assembly.py` — the Assembly audio service plus the
  `get_transcript_provider()` factory that builds it from settings.
- `app/providers/audio.py` — the audio file itself: download, persisted
  per-video cache paths, stale sweeping, and `AudioFileLease`, the object that
  owns one job's hold on the file (download once, reuse across attempts, release
  when the job settles). The composite holds a lease; it never manages paths or
  temp files itself.
- `app/speech_to_text/deepgram.py` — the Deepgram audio service plus
  `get_deepgram_provider()`. Both audio leaves share the composite's
  `fetch(youtube_url, resume_token, webhook_url, language, audio_path)`
  signature, and the composite passes all five **explicitly** — every parameter
  defaults to `""` and each leaf treats `""` as "not provided". Never reassemble
  that call with a `**kwargs` dict: it hides the leaves' signatures, so renaming
  a leaf parameter would silently drop the value instead of failing.

Internal leaf fetchers (captions, Assembly, Deepgram) are subclasses of
`SpeechToTextProvider` (`app/speech_to_text/base.py`); registry strategies are
`*TranscriptProvider` subclasses of `TranscriptProviderStrategy`
(`app/providers/base.py`).

## Single source of truth for yt-dlp configuration

`build_ydlp_options()` in `app/integrations/ytdlp.py` is the **only** place that
knows how to drive yt-dlp (cookie file, POT/bot-check, proxy, extractor args,
format selection). Providers pass provider-specific overrides as
keyword arguments; they must never reimplement option building. Drift between
call sites previously caused a 403 regression — keep it centralized.

## Pipeline flow

`app/services/pipeline.py` (orchestrated by the thin Celery task
`app/tasks/transcription.py`) runs in this order:

1. `registry.resolve_provider(settings, url)` builds the single strategy the
   job's source resolves to: `drive` for captionless links, `youtube` for
   caption-bearing ones. Unconfigured strategies and cloud strategies while
   live calls are disabled raise instead of being silently skipped — there is
   no fallback chain.
2. `try_provider()` calls the provider's `fetch()`, routing resume tokens when
   `resume_provider` matches the provider `name`, and returns the result or
   `None` on a soft miss. `fetch()` accepts an optional keyword-only
   `webhook_url` (ignored by every strategy except Assembly); the pipeline
   builds it per job via `build_assembly_webhook_url()` (`app/services/webhooks.py`) and forwards it
   only to strategies declaring `supports_webhook`.
3. The result is submitted via `build_result_submission()`
   (`app/services/submissions.py`) through the backend `JobService`
   (`app/services/jobs.py`).

`_live_pipeline_enabled()` gates whether live external calls are allowed
(`LIVE_EXTERNAL_CALLS`); treat the gate as a runtime switch, not a
provider-category property. When disabled, live data is never fabricated —
jobs fail with `ExternalServiceError`.

## Service layer

`app/tasks/transcription.py` is a scheduling shell; the orchestration it used
to own lives in `app/services/`, one concern per module:

- `event_loop.py` — the process-wide asyncio event loop (`run_async`) and its
  atexit teardown.
- `webhooks.py` — `build_assembly_webhook_url()` (completion callback URLs).
- `submissions.py` — `build_submission()` / `build_result_submission()`.
- `jobs.py` — `JobService` (backend client lifecycle: load/advance, submit,
  best-effort fail, close) plus `fail_job()` for out-of-pipeline failure
  marking.
- `pipeline.py` — `run_pipeline()`, `perform_transcription()`, `try_provider()`,
  `user_safe_message()` and the user-safe message constants.

The task owns Celery concerns only: retry semantics, the async-job wait budget
(`_CLOUD_JOB_ATTEMPTS`, `_job_retry_countdown`), its own task-side failure
message (`_CLOUD_JOB_TIMEOUT_SAFE_MESSAGE`), and webhook-driven clean exit.
Nothing above it shares code with the task module.

## Completion webhooks (Assembly)

Assembly transcription runs ~real-time, far beyond the in-worker retry budget.
When `ASSEMBLY_WEBHOOK_BASE_URL` is set, the worker stops poller-retrying and
lets Assembly deliver completion out-of-band:

1. The pipeline builds a callback URL — ``<base>?job_id=<job_id>&provider=youtube``
   (`app/services/webhooks.py:build_assembly_webhook_url`) and passes it to
   `AssemblyTranscriptService.fetch(..., webhook_url=...)`.
2. Assembly submits the transcript with that `webhook_url`; a still-processing
   poll raises `TranscriptJobPending(..., webhook=True)`.
3. The Celery task ends cleanly with `{"status": "submitted"}` — the job is
   **not** failed and **not** retried; it stays `processing` until completion.

The webhook receiver lives on the **backend** (separate repo). Contract:

- Endpoint: the exact URL the worker submitted (`assembly_webhook_base_url`
  with the job/provider query params preserved).
- Assembly posts `{"transcript_id": "...", "status": "completed"|"error", ...}`.
- On `completed`, the backend must re-enqueue the worker task
  `download_and_transcribe(job_id, resume_token=<transcript_id>,
  resume_provider="youtube")` (resume must route to the strategy name, not
  `assemblyai`; the legacy `"yt-dlp"` alias also resolves).
- On `error`, the backend should fail the job.

Behaviour without a configured webhook base URL is unchanged legacy
poller-retry. When the setting is present, the backend is responsible for
recovering jobs whose webhook never fires (e.g. replay/timeout), since the
worker no longer holds a retry budget for them.

## Celery application name

`celery_app_name` (default `"jumpto"`) is the Celery app identity from
`app/core/config.py`. It prefixes the registered task name
(`<app_name>.transcribe_video`), so it must match every producer that enqueues
jobs over the broker. Change it via `CELERY_APP_NAME` only in coordination
with producers; the default must stay stable.

## SOLID compliance rules

Every new feature and every new piece of code must follow SOLID. These are the
concrete rules the pipeline already follows — keep them, and apply them to new
code:

- **S — Single Responsibility**: one module, one class, one function — one
  reason to change. The Celery task is a scheduling shell; orchestration lives
  in `app/services/`, one concern per module (pipeline, jobs, submissions,
  webhooks, event loop). If a function mixes orchestration with mapping, split
  it into the owning service module.
- **O — Open/Closed**: extend via new modules/classes, not by editing existing
  ones. The provider registry self-registers (a new provider = a new module via
  `register_provider()` + one import line in `_ensure_registered()`); the
  transcript cache stays closed because cache-affecting settings are declared
  by each provider spec via `cache_config_fields`, read by
  `app/storage/cache.py::_cache_namespace` — never a hard-coded list in the
  cache module.
- **L — Liskov Substitution**: subclasses must honour the base-class contract,
  and error classes must not invert their base's meaning. `TranscriptJobPending`
  and `PermanentExternalServiceError` are **siblings** of `ExternalServiceError`
  (all derive from `DomainError`), never subclasses — `except
  ExternalServiceError` means "soft miss, fall through" and must never swallow a
  terminal or pending outcome.
- **I — Interface Segregation**: keep interfaces minimal; declare capabilities,
  don't sniff internals. The strategy exposes `name`, `supports_resume`,
  `supports_webhook`, `fetch()`. Deployment metadata (`uses_cloud`) and
  cache-affecting settings (`cache_config_fields`) live on `TranscriptProviderSpec`,
  not the strategy. `try_provider()` forwards the `webhook_url` only to
  strategies declaring `supports_webhook` — never a hard-coded name list.
- **D — Dependency Inversion**: depend on abstractions, not concretions.
  High-level pipeline code depends on the registry/spec and `JobService`, not
  concrete providers or `BackendClient`. Inject collaborators (settings,
  services, client factories): `YouTubeTranscriptProvider` accepts injected
  `settings`, and leaf services (`media.py`, `build_ydlp_options`) accept an
  optional `settings=` override instead of re-reading the singleton, so tests
  inject fakes instead of monkeypatching module globals. `get_settings()`
  singletons are acceptable for process-wide config but should not leak into
  leaf services that need testability.