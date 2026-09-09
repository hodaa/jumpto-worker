# JumpTo Worker Performance Audit

**Date:** 2026-09-09  
**Reviewed revision:** `2b020312f721182641c708a385564fe9cc703a0d`  
**Scope:** worker execution path, provider orchestration, HTTP/Redis usage, Celery configuration, and transcript construction.
**Implementation status:** the recommendations in this report were implemented in the follow-up changes on this branch; the validation section below reflects the updated test suite.

## Executive summary

The worker is small and its local Python transforms are not currently the main risk. The dominant cost is remote work: provider calls, YouTube/yt-dlp activity, Assembly polling, and backend round trips. The most important issues are in orchestration rather than algorithms:

1. A resumed cloud job still tries providers that precede the original provider in the chain on every retry. This adds calls and can cause the worker to complete from a different provider while leaving the original cloud job running.
2. Permanent provider failures are swallowed as ordinary misses. Invalid credentials or account errors can therefore fan out into several more provider calls and a yt-dlp attempt.
3. The transcript cache only runs inside the terminal yt-dlp strategy. Successful cloud-provider jobs bypass it, and concurrent duplicate jobs have no single-flight lock.
4. Assembly.ai polling occupies a Celery slot and can issue up to 600 one-second-spaced status requests for one job.

These are **high-confidence code-review findings**. No live backend, broker, Redis, or provider endpoints were available, so this is not a production latency/SLO measurement.

## Implemented in this change set

- Resume tokens now start at their originating strategy instead of replaying earlier providers.
- Permanent provider/account failures now stop the chain instead of triggering fallback traffic.
- Successful results are cached at the orchestration boundary with versioned settings-aware keys and a Redis single-flight lock.
- Assembly.ai submission/status checks now use Celery’s resumable retry path rather than a 600-request polling loop.
- HTTP clients are pooled per persistent worker event loop, and the Celery task loop is reused across tasks.
- Backend retries use bounded jittered backoff and honor numeric `Retry-After` values.
- Transcript submission construction is single-pass, deployment concurrency uses `CELERY_WORKER_CONCURRENCY`, and the hard time limit includes a configurable cleanup margin.

## Current execution path

For each task, the worker generally performs:

1. Create a new `BackendClient` and fetch the job.
2. Advance a pending job with a second backend request.
3. Build the configured provider chain and try providers sequentially.
4. Build a potentially large word payload.
5. Store the transcript and complete the job with two more backend requests.
6. Close the backend HTTP client.

The default Celery deployment runs eight prefork workers, with `worker_prefetch_multiplier=1` and late acknowledgements. The local yt-dlp strategy uses `asyncio.to_thread` for blocking work; cloud strategies use short-lived `httpx.AsyncClient` instances.

## Findings

### P1 — Resumed jobs retry providers before the originating provider

**Evidence:** `app/tasks/transcription.py:94-96`.

A resume token is passed only to the provider whose name matches `resume_provider`, but the loop still invokes every provider before it. For example, a Supadata retry with a chain of `transcriptfetch -> supadata -> ...` calls TranscriptFetch again before checking the existing Supadata job. The current test at `tests/unit/test_pipeline.py:307-328` verifies this behavior rather than the intended direct routing.

**Impact:** every cloud retry can add one or more unnecessary network calls and their full timeout. If an earlier provider succeeds, the original async job is orphaned and the result may come from a different provider. This is both a latency/cost issue and a correctness risk.

**Recommendation:** when `resume_token` is present, route directly to `resume_provider` (or slice the chain at that provider and do not call earlier providers). Validate that the provider exists and is resumable; fail clearly if it does not. Add a test asserting that earlier providers are not called.

### P1 — Permanent provider errors are treated as fall-through misses

**Evidence:** `app/tasks/transcription.py:124-143` catches every `Exception`. `TranscriptFetchPermanentError`, `SupadataPermanentError`, and `VidWordsPermanentError` all inherit from `ExternalServiceError`, so invalid credentials, insufficient credits, and invalid requests continue through the chain.

**Impact:** a configuration/account failure can produce several sequential API calls and eventually an expensive yt-dlp attempt. At eight workers this can amplify a bad deployment into avoidable provider traffic and longer queue times. It also obscures the root cause in the final user-facing failure.

**Recommendation:** distinguish soft misses, retryable/transient errors, and permanent errors. Fall through only for a soft miss or explicitly retryable per-video failure; propagate permanent account/configuration errors immediately. If fallback is intentionally retained, add a circuit breaker or cooldown for repeated provider-wide failures.

### P1 — Cache coverage is narrow and there is no duplicate-work protection

**Evidence:** the cache is consulted and populated only by `YtDlpTranscriptStrategy` in `app/providers/local.py:48-72`. The cloud strategies do not read or write it. `app/providers/cache.py:140-148` provides a process-cached Redis client, but no per-video lock or in-flight coordination.

**Impact:** with the normal cloud-first configuration, repeated requests for the same video repeat the cloud call and can consume credits. Two jobs arriving together both miss the cache and independently perform the expensive provider/yt-dlp work. The existing cache helps only jobs that reach yt-dlp and only after a previous job has fully completed.

**Recommendation:** put a result cache at the orchestration boundary, keyed by video ID plus language/mode and a cache schema version. Decide explicitly whether provider provenance is part of the key. Add a short-lived Redis single-flight lock (`SET ... NX EX`) so only one job populates a missing key; followers should wait briefly and re-check. Consider a short negative-cache TTL for confirmed “no transcript” results.

### P1 — Assembly polling can consume a worker slot for ten minutes

**Evidence:** `app/providers/transcript.py:419-438` sleeps one second and performs up to `_MAX_POLL_ATTEMPTS = 600` status requests.

**Impact:** one slow Assembly job can occupy one of eight Celery slots and generate up to 600 status requests. Eight simultaneous slow jobs can tie up the whole worker. The polling is asynchronous, so it does not busy-spin the CPU, but it still reduces throughput and increases provider traffic.

**Recommendation:** prefer an async-job/resume model like the Supadata path, re-queueing the task with provider-aware backoff. If polling must remain, honor `Retry-After`, use exponential backoff capped at a reasonable interval, and instrument poll count and elapsed time.

### P2 — HTTP connection pools are discarded at provider-call boundaries

**Evidence:** `httpx.AsyncClient` is created in each `fetch` call in `app/providers/transcriptfetch.py:136`, `app/providers/supadata.py:103`, and `app/providers/vidwords.py:95`. Assembly creates another one in `app/providers/transcript.py:368`. The backend client reuses its pool only within one task (`app/client/backend.py:23-30`).

**Impact:** cloud calls across jobs cannot reuse TCP/TLS connections. This adds handshake latency and connection churn under load. Supadata does reuse one client for its transcript and metadata requests, which is good, but that pool is still discarded after the call.

**Recommendation:** use a provider client lifecycle that is long-lived within a worker process/event loop, or use a pooled synchronous client if retaining Celery’s synchronous task entry point. Do not share an `AsyncClient` across unrelated event loops. Measure connection reuse and TLS handshake time before and after.

### P2 — Backend retries are immediate and do not honor server backoff

**Evidence:** `app/client/backend.py:80-110` retries network/5xx failures up to three times with no delay or jitter and does not inspect `Retry-After`.

**Impact:** a backend incident creates three immediate requests per operation, increasing load exactly when the backend is unhealthy. Since POST operations are retried as well, the backend endpoints must also be idempotent to avoid duplicate side effects after an ambiguous network failure.

**Recommendation:** add bounded exponential backoff with jitter, honor `Retry-After`, and retry only operations that are safe/idempotent or carry an idempotency key. Emit a retry counter and final failure reason.

### P2 — `asyncio.run` plus `to_thread` creates avoidable per-task runtime churn

**Evidence:** the Celery task calls `asyncio.run` in `app/tasks/transcription.py:194-195`. The local strategy dispatches metadata, cache, caption, and download work through `asyncio.to_thread` in `app/providers/local.py:54-63` and `app/providers/transcript.py:164-166,365`.

**Impact:** each task creates and closes an event loop. The default executor used by `to_thread` is also created/shut down with that loop, causing thread-pool churn and preventing reuse across tasks. This is smaller than provider/network latency, but it is paid on every local-pipeline job and makes concurrency harder to reason about.

**Recommendation:** either keep the task synchronous and run blocking provider operations in the Celery process/thread model, or use a persistent async worker lifecycle with a long-lived loop/executor. Avoid sharing async clients across the current per-task loops.

### P2 — Transcript construction makes multiple large in-memory copies

**Evidence:** `_build_submission` first builds `_words` and then `words` in `app/tasks/transcription.py:159-170`. `BackendClient.store_transcript` then creates another list of dictionaries in `app/client/backend.py:43-57` before JSON encoding.

**Measured synthetic result:** on Python 3.11 with 50,000 words, `_build_submission` took about **221 ms / 12.6 MiB peak**, followed by payload materialization at **42 ms / 9.2 MiB peak**. At 100,000 words, the combined submission build alone took about **388 ms / 25.3 MiB peak**. These are microbenchmarks, not production measurements, but they show the scaling shape.

**Impact:** long videos increase worker RSS and JSON serialization time; multiple concurrent long jobs can create memory pressure. The transcript is also sent as one large request, so backend parsing and request buffering scale with it.

**Recommendation:** build normalized `TranscriptWordData` in one pass, avoid the intermediate tuple list, and profile JSON encoding separately. If long videos are expected, add a size limit and a chunked/batched transcript endpoint (or compression) rather than one unbounded request.

### P2 — Configured concurrency is overridden by deployment commands

**Evidence:** `celery_app.py:51` reads `settings.celery_worker_concurrency`, but `Dockerfile:28`, `docker-compose.yml:21-27`, and the README hard-code `--concurrency=8`.

**Impact:** changing `CELERY_WORKER_CONCURRENCY` has no effect in the documented Docker deployment. Eight prefork processes may be too many for CPU/RAM when yt-dlp, ffmpeg, or long transcripts are active, and too few when the workload is mostly short cloud calls.

**Recommendation:** remove the hard-coded CLI override and use the setting, or template the deployment command. Benchmark separate cloud-heavy and yt-dlp-heavy workloads while recording throughput, p95 latency, RSS, CPU, network bandwidth, and provider rate-limit responses.

### P2 — The application timeout and Celery hard time limit have no cleanup margin

**Evidence:** `asyncio.wait_for(..., timeout=settings.job_timeout_seconds)` is used at `app/tasks/transcription.py:50-52`, while Celery sets `task_time_limit=settings.job_timeout_seconds` at `app/tasks/celery_app.py:46`.

**Impact:** a job that consumes its full application budget can be hard-killed at essentially the same time as its cleanup/failure reporting path. This can leave temporary work or an indeterminate backend job state.

**Recommendation:** reserve a margin between the soft application timeout and the hard Celery limit, handle `SoftTimeLimitExceeded`, and ensure temporary files/remote job state are cleaned up or reconciled.

## Validation performed

- `175` unit tests passed in `0.62 s` in the audit environment.
- Coverage run passed the configured 80% gate at **86.75%**.
- Synthetic transcript benchmarks were run with generated in-memory data; they do not represent network latency or production video distributions.
- No live backend, Redis, Celery broker, YouTube, TranscriptFetch, Supadata, VidWords, or Assembly.ai load test was run. Real provider latency, rate limits, cache hit rate, queue wait, memory per worker, and duplicate-job frequency remain unknown.

## Recommended implementation order

1. Fix direct resume routing and add tests for provider call counts.
2. Stop fall-through on permanent provider/account errors; add explicit error classes/metrics.
3. Add orchestration-level caching plus Redis single-flight protection.
4. Replace or back off Assembly polling and add provider latency/poll metrics.
5. Remove hard-coded concurrency, add timeout margin, and run a representative load test.
6. Optimize large payload construction only after measuring long-video memory and backend request time.

## Metrics needed for a production baseline

Record at least: queue wait time, end-to-end task duration, backend request duration/status/retry count, provider duration/status, provider fallback count, cache hit/miss/lock wait, transcript word count and payload bytes, Assembly poll count, task retry count, worker RSS/CPU, and completion/failure rate by provider. Break these down by video duration and provider so concurrency and timeout changes can be evaluated against p50/p95/p99 rather than averages.
