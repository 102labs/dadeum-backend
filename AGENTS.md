# Dadeum Backend Agent Guide

## Autonomy

You are an autonomous coding agent. Execute clear implementation tasks to completion without asking for permission. Ask only when the next step is destructive, credential-gated, production-impacting, or materially ambiguous.

## Product Direction

This repository is building the backend side of a short business writing rewrite feature. The system is split into two parts:

- Next.js SaaS app: browser-facing proxy, auth, subscription checks, usage limits, UI, and request signing.
- Lightsail Core: internal rewrite engine, server-to-server request validation, LLM orchestration, LangGraph pipeline, semantic preservation audit, and structured rewrite response.

The privacy rule is strict for plaintext bodies. Fast synchronous requests must not persist source text, rewritten text, diff body, finding body, or raw LLM request/response body by default. Strict asynchronous jobs may persist only encrypted source payloads and encrypted final results with a short TTL. Plaintext bodies, raw LLM request/response bodies, and decrypted values must not be written to logs, analytics, or non-encrypted database columns unless `HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=true` is explicitly enabled for a temporary debugging window; raw LLM request/response bodies and encrypted payload bytes remain excluded even then.

## Current Repository Focus

The current implemented scope is the Lightsail Core service under:

```text
services/humanize-core/
```

Core stack:

```text
Python 3.12
FastAPI
Pydantic / pydantic-settings
LangGraph
SQLite (durable strict job store, AES-GCM encrypted payloads)
Uvicorn
Docker Compose
Caddy
```

LLM providers (`HUMANIZE_MODEL_PROVIDER`): `stub`, `openai`, `anthropic`, `openrouter`.
Only `openrouter` implements every graph stage. See "Provider Capabilities" below.

## Production Operations

The running production Lightsail Core service is deployed on AWS Lightsail Ubuntu.
The source checkout on the server is:

```text
/opt/dadeum/dadeum-backend
```

Use this as the base path when giving production operation commands. The Core
compose project lives under:

```text
/opt/dadeum/dadeum-backend/services/humanize-core
```

Production Docker Compose reads `services/humanize-core/.env`. The compose file
mounts the named Docker volume `humanize-core-data` at `/data` inside the
`humanize-core` container, so persistent Core runtime files should prefer
`/data/...` paths unless a host bind mount is intentionally added.

## Lightsail Core API

Required endpoints:

```text
GET /health
POST /v1/rewrite
GET /v1/rewrite-jobs/{jobId}
DELETE /v1/rewrite-jobs/{jobId}
```

`GET /health` returns:

```json
{
  "status": "ok"
}
```

`POST /v1/rewrite` and `/v1/rewrite-jobs/*` are internal only. They must be called by the Next.js server, not directly by a browser.

## Rewrite Request Contract

Core accepts:

```py
class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    text: str = Field(min_length=1)
    user_intent: str = ""                      # stripped
    rewrite_mode: Literal["fast", "strict"] = "fast"
    tone: Literal["keep", "formal", "friendly"] = "keep"
    protected_terms: list[str] = []            # stripped, empty entries dropped
    max_rounds: int = Field(default=1, ge=1, le=3)
    preserve_formatting: bool = True
```

Next.js should build this Core request by combining the browser-provided text and rewrite controls with internal fields:

```json
{
  "text": "윤문할 원문",
  "user_intent": "",
  "rewrite_mode": "fast",
  "tone": "keep",
  "protected_terms": [],
  "max_rounds": 1,
  "preserve_formatting": true
}
```

Core accepts at most `HUMANIZE_MAX_CHARS` characters per request (default 5,000; longer text returns `422`). Core infers internal rulebook hints from the text itself with a local regex detector; there is no separate LLM detect call.

How the controls are actually used:

- `user_intent`, `tone`, `protected_terms`, `preserve_formatting`: shape the rewrite/style-repair/audit/review prompts and the local preservation checks.
- `rewrite_mode`: chooses delivery only. `fast` runs the graph synchronously inside `POST /v1/rewrite`; `strict` stores an encrypted job, returns `202 Accepted` with a job id, and the worker runs the **same graph**. There is no separate "strict" prompt or extra review depth.
- `max_rounds`: validated (1-3) but **not used** by the graph. Round count in `usage.rounds` is `1 + style-gate repair rounds` and is controlled by `HUMANIZE_STYLE_GATE_MAX_ROUNDS`.

The Next.js server polls `GET /v1/rewrite-jobs/{jobId}` for strict jobs.

## Rewrite Response Contract

Fast synchronous Core requests and completed strict jobs return:

```json
{
  "revisedText": "윤문 결과",
  "changes": [
    {
      "original": "원문 일부",
      "revised": "수정된 표현",
      "reason": "변경 이유",
      "type": "clarity",
      "riskLevel": "low"
    }
  ],
  "summary": ["전체 변경 요약"],
  "warnings": [],
  "usage": {
    "inputTokens": 1200,
    "outputTokens": 900,
    "latencyMs": 8420,
    "rounds": 1
  }
}
```

`changes[].original` and `changes[].revised` are guaranteed to be exact substrings of the request text and `revisedText`: `finalize` always rebuilds the list from a sequence diff of the two, attaching model reasons by location and asking the provider to explain the rest. At most 30 display changes are emitted, in document order; when there are more, a 31st entry with empty snippets says how many were left out (the overflow note starts with `세부 변경 구간이`, which the eval harness keys on). `changes[].type` is one of `clarity | tone | concision | structure | grammar | meaning`; `riskLevel` is `low | medium | high`.

Strict `POST /v1/rewrite` requests return `202 Accepted` with:

```json
{
  "jobId": "uuid",
  "requestId": "req_...",
  "status": "queued",
  "pollAfterMs": 1000
}
```

`GET /v1/rewrite-jobs/{jobId}` and `DELETE /v1/rewrite-jobs/{jobId}` return the job status record:

```json
{
  "jobId": "uuid",
  "requestId": "req_...",
  "status": "queued | running | succeeded | failed | cancelled | expired",
  "rewriteMode": "strict",
  "textLength": 1200,
  "attempts": 1,
  "maxAttempts": 2,
  "createdAt": "2026-07-13T00:00:00Z",
  "expiresAt": "2026-07-14T00:00:00Z",
  "startedAt": null,
  "completedAt": null,
  "latencyMs": null,
  "errorCode": null,
  "result": null
}
```

`result` is populated only when `status == "succeeded"`. `errorCode` is one of `input_limit_exceeded`, `model_not_configured`, `invalid_model_response`, `internal_error`. `invalid_model_response` and `internal_error` are retried up to `maxAttempts`; the other two fail immediately. Unknown job ids return `404`. If job storage cannot be opened, strict endpoints return `503`.

Fast-mode error mapping: `422` input limit, `503` provider not configured, `502` invalid structured model response.

## Server-to-Server Security

Every `/v1/rewrite` request must validate these headers:

```text
X-Core-Api-Key
X-Request-Id
X-Timestamp
X-Body-SHA256
X-Signature
```

Validation rules:

```text
1. All five headers are present and non-empty.
2. X-Core-Api-Key matches HUMANIZE_CORE_API_KEY (constant-time compare).
3. X-Timestamp is within 300 seconds of server time. Accepted formats:
   unix seconds, unix milliseconds (> 10_000_000_000), or ISO-8601.
4. X-Body-SHA256 matches sha256(rawBody) (hex, case-insensitive).
   For GET/DELETE the body is empty, so hash the empty byte string.
5. X-Signature matches HMAC-SHA256 over:
   `${timestamp}.${requestId}.${bodyHash}`
   using HUMANIZE_CORE_SIGNING_SECRET (hex, case-insensitive).
```

The same headers are required on `/v1/rewrite-jobs/*`. Authentication failures return `401 Unauthorized` with no detail. Auth runs before body validation, so an invalid body with a bad signature is a `401`, not a `422`.

Do not enable browser CORS for Core.

## LangGraph Pipeline

The graph (`humanize_core/graph.py`) is:

```text
prepare -> rewrite -> style_gate -> audit -> (review) -> finalize
```

`fast` and `strict` run the identical graph; only delivery differs.

Responsibilities:

- `prepare`: enforce `HUMANIZE_MAX_CHARS`; run the local regex detector (`im_not_ai/audit.py::local_detect`, ~50 rules with S1/S2/S3 severity) on the source; build compact rulebook hints (max 24 rules, up to 3 short match samples each, never spans overlapping numbers/quotes/protected terms). No LLM call.
- `rewrite`: one structured LLM call. Text at or above `HUMANIZE_CHUNK_MIN_CHARS` (default 1,000) is split at sentence boundaries into chunks of about `HUMANIZE_CHUNK_TARGET_CHARS` and rewritten in parallel, then reassembled with the original separators. The full rulebook (`im_not_ai/resources/strict-rules.md`, bodies stripped) sits in the static system prompt; detected rules travel in the user payload with their full rule cards.
- `style_gate`: re-run the local detector on the draft. If any S1 remains, or S2 count is at or above `HUMANIZE_STYLE_GATE_S2_THRESHOLD` (3), or the text was chunked (one transition-smoothing pass), call the provider's `style_repair` up to `HUMANIZE_STYLE_GATE_MAX_ROUNDS` (2) times. A repair is discarded if it adds completion warnings, increases preservation damage, or worsens the severity score. Skipped when the provider has no `style_repair`.
- `audit`: local checks first: completion (empty, too short vs. original, low sentence/paragraph coverage, cut off mid-sentence) and exact preservation counts (protected terms, quotes, URLs, emails, code spans, dates, numbers/units) must not go down or up versus the source. Then, if the provider has `audit`, a model audit for harmful meaning changes is merged in. Any completion warning forces `fail`. `riskLevel` per change is set by the explain step in `finalize`, not by the audit.
- `review` (conditional): entered when audit status is `fail` or `conditional_pass`, or any flagged edit needs repair (this is the pre-2026-10-09 whole-text review; the sentence-scoped segment review from `main` was reverted on `stable-v1` because it let low-severity meaning changes ship). If the provider has `review`, the model receives the full draft plus the audit record and returns the complete repaired passage; the output is re-checked locally and replaced by the local repair path if it truncated the text, damaged more preserved values, or reverted the draft wholesale to the source (draft change rate >= 5% coming back <= 1%). Without a provider `review`, the local repair path restores flagged values/sentences from the source. Style-only restores that would reintroduce an S1 violation are kept as the gated draft with a warning.
- `finalize`: merge warnings (audit warnings and flagged-edit issues when no review ran; review warnings, final audit warnings, and blocking issues otherwise), build the display `changes` from a sequence diff of source vs. final text (every real edit is listed, snippets are cut at word boundaries, and a model-authored change is matched to at most one diff group by location, only when both its snippets still match), then send every diff group to the provider's `explain_changes` (if any) with the matched model reason as `draft_reason` hint. The explainer writes the user-facing `reason`/`type`/`riskLevel` per group and the response `summary`. An explain failure keeps the model/generic reasons and the stage summaries, and never fails the request. Review/style-gate stages do not append internal notes to `summary`. Tokens are summed across rewrite + style gate + audit + review + explain, `usage.rounds = 1 + style-gate rounds`.

Every stage logs `graph.stage.started / succeeded / failed` with durations and counts.

## Provider Capabilities

| provider     | rewrite | style_repair | model audit | model review | explain changes | notes |
|--------------|---------|--------------|-------------|--------------|-----------------|-------|
| `stub`       | local   | no           | local only  | local only   | no              | deterministic; used by tests |
| `openai`     | yes     | no           | local only  | local only   | no              | Responses API, strict JSON Schema |
| `anthropic`  | yes     | no           | local only  | local only   | no              | Messages API, JSON parsed from text; non-JSON falls back to raw text |
| `openrouter` | yes     | yes          | yes         | yes          | yes             | Chat Completions `response_format: json_schema`, `require_parameters: true` |

Production is expected to run `openrouter`. With `openai` or `anthropic` the style gate is skipped, audit/review are local rule checks only, and unexplained diff groups keep a generic reason.

The structured-output schemas sent to the model for rewrite, style repair, audit, and review are the full `RewriteResult` / `AuditResult` / `StrictReviewResult` models in `im_not_ai/schemas.py` (as before 2026-10-09; the model fills bookkeeping fields such as `qualityLevel` and `rollbackRequired`, code overwrites token counts). Only the explain call uses a slim schema (`ChangeExplanationOutput`). `Change` field descriptions (reason format, `type` meanings, `riskLevel` meanings) travel inside the JSON schema and are repeated in the explain prompt as `changes_contract`.

OpenRouter model selection (`llm.py::OpenRouterRewriteLLM`):

- rewrite and style_repair: `[HUMANIZE_MODEL_NAME if set and not "stub" else HUMANIZE_REWRITE_MODEL_NAME, HUMANIZE_REWRITE_FALLBACK_MODEL_NAME]`
- audit: `[HUMANIZE_STRICT_AUDIT_MODEL_NAME, primary rewrite model]`
- review: `[HUMANIZE_STRICT_REVIEW_MODEL_NAME, primary rewrite model]`
- explain changes: `[HUMANIZE_EXPLAIN_MODEL_NAME if set, primary rewrite model]`

Models in each list are tried in order; any exception moves to the next one. When all fail the call raises `LLMResponseError`. Every call uses `max_tokens=20000` and no temperature.

LLM call budget per request with `openrouter`: 1 rewrite (or N parallel chunk calls) + 0-2 style repairs + 1 audit + 0-1 review + 1 explain (skipped only when the text did not change). Fast mode runs all of this synchronously; there is no request timeout in Core.

## Environment Variables

All settings live in `humanize_core/config.py` (`Settings`, loaded from env and `.env`). Defaults shown are the code defaults.

Provider and models:

```text
HUMANIZE_MODEL_PROVIDER=stub                     # stub | openai | anthropic | openrouter
HUMANIZE_MODEL_NAME=stub                         # openai/anthropic model; for openrouter overrides the rewrite primary when not "stub"
OPENAI_API_KEY                                   # required for provider=openai
ANTHROPIC_API_KEY                                # required for provider=anthropic
OPENROUTER_API_KEY                               # required for provider=openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_TITLE=Dadeum Humanize Core        # sent as X-Title
OPENROUTER_SITE_URL                              # optional, sent as HTTP-Referer
HUMANIZE_REWRITE_MODEL_NAME=openai/gpt-5-mini    # alias: HUMANIZE_FAST_MODEL_NAME
HUMANIZE_REWRITE_FALLBACK_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_AUDIT_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_REVIEW_MODEL_NAME=openai/gpt-5.4-mini
HUMANIZE_EXPLAIN_MODEL_NAME=                     # optional; change-explanation model, falls back to the rewrite primary
HUMANIZE_EVAL_JUDGE_MODEL_NAME=                  # eval only (scripts/eval_golden.py --judge), not used by the service
```

Security and limits:

```text
HUMANIZE_CORE_API_KEY=                           # empty -> every request is 401
HUMANIZE_CORE_SIGNING_SECRET=                    # empty -> every request is 401
HUMANIZE_MAX_CHARS=5000
HUMANIZE_CHUNK_MIN_CHARS=1000                    # text at/above this is chunked
HUMANIZE_CHUNK_TARGET_CHARS=1000
HUMANIZE_STYLE_GATE_MAX_ROUNDS=2                 # 0 disables the style gate
HUMANIZE_STYLE_GATE_S2_THRESHOLD=3
```

Strict job store and worker:

```text
HUMANIZE_JOB_STORE_PATH=humanize_jobs.sqlite3    # compose sets /data/humanize_jobs.sqlite3
HUMANIZE_JOB_ENCRYPTION_KEY                      # 32-byte hex or base64url; falls back to sha256(HUMANIZE_CORE_SIGNING_SECRET)
HUMANIZE_JOB_WORKER_ENABLED=true
HUMANIZE_JOB_POLL_INTERVAL_SECONDS=1.0           # also drives pollAfterMs (min 250)
HUMANIZE_JOB_LOCK_SECONDS=600                    # stale running jobs are reclaimed after this
HUMANIZE_JOB_RETENTION_SECONDS=86400             # TTL for the job row and encrypted result
HUMANIZE_JOB_MAX_ATTEMPTS=2
```

Debug logging:

```text
HUMANIZE_DEBUG_LOG_ENABLED=true
HUMANIZE_DEBUG_LOG_DIR=~/.dadeum/humanize-core/logs   # production .env sets /data/humanize-core/logs
HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=false
```

The signature timestamp tolerance (300 seconds) is a code constant, not an env var.

The worker runs in-process inside the Uvicorn app (single asyncio task started in the FastAPI lifespan). The job store is SQLite in WAL mode; running more than one Core process against the same file is not designed for.

OpenAI calls use the Responses API with strict JSON Schema structured output, not legacy `json_object` mode. The model generates only `revisedText`, `changes`, and `summary`; Core fills token usage from API response metadata.

## Debug Logging

Core keeps step logs for rewrite debugging. The production Docker location is:

```text
/data/humanize-core/logs/YYYY-MM-DD.log
```

These logs record request/job ids, graph step names, per-step durations, statuses, token counts, warning/change counts, retry decisions, and error codes. Fast requests log `api.rewrite.accepted / succeeded / failed`; strict jobs add `job.enqueued`, `job.claimed`, `job.processing.started`, `job.payload.loaded`, `job.succeeded`, `job.failed`, `job.cancelled`. Status polls are not logged.

Redaction is **key-name based** (`debug_log.py`): any detail key containing `text`, `source`, `revised`, `change`, `summary`, `warning`, `finding`, `diff`, `intent`, `protected`, `term`, `prompt`, `raw`, `body`, `payload`, `result`, `original`, `cipher`, `nonce` is replaced with `[REDACTED length=N]` unless the key is in the safe list or ends with a metric suffix (`_count`, `_length`, `_ms`, `_tokens`, ...). Callers deliberately pass plaintext under those keys (for example `source_text`, `revised_text`, `changes`) so the plaintext flag can switch them on. **When adding a new detail key that carries body text, make sure its name hits the blocklist**; a key like `draft` would leak. Any string value longer than 256 chars is also redacted regardless of key.

For a temporary explicit debugging window, `HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=true` may be enabled to include source text, rewritten text, summaries, warnings, and change/audit details in the text log. Turn it off after debugging, and never log raw LLM request/response bodies or encrypted payload bytes.

When developing core features, use the local `log` skill as the workflow reminder: add operational logs for important pipeline stages, document where they are stored, and add tests proving plaintext is redacted by default and included only when the explicit debug flag is enabled.

## Next.js SaaS Scope

The Next.js SaaS app is responsible for:

- `/ai` rewrite UI.
- `POST /api/rewrite`.
- Supabase session validation.
- Subscription lookup.
- Free/pro plan resolution.
- Per-request character limit checks.
- Daily/monthly usage checks.
- Creating `rewrite_usage_events` without plaintext text bodies.
- Signing and forwarding fast requests to Core.
- Creating and polling strict rewrite jobs for asynchronous Core processing.
- Updating usage events to `succeeded` or `failed`.

Browser request type:

```ts
type RewriteClientRequest = {
  text: string;
  user_intent?: string;
  rewrite_mode?: "fast" | "strict";
  tone?: "keep" | "formal" | "friendly";
  protected_terms?: string[];
  max_rounds?: number;
  preserve_formatting?: boolean;
};
```

Plan limits:

```ts
const rewritePlanLimits = {
  free: {
    maxCharsPerRequest: 3000,
    dailyRequests: 5,
    monthlyRequests: 30,
  },
  pro: {
    maxCharsPerRequest: 5000,
    dailyRequests: 100,
    monthlyRequests: 1000,
  },
};
```

Active subscription means `pro`; no active subscription means `free`.

## Usage Event Table

The SaaS app should create a plaintext-body-free usage table:

```sql
create table public.rewrite_usage_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.users(id) on delete cascade,
  plan text not null check (plan in ('free', 'pro')),
  request_id text not null unique,
  text_length integer not null,
  status text not null check (status in ('pending', 'succeeded', 'failed')),
  latency_ms integer,
  error_code text,
  created_at timestamptz not null default now()
);
```

Never store in plaintext:

```text
원문
윤문 결과
diff 본문
finding 본문
LLM raw request/response body
```

Strict async job storage may contain encrypted source payloads and encrypted final results only for active processing and short result retrieval. Encrypted source payloads must be purged after terminal success or final failure. Encrypted results must expire by TTL or user deletion.

## Test Requirements

Lightsail Core tests (`tests/test_api.py` and `tests/test_eval_golden.py`, 121 tests, all on the `stub` provider or fake LLM doubles) cover:

- `/health` returns ok.
- Missing or invalid API key returns `401`.
- Missing or invalid HMAC returns `401`.
- Expired timestamp returns `401`.
- Body hash mismatch returns `401`.
- Invalid enum and unknown/legacy fields return `422` after auth passes.
- Valid fast request returns structured `RewriteResponse`.
- Strict request returns `202 Accepted` with a job id; `max_rounds` is ignored.
- Strict job status returns result only after completion; store survives lifespan restart.
- Strict job storage does not contain plaintext source text or rewritten text.
- Logs do not include source text or rewritten text by default and do include them with the plaintext flag.
- Prompt contents (rulebook, hints, tone guidance, completion contract).
- Local detector rules and false-positive guards.
- Graph routing: clean audit skips review; conditional/fail or any repair flag routes to the whole-text review; truncated, value-damaging, or wholesale-reverting review output falls back to local repair; style gate repairs and regressions; chunk split/reassembly.
- Display changes: reasons attach to diff groups by location (a snippet absent from the final text never contributes its reason, each model change is used once), snippets end on word boundaries, every group is sent to `explain_changes` with the model reason as hint and the explainer's summary replaces the stage summaries, an explain failure keeps the generic reason; review leaves no internal notes in `summary`.
- Model-facing schemas carry no internal bookkeeping fields and do carry the `Change` field descriptions.
- Provider request shapes for OpenAI Responses, Anthropic Messages, and OpenRouter Chat Completions (schema normalisation, usage extraction, no temperature).
- Eval harness (`scripts/eval_golden.py`): register detection, per-case expectations, change-list quality, multi-run aggregation, baseline diff regression rules, stage telemetry capture, judge payload/schema, golden-set field validity, stub end-to-end run.

Add to this list when you add behavior. Keep tests provider-free (no network).

Next.js tests should cover:

- Unauthenticated request returns `401`.
- Free users above 3,000 chars are blocked before Core call.
- Free users above daily or monthly limits receive `429`.
- Pro users at or below 5,000 chars call Core.
- Core success updates usage event to `succeeded`.
- Core failure updates usage event to `failed`.
- Logs and non-encrypted DB columns do not contain source text or rewrite result.

## V1 Exclusions

Do not implement these unless explicitly requested:

```text
SSE streaming
Team plan
API key issuance
plaintext body history storage
```

## Verification

For Core changes, run:

```bash
cd services/humanize-core
uv run --python 3.12 --with '.[dev]' pytest -q
```

For prompt, rulebook, or detector changes, also run the golden-set eval before and after and compare (`scripts/eval_golden.py --baseline ...`; add `--judge --pairwise` for LLM-judged naturalness, see the service README). Regression = residual S1 or preservation damage up, expectations down, or judge overall down 0.5+.

Report any skipped verification clearly.

## Repository Layout Notes

```text
services/humanize-core/
  humanize_core/
    api.py          FastAPI app, routes, error mapping
    security.py     header/HMAC verification
    config.py       Settings (all env vars)
    schemas.py      public request/response models
    graph.py        LangGraph pipeline (6 stages) + local audit/repair helpers
    llm.py          provider adapters (stub / openai / anthropic / openrouter)
    jobs.py         SQLite job store, AES-GCM cipher, in-process worker
    debug_log.py    redacting daily text logger
    diff.py         display-safe change rebuilding
    im_not_ai/
      audit.py         local regex detector (local_detect), severity scoring
      preservation.py  exact preserve target extraction
      prompts.py       all system/user prompts
      resources.py     rulebook loader, compact rulebook, rule cards
      resources/strict-rules.md   the active rulebook
      schemas.py       internal structured-output models
  scripts/
    eval_golden.py            golden-set eval harness (pattern metrics, expectations, change-list quality, LLM judge, stage telemetry)
    strict_rewrite_probe.py   prepare->rewrite only, local diagnostics
  evals/golden_set.json       36 fixed Korean cases with per-case expectations
  tests/test_api.py
  tests/test_eval_golden.py   eval harness tests
```

Not used by the runtime graph (kept for reference/tests only): `im_not_ai/metrics.py`, `im_not_ai/metrics_v2.py`, `im_not_ai/baseline.json`, `im_not_ai/baseline_v2_diff.json`, `im_not_ai/resources/strict-rules-old.md`, and `llm.py::_system_prompt / _user_prompt`. Do not extend them; extend `graph.py` / `prompts.py` instead.

The Dockerfile installs from `pyproject.toml` with `pip`, not from `uv.lock`, so production dependency versions float within the declared ranges.
