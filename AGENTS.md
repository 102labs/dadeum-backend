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
Pydantic
LangGraph
Uvicorn
Docker Compose
Caddy
```

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
    text: str
    user_intent: str = ""
    rewrite_mode: Literal["fast", "strict"] = "fast"
    tone: Literal["keep", "formal", "friendly"] = "keep"
    protected_terms: list[str] = []
    max_rounds: int = 1
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

Core accepts at most 5,000 characters per request. Core infers any internal genre hints from the text itself. `user_intent`, `rewrite_mode`, `tone`, `protected_terms`, `max_rounds`, and `preserve_formatting` are request controls and must shape rewrite strength, tone, preservation, review depth, and formatting behavior. Fast mode returns synchronously from `POST /v1/rewrite`. Strict mode is durable asynchronous: `POST /v1/rewrite` returns `202 Accepted` with a job id, and the Next.js server polls `GET /v1/rewrite-jobs/{jobId}`.

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

Strict `POST /v1/rewrite` requests return `202 Accepted` with:

```json
{
  "jobId": "uuid",
  "requestId": "req_...",
  "status": "queued",
  "pollAfterMs": 1000
}
```

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
1. X-Core-Api-Key matches HUMANIZE_CORE_API_KEY.
2. X-Timestamp is within the configured 5 minute tolerance.
3. X-Body-SHA256 matches sha256(rawJsonBody).
4. X-Signature matches HMAC-SHA256 over:
   `${timestamp}.${requestId}.${bodyHash}`
   using HUMANIZE_CORE_SIGNING_SECRET.
```

Authentication failures return `401 Unauthorized`.

Do not enable browser CORS for Core.

## LangGraph Pipeline

The v1 graph flow is:

```text
prepare -> rewrite -> audit -> finalize
```

Responsibilities:

- `prepare`: enforce maximum input length, split/analyze input as needed, and infer internal genre hints from the text.
- `rewrite`: map `user_intent`, `rewrite_mode`, `tone`, and `preserve_formatting` to model instructions; request structured output from the LLM.
- `audit`: check numbers, dates, proper nouns, and protected terms for preservation; add warnings and high risk flags when needed.
- `finalize`: return `revisedText`, `changes`, `summary`, `warnings`, and `usage`.

## Environment Variables

Lightsail Core uses:

```text
OPENAI_API_KEY or ANTHROPIC_API_KEY
HUMANIZE_MODEL_PROVIDER
HUMANIZE_MODEL_NAME
HUMANIZE_CORE_API_KEY
HUMANIZE_CORE_SIGNING_SECRET
HUMANIZE_MAX_CHARS=5000
HUMANIZE_JOB_STORE_PATH
HUMANIZE_JOB_ENCRYPTION_KEY
HUMANIZE_JOB_RETENTION_SECONDS
HUMANIZE_DEBUG_LOG_ENABLED=true
HUMANIZE_DEBUG_LOG_DIR=/data/humanize-core/logs
HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=false
```

Local tests may use:

```text
HUMANIZE_MODEL_PROVIDER=stub
HUMANIZE_MODEL_NAME=stub
```

OpenAI production calls should use the Responses API with strict JSON Schema
structured output, not legacy `json_object` mode. The model should generate only
`revisedText`, `changes`, and `summary`; Core should fill token usage from API
response metadata.

## Debug Logging

Core keeps step logs for rewrite debugging. The production Docker location is:

```text
/data/humanize-core/logs/YYYY-MM-DD.log
```

These logs should make asynchronous job behavior debuggable by recording request/job ids, graph step names, per-step durations, statuses, token counts, warning/change counts, retry decisions, and error codes.

Default logging must redact plaintext source text, rewritten text, diff bodies, finding bodies, protected term values, user intent text, prompts, raw LLM request/response bodies, encrypted payload bytes, and decrypted job payload/result values. Log lengths/counts/statuses instead.

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

Lightsail Core tests must cover:

- `/health` returns ok.
- Missing or invalid API key returns `401`.
- Missing or invalid HMAC returns `401`.
- Expired timestamp returns `401`.
- Body hash mismatch returns `401`.
- Invalid enum returns `422`.
- Valid fast request returns structured `RewriteResponse`.
- Strict request returns `202 Accepted` with a job id.
- Strict job status returns result only after completion.
- Strict job storage does not contain plaintext source text or rewritten text.
- Logs do not include source text or rewritten text.

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

Report any skipped verification clearly.
