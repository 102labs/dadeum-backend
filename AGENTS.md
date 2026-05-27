# Dadeum Backend Agent Guide

## Autonomy

You are an autonomous coding agent. Execute clear implementation tasks to completion without asking for permission. Ask only when the next step is destructive, credential-gated, production-impacting, or materially ambiguous.

## Product Direction

This repository is building the backend side of a short business writing rewrite feature. The system is split into two parts:

- Next.js SaaS app: browser-facing proxy, auth, subscription checks, usage limits, UI, and request signing.
- Lightsail Core: internal rewrite engine, server-to-server request validation, LLM orchestration, LangGraph pipeline, semantic preservation audit, and structured rewrite response.

The v1 privacy rule is strict: do not persist source text, rewritten text, diff body, finding body, or raw LLM request/response body.

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

## Lightsail Core API

Required endpoints:

```text
GET /health
POST /v1/rewrite
```

`GET /health` returns:

```json
{
  "status": "ok"
}
```

`POST /v1/rewrite` is internal only. It must be called by the Next.js server, not directly by a browser.

## Rewrite Request Contract

Core accepts:

```py
class RewriteRequest(BaseModel):
    text: str
    document_type: Literal[
        "auto",
        "report",
        "formal",
        "email",
        "proposal",
        "meeting_notes",
        "blog",
        "column",
    ]
    intensity: Literal["conservative", "standard", "strong"]
    concision: Literal["preserve", "tighten", "compact"]
    tone: Literal["keep", "neutral", "formal", "executive", "friendly"]

    intent: Literal["business_polish"]
    protected_terms: list[str] = []
    quality_mode: Literal["balanced"] = "balanced"
    focus_categories: list[str] = []
    max_rounds: Literal[1] = 1
    preserve_formatting: bool = True
```

Next.js should build this Core request by combining the browser-provided four settings with these v1 defaults:

```json
{
  "intent": "business_polish",
  "quality_mode": "balanced",
  "protected_terms": [],
  "focus_categories": [],
  "max_rounds": 1,
  "preserve_formatting": true
}
```

## Rewrite Response Contract

Core returns:

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

- `prepare`: enforce maximum input length, split/analyze input as needed, infer document type when `document_type` is `auto`.
- `rewrite`: map document type, intensity, concision, and tone to model instructions; request structured output from the LLM.
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
HUMANIZE_MAX_CHARS=10000
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

## Next.js SaaS Scope

The Next.js SaaS app is responsible for:

- `/ai` rewrite UI.
- `POST /api/rewrite`.
- Supabase session validation.
- Subscription lookup.
- Free/pro plan resolution.
- Per-request character limit checks.
- Daily/monthly usage checks.
- Creating `rewrite_usage_events` without text bodies.
- Signing and forwarding requests to Core.
- Updating usage events to `succeeded` or `failed`.

Browser request type:

```ts
type RewriteClientRequest = {
  text: string;
  document_type:
    | "auto"
    | "report"
    | "formal"
    | "email"
    | "proposal"
    | "meeting_notes"
    | "blog"
    | "column";
  intensity: "conservative" | "standard" | "strong";
  concision: "preserve" | "tighten" | "compact";
  tone: "keep" | "neutral" | "formal" | "executive" | "friendly";
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
    maxCharsPerRequest: 10000,
    dailyRequests: 100,
    monthlyRequests: 1000,
  },
};
```

Active subscription means `pro`; no active subscription means `free`.

## Usage Event Table

The SaaS app should create a body-free usage table:

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

Never store:

```text
원문
윤문 결과
diff 본문
finding 본문
LLM raw request/response body
```

## Test Requirements

Lightsail Core tests must cover:

- `/health` returns ok.
- Missing or invalid API key returns `401`.
- Missing or invalid HMAC returns `401`.
- Expired timestamp returns `401`.
- Body hash mismatch returns `401`.
- Invalid enum returns `422`.
- Valid request returns structured `RewriteResponse`.
- Logs do not include source text or rewritten text.

Next.js tests should cover:

- Unauthenticated request returns `401`.
- Free users above 3,000 chars are blocked before Core call.
- Free users above daily or monthly limits receive `429`.
- Pro users at or below 10,000 chars call Core.
- Core success updates usage event to `succeeded`.
- Core failure updates usage event to `failed`.
- Logs and DB do not contain source text or rewrite result.

## V1 Exclusions

Do not implement these unless explicitly requested:

```text
POST /v1/rewrite-jobs
Redis
queue/worker
polling
SSE streaming
Team plan
API key issuance
body history storage
```

## Verification

For Core changes, run:

```bash
cd services/humanize-core
uv run --python 3.12 --with '.[dev]' pytest -q
```

Report any skipped verification clearly.
