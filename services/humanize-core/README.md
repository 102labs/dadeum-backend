# Humanize Core

Internal FastAPI service for short business writing rewrite requests.

## Endpoints

- `GET /health`
- `POST /v1/rewrite`

`/v1/rewrite` is intended for server-to-server calls from the Next.js app only. The service does not enable browser CORS and requires:

- `X-Core-Api-Key`
- `X-Request-Id`
- `X-Timestamp`
- `X-Body-SHA256`
- `X-Signature`

Signature payload:

```text
${timestamp}.${requestId}.${sha256(rawJsonBody)}
```

The signature is `HMAC-SHA256` using `HUMANIZE_CORE_SIGNING_SECRET`.

## Rewrite Contract

The browser-facing API accepts only the user-selected rewrite controls:

```ts
type CoreRewriteRequest = {
  text: string;
  user_intent?: string;
  rewrite_mode?: "fast" | "strict";
  tone?: "keep" | "formal" | "friendly";
  protected_terms?: string[];
  max_rounds?: number;
  preserve_formatting?: boolean;
};
```

The Next.js server signs and forwards the Core payload with internal fields:

```json
{
  "text": "윤문할 원문",
  "user_intent": "",
  "rewrite_mode": "strict",
  "tone": "keep",
  "protected_terms": [],
  "max_rounds": 1,
  "preserve_formatting": true
}
```

Core infers internal genre hints from the text itself. The rewrite logic uses
`user_intent`, `tone`, `protected_terms`, and `preserve_formatting` to choose
the rewrite direction, preservation policy, and formatting policy.
`rewrite_mode` and `max_rounds` are accepted only for request compatibility; Core
now runs the same strict-only routine for every request.

## Local Run

```bash
cp .env.example .env
docker compose up --build
```

The Compose stack exposes Caddy on port `80` and proxies only `/health` and
`/v1/rewrite` to the FastAPI container.

For local tests, the default `stub` provider avoids external LLM calls. Production can use the OpenRouter path:

```text
HUMANIZE_MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=...
HUMANIZE_REWRITE_MODEL_NAME=openai/gpt-5-mini
HUMANIZE_REWRITE_FALLBACK_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_AUDIT_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_REVIEW_MODEL_NAME=openai/gpt-5.4-mini
```

`rewrite_mode` defaults to `strict`, and fast/strict branching has been removed.
The graph path is:

```text
prepare -> rewrite -> audit -> review? -> finalize
```

`rewrite` performs one full-pass rewrite using `strict-rules.md`. `audit`
compares the draft to the original and flags only harmful preservation problems:
changed facts, numbers, dates, units, names, quotations, protected terms, order,
polarity, causality, omitted content, or added claims. If audit has no repair
items, the graph finalizes immediately. If audit finds a repair item, `review`
applies only those corrections and returns the final candidate.

The OpenAI provider uses the Responses API with strict JSON Schema structured
output for `revisedText`, `changes`, and `summary`. Usage metrics come from the
provider response metadata, not from model-generated JSON.

The OpenRouter provider uses Chat Completions with `response_format:
json_schema` for rewrite, audit, and review. It sets provider routing to require
models that support the requested structured-output parameters.

## Privacy Boundary

The service keeps source text and rewritten text in process memory only for the duration of the request. It does not write request bodies, model raw bodies, rewrite output, diffs, or findings to a database or log.
