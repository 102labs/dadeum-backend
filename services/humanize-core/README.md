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
  "rewrite_mode": "fast",
  "tone": "keep",
  "protected_terms": [],
  "max_rounds": 1,
  "preserve_formatting": true
}
```

Core infers internal genre hints from the text itself. The rewrite logic uses
`user_intent`, `rewrite_mode`, `tone`, and `preserve_formatting` to choose the
rewrite strength, tone policy, and formatting policy. Strict mode uses the
requested `max_rounds` value up to 3 rounds; fast mode uses 1 round.

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
HUMANIZE_FAST_MODEL_NAME=openai/gpt-5-mini
HUMANIZE_FAST_FALLBACK_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_DETECT_MODEL_NAME=openai/gpt-5-mini
HUMANIZE_STRICT_REWRITE_MODEL_NAME=~anthropic/claude-sonnet-latest
HUMANIZE_STRICT_AUDIT_MODEL_NAME=openai/gpt-5
HUMANIZE_STRICT_REVIEW_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_ESCALATION_MODEL_NAME=~anthropic/claude-opus-latest
```

`rewrite_mode` defaults to `fast`. Fast mode ports the `docs/im-not-ai` monolith path by combining `quick-rules.md`, `metrics_v2.py`, and a single structured rewrite call. `strict` mode runs a LangGraph path:

```text
prepare -> detect -> rewrite -> audit -> review -> finalize
```

Strict mode can loop from `review` back to `rewrite` up to `HUMANIZE_STRICT_MAX_ROUNDS` when audit or residual-pattern review requires another pass.

The OpenAI provider uses the Responses API with strict JSON Schema structured
output for `revisedText`, `changes`, and `summary`. Usage metrics come from the
provider response metadata, not from model-generated JSON.

The OpenRouter provider uses Chat Completions with `response_format:
json_schema` for Fast and every Strict node. It sets provider routing to require
models that support the requested structured-output parameters.

## Privacy Boundary

The service keeps source text and rewritten text in process memory only for the duration of the request. It does not write request bodies, model raw bodies, rewrite output, diffs, or findings to a database or log.
