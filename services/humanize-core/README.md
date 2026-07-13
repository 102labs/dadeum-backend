# Humanize Core

Internal FastAPI service for short business writing rewrite requests.

## Endpoints

- `GET /health`
- `POST /v1/rewrite`
- `GET /v1/rewrite-jobs/{jobId}`
- `DELETE /v1/rewrite-jobs/{jobId}`

These endpoints are intended for server-to-server calls from the Next.js app only. The service does not enable browser CORS and requires:

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

Core accepts up to 5,000 characters per request. Core infers internal genre
hints from the text itself. The rewrite logic uses `user_intent`, `tone`,
`protected_terms`, and `preserve_formatting` to choose the rewrite direction,
preservation policy, and formatting policy.

`rewrite_mode=fast` keeps the existing synchronous behavior and returns a
`RewriteResponse` from `POST /v1/rewrite`.

`rewrite_mode=strict` is asynchronous. `POST /v1/rewrite` validates and encrypts
the payload, stores a durable job, and returns `202 Accepted`:

```json
{
  "jobId": "uuid",
  "requestId": "req_...",
  "status": "queued",
  "pollAfterMs": 1000
}
```

The Next.js server polls `GET /v1/rewrite-jobs/{jobId}` using the same signed
header scheme. A succeeded job includes `result`; queued/running/failed jobs do
not expose plaintext bodies. `DELETE /v1/rewrite-jobs/{jobId}` cancels queued or
running work and purges encrypted payload/result fields.

## Local Run

```bash
cp .env.example .env
docker compose up --build
```

The Compose stack exposes Caddy on port `80` and proxies `/health`,
`/v1/rewrite`, and `/v1/rewrite-jobs/*` to the FastAPI container. It mounts a
named volume at `/data` for durable SQLite job storage.

For local tests, the default `stub` provider avoids external LLM calls. Production can use the OpenRouter path:

```text
HUMANIZE_MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=...
HUMANIZE_REWRITE_MODEL_NAME=openai/gpt-5-mini
HUMANIZE_REWRITE_FALLBACK_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_AUDIT_MODEL_NAME=~anthropic/claude-haiku-latest
HUMANIZE_STRICT_REVIEW_MODEL_NAME=openai/gpt-5.4-mini
HUMANIZE_JOB_STORE_PATH=/data/humanize_jobs.sqlite3
HUMANIZE_JOB_ENCRYPTION_KEY=<32-byte base64url or hex key>
HUMANIZE_DEBUG_LOG_ENABLED=true
HUMANIZE_DEBUG_LOG_DIR=/data/humanize-core/logs
HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=false
```

`rewrite_mode` defaults to `fast`; strict requests run through the durable async
job queue.
The graph path is:

```text
prepare -> rewrite -> audit -> review? -> finalize
```

The worker executes the same graph for strict jobs. `rewrite` performs one
full-pass rewrite using `strict-rules.md`. The full rulebook rides in the
static system prompt so providers can prefix-cache it; the per-request user
payload carries only the text, settings, and compact rulebook hints (with
occurrence counts, never raw source spans). `audit`
compares the draft to the original and flags only harmful preservation problems:
changed facts, numbers, dates, units, names, quotations, protected terms, order,
polarity, causality, omitted content, or added claims. If audit has no repair
items, the graph finalizes immediately. If audit finds a repair item, `review`
applies only those corrections and returns the final candidate. The graph
re-checks the model review output locally; if the review step itself truncated
the text or damaged preserved values relative to the draft, the graph discards
it and falls back to the local repair path with a warning.

The OpenAI provider uses the Responses API with strict JSON Schema structured
output for `revisedText`, `changes`, and `summary`. Usage metrics come from the
provider response metadata, not from model-generated JSON.

The OpenRouter provider uses Chat Completions with `response_format:
json_schema` for rewrite, audit, and review. It sets provider routing to require
models that support the requested structured-output parameters.

## Golden-Set Eval

`evals/golden_set.json` holds 32 fixed Korean inputs grouped by failure type
(번역투, AI 관용구, 수치·인용 보존, 격식/해요체, 긴 글, 엣지 케이스).
`scripts/eval_golden.py` runs each case through the full graph and scores the
output with the detectors already in the codebase: residual S1/S2 patterns
(`local_detect` on the output), change rate, over-polish signals, preservation
damage, and completion warnings.

```bash
# full run against the configured provider (.env), saves a JSON report
.venv/bin/python scripts/eval_golden.py

# compare against a previous report to catch regressions after prompt changes
.venv/bin/python scripts/eval_golden.py --baseline evals/reports/report-<ts>.json

# subset / offline harness check
.venv/bin/python scripts/eval_golden.py --tag 번역투
.venv/bin/python scripts/eval_golden.py --stub
```

On the server (where `.env` lives) run it through Compose. The one-off
container reuses the service's `env_file`; mount the reports dir so results
survive the container:

```bash
docker compose build humanize-core
docker compose run --rm \
  -v "$PWD/evals/reports:/app/evals/reports" \
  humanize-core python scripts/eval_golden.py
```

Run it once before a prompt/rulebook change and once after, then compare with
`--baseline`. A drop in residual S1 with unchanged preservation failures means
the change is safe to keep. Do not edit existing case texts (ids stay
comparable across reports); add new cases instead.

## Async Debug Logs

Strict async jobs write operational logs to daily text files:

```text
/data/humanize-core/logs/YYYY-MM-DD.log
```

Override the directory with `HUMANIZE_DEBUG_LOG_DIR`; disable the file logs with
`HUMANIZE_DEBUG_LOG_ENABLED=false`.

Quick checks:

```bash
tail -f /data/humanize-core/logs/$(date +%F).log
tail -n 1 /data/humanize-core/logs/$(date +%F).log
```

Each line is one event:

```text
timestamp | LEVEL | source-file | event=... | message | key=value ...
```

Useful event names include `job.enqueued`, `job.claimed`,
`graph.stage.started`, `graph.stage.succeeded`, `graph.stage.failed`,
`job.succeeded`, and `job.failed`. Events include request/job ids, step names,
durations, statuses, token counts, warning/change counts, retry decisions, and
error codes. Repeated polling reads are intentionally not logged.

By default, logs redact plaintext source text, rewritten text, diff/change
bodies, findings, protected term values, user intent text, prompts, and raw LLM
payloads. During an explicit debugging window, set:

```text
HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=true
```

When enabled, logs include source text, revised text, summaries, warnings, and
change/audit details. Turn it back off after debugging.

## Privacy Boundary

Fast synchronous requests keep source text and rewritten text in process memory
only for the duration of the request.

Strict async jobs persist only encrypted source payloads and encrypted final
results, with a TTL controlled by `HUMANIZE_JOB_RETENTION_SECONDS`. The worker
deletes the encrypted source payload after terminal success or final failure.
The service still does not write plaintext request bodies, plaintext model raw
bodies, plaintext rewrite output, plaintext diffs, or plaintext findings to a
database or log unless `HUMANIZE_DEBUG_LOG_INCLUDE_PLAINTEXT=true` is explicitly
enabled for a temporary debugging window. Debug logs still avoid raw LLM
request/response bodies, encrypted payload bytes, and decrypted job payload
storage values.
