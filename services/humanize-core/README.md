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

Core accepts up to `HUMANIZE_MAX_CHARS` characters per request (default 5,000;
longer input is `422`). Unknown fields are rejected (`422`). Core infers
internal rulebook hints from the text itself with a local regex detector. The
rewrite logic uses `user_intent`, `tone`, `protected_terms`, and
`preserve_formatting` to choose the rewrite direction, preservation policy, and
formatting policy. `max_rounds` is validated (1-3) but not used by the graph;
`usage.rounds` reports `1 + style-gate repair rounds`.

`rewrite_mode=fast` runs the graph synchronously and returns a `RewriteResponse`
from `POST /v1/rewrite`. Errors: `422` input limit, `503` provider not
configured, `502` invalid structured model response.

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
header scheme (hash of the empty body). The status record carries `status`
(`queued | running | succeeded | failed | cancelled | expired`), `attempts`,
`maxAttempts`, timestamps, `latencyMs`, `errorCode`, and `result`. A succeeded
job includes `result`; other states do not expose plaintext bodies.
`DELETE /v1/rewrite-jobs/{jobId}` cancels queued or running work and purges
encrypted payload/result fields. Unknown ids return `404`.

Strict jobs are processed by one in-process worker task started with the app.
`invalid_model_response` and `internal_error` are retried up to
`HUMANIZE_JOB_MAX_ATTEMPTS` (2); `input_limit_exceeded` and
`model_not_configured` fail immediately. Running jobs whose lock is older than
`HUMANIZE_JOB_LOCK_SECONDS` (600) are reclaimed. Rows expire after
`HUMANIZE_JOB_RETENTION_SECONDS` (86400).

## Local Run

```bash
cp .env.example .env
docker compose up --build
```

The Compose stack exposes Caddy on port `80` and proxies `/health`,
`/v1/rewrite`, and `/v1/rewrite-jobs/*` to the FastAPI container. It mounts a
named volume at `/data` for durable SQLite job storage.

For local tests, the default `stub` provider avoids external LLM calls. Production uses the OpenRouter path:

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

The four model names above are the code defaults in `config.py`. See
`.env.example` for the full variable list (chunking, style gate, job worker).

## Graph

```text
prepare -> rewrite -> style_gate -> audit -> (review) -> finalize
```

- `prepare`: length check, local regex detection (`im_not_ai/audit.py`), compact
  rulebook hints. No LLM call.
- `rewrite`: one structured LLM call. Text of 1,000+ chars
  (`HUMANIZE_CHUNK_MIN_CHARS`) is split at sentence boundaries into ~1,000-char
  chunks, rewritten in parallel, and reassembled. The rulebook
  (`strict-rules.md`, bodies stripped) rides in the static system prompt so
  providers can prefix-cache it; the user payload carries the text, settings,
  and hints with full rule cards and up to 3 short match samples (never spans
  overlapping numbers, quotes, or protected terms).
- `style_gate`: re-detect on the draft. If any S1 remains, S2 count reaches
  `HUMANIZE_STYLE_GATE_S2_THRESHOLD` (3), or the text was chunked (one
  transition-smoothing pass), call `style_repair` up to
  `HUMANIZE_STYLE_GATE_MAX_ROUNDS` (2) times. Repairs that add completion
  warnings, increase preservation damage, or worsen the severity score are
  discarded. Only the OpenRouter provider implements `style_repair`.
- `audit`: local completion checks (empty, too short, low sentence/paragraph
  coverage, cut off mid-sentence) and exact preservation counts (protected
  terms, quotes, URLs, emails, code spans, dates, numbers/units must not go
  down or up). The OpenRouter provider adds a model audit for harmful meaning
  changes. Completion warnings force `fail`.
- `review` (only when audit is `fail`/`conditional_pass` or flags a repair):
  the model applies only the audit corrections. The output is re-checked; if
  it truncated the text, damaged more preserved values, or reverted the draft
  wholesale to the source, it is replaced by the local repair path with a
  warning. Without a provider `review` the local repair runs directly. Style
  restores that would reintroduce an S1 violation are kept as the gated draft.
- `finalize`: merge warnings, rebuild display-safe `changes` (exact substrings
  of source and result, max 12), sum tokens across all stages.

Provider capability matrix:

| provider     | rewrite | style_repair | model audit | model review |
|--------------|---------|--------------|-------------|--------------|
| `stub`       | local   | no           | local only  | local only   |
| `openai`     | yes     | no           | local only  | local only   |
| `anthropic`  | yes     | no           | local only  | local only   |
| `openrouter` | yes     | yes          | yes         | yes          |

The OpenAI provider uses the Responses API with strict JSON Schema structured
output for `revisedText`, `changes`, and `summary`. Usage metrics come from the
provider response metadata, not from model-generated JSON.

The OpenRouter provider uses Chat Completions with `response_format:
json_schema` (`strict: true`, schema normalised to `additionalProperties:
false` with every property required) for rewrite, style repair, audit, and
review. It sets `provider.require_parameters: true` and sends no temperature.
Model lists are tried in order and fall through on any exception: rewrite and
style repair use `[rewrite, rewrite fallback]`, audit `[audit, rewrite]`,
review `[review, rewrite]`. When `HUMANIZE_MODEL_NAME` is set to anything but
`stub` it replaces the rewrite primary. Every call uses `max_tokens=20000`.

Per-request LLM budget with OpenRouter: 1 rewrite (or N chunk calls) + 0-2
style repairs + 1 audit + 0-1 review. Fast mode waits for all of it; Core sets
no request timeout.

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

Redaction is key-name based (`debug_log.py`): detail keys containing `text`,
`source`, `revised`, `change`, `summary`, `warning`, `finding`, `intent`,
`protected`, `term`, `prompt`, `raw`, `body`, `payload`, `result`, and similar
are written as `[REDACTED length=N]` unless they end with a metric suffix such
as `_count` or `_ms`. Strings over 256 chars are redacted regardless of key.
When adding a detail key that carries body text, pick a name the blocklist
catches. During an explicit debugging window, set:

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
