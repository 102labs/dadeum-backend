import asyncio
import base64
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import ValidationError

from humanize_core.config import Settings
from humanize_core.debug_log import RewriteDebugLogger
from humanize_core.graph import InputLimitError, RewriteGraphRunner
from humanize_core.llm import LLMConfigurationError, LLMResponseError
from humanize_core.schemas import (
    RewriteJobAccepted,
    RewriteJobStatus,
    RewriteJobStatusValue,
    RewriteRequest,
    RewriteResponse,
)

logger = logging.getLogger("humanize_core")


class RewriteJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class RewriteJobRecord:
    job_id: str
    request_id: str
    status: RewriteJobStatusValue
    rewrite_mode: str
    text_length: int
    attempts: int
    max_attempts: int
    created_at: float
    expires_at: float
    started_at: float | None
    completed_at: float | None
    latency_ms: int | None
    error_code: str | None
    payload_nonce: bytes | None
    payload_ciphertext: bytes | None
    result_nonce: bytes | None
    result_ciphertext: bytes | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RewriteJobRecord":
        return cls(
            job_id=row["id"],
            request_id=row["request_id"],
            status=row["status"],
            rewrite_mode=row["rewrite_mode"],
            text_length=row["text_length"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            latency_ms=row["latency_ms"],
            error_code=row["error_code"],
            payload_nonce=row["payload_nonce"],
            payload_ciphertext=row["payload_ciphertext"],
            result_nonce=row["result_nonce"],
            result_ciphertext=row["result_ciphertext"],
        )


class JobPayloadCipher:
    def __init__(self, settings: Settings) -> None:
        material = settings.job_encryption_key or settings.signing_secret
        if not material:
            raise RewriteJobError("HUMANIZE_JOB_ENCRYPTION_KEY or HUMANIZE_CORE_SIGNING_SECRET is required")
        self._key = _decode_or_derive_key(material)
        self.key_id = hashlib.sha256(self._key).hexdigest()[:16]
        self._aesgcm = AESGCM(self._key)

    def encrypt_json(self, value: dict[str, Any]) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return nonce, self._aesgcm.encrypt(nonce, raw, None)

    def decrypt_json(self, nonce: bytes, ciphertext: bytes) -> dict[str, Any]:
        raw = self._aesgcm.decrypt(nonce, ciphertext, None)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise RewriteJobError("encrypted job payload is not a JSON object")
        return value


class SqliteRewriteJobStore:
    def __init__(self, settings: Settings, cipher: JobPayloadCipher | None = None) -> None:
        self.settings = settings
        self.cipher = cipher or JobPayloadCipher(settings)
        self._lock = RLock()
        self._conn = self._connect(settings.job_store_path)
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_job(self, request_id: str, request: RewriteRequest) -> RewriteJobRecord:
        now = time.time()
        expires_at = now + self.settings.job_retention_seconds
        payload_nonce, payload_ciphertext = self.cipher.encrypt_json(request.model_dump(mode="json"))
        job_id = str(uuid.uuid4())
        with self._lock:
            try:
                self._conn.execute(
                    """
                    insert into rewrite_jobs (
                      id, request_id, status, rewrite_mode, text_length,
                      payload_key_id, payload_nonce, payload_ciphertext,
                      attempts, max_attempts, created_at, expires_at
                    )
                    values (?, ?, 'queued', ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        job_id,
                        request_id,
                        request.rewrite_mode,
                        len(request.text),
                        self.cipher.key_id,
                        payload_nonce,
                        payload_ciphertext,
                        self.settings.job_max_attempts,
                        now,
                        expires_at,
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                existing = self.get_by_request_id(request_id)
                if existing is None:
                    raise
                return existing
        record = self.get(job_id)
        if record is None:
            raise RewriteJobError("failed to create rewrite job")
        return record

    def get(self, job_id: str) -> RewriteJobRecord | None:
        with self._lock:
            row = self._conn.execute(
                "select * from rewrite_jobs where id = ?",
                (job_id,),
            ).fetchone()
        return RewriteJobRecord.from_row(row) if row else None

    def get_by_request_id(self, request_id: str) -> RewriteJobRecord | None:
        with self._lock:
            row = self._conn.execute(
                "select * from rewrite_jobs where request_id = ?",
                (request_id,),
            ).fetchone()
        return RewriteJobRecord.from_row(row) if row else None

    def claim_next(self, worker_id: str) -> RewriteJobRecord | None:
        self.expire_jobs()
        now = time.time()
        locked_until = now + self.settings.job_lock_seconds
        with self._lock:
            self._conn.execute("begin immediate")
            row = self._conn.execute(
                """
                select * from rewrite_jobs
                where status = 'queued'
                   or (status = 'running' and locked_until < ? and attempts < max_attempts)
                order by created_at
                limit 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            self._conn.execute(
                """
                update rewrite_jobs
                set status = 'running',
                    attempts = attempts + 1,
                    locked_by = ?,
                    locked_until = ?,
                    started_at = coalesce(started_at, ?)
                where id = ?
                """,
                (worker_id, locked_until, now, row["id"]),
            )
            self._conn.commit()
            updated = self._conn.execute(
                "select * from rewrite_jobs where id = ?",
                (row["id"],),
            ).fetchone()
        return RewriteJobRecord.from_row(updated) if updated else None

    def load_request(self, record: RewriteJobRecord) -> RewriteRequest:
        if record.payload_nonce is None or record.payload_ciphertext is None:
            raise RewriteJobError("rewrite job payload is not available")
        try:
            payload = self.cipher.decrypt_json(record.payload_nonce, record.payload_ciphertext)
            return RewriteRequest.model_validate(payload)
        except (ValidationError, ValueError, RewriteJobError) as exc:
            raise RewriteJobError("rewrite job payload is invalid") from exc

    def load_result(self, record: RewriteJobRecord) -> RewriteResponse | None:
        if record.status != "succeeded" or record.result_nonce is None or record.result_ciphertext is None:
            return None
        try:
            payload = self.cipher.decrypt_json(record.result_nonce, record.result_ciphertext)
            return RewriteResponse.model_validate(payload)
        except (ValidationError, ValueError, RewriteJobError) as exc:
            raise RewriteJobError("rewrite job result is invalid") from exc

    def complete_success(self, job_id: str, response: RewriteResponse) -> None:
        now = time.time()
        result_nonce, result_ciphertext = self.cipher.encrypt_json(response.model_dump(mode="json"))
        with self._lock:
            self._conn.execute(
                """
                update rewrite_jobs
                set status = 'succeeded',
                    payload_nonce = null,
                    payload_ciphertext = null,
                    result_key_id = ?,
                    result_nonce = ?,
                    result_ciphertext = ?,
                    locked_by = null,
                    locked_until = null,
                    error_code = null,
                    input_tokens = ?,
                    output_tokens = ?,
                    latency_ms = ?,
                    completed_at = ?
                where id = ? and status = 'running'
                """,
                (
                    self.cipher.key_id,
                    result_nonce,
                    result_ciphertext,
                    response.usage.inputTokens,
                    response.usage.outputTokens,
                    response.usage.latencyMs,
                    now,
                    job_id,
                ),
            )
            self._conn.commit()

    def fail_job(self, job_id: str, error_code: str, *, retryable: bool) -> None:
        now = time.time()
        record = self.get(job_id)
        if record is None:
            return
        should_retry = retryable and record.attempts < record.max_attempts
        with self._lock:
            if should_retry:
                self._conn.execute(
                    """
                    update rewrite_jobs
                    set status = 'queued',
                        locked_by = null,
                        locked_until = null,
                        error_code = ?
                    where id = ? and status = 'running'
                    """,
                    (error_code, job_id),
                )
            else:
                self._conn.execute(
                    """
                    update rewrite_jobs
                    set status = 'failed',
                        payload_nonce = null,
                        payload_ciphertext = null,
                        locked_by = null,
                        locked_until = null,
                        error_code = ?,
                        completed_at = ?
                    where id = ? and status = 'running'
                    """,
                    (error_code, now, job_id),
                )
            self._conn.commit()

    def cancel(self, job_id: str) -> RewriteJobRecord | None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                update rewrite_jobs
                set status = 'cancelled',
                    payload_nonce = null,
                    payload_ciphertext = null,
                    result_nonce = null,
                    result_ciphertext = null,
                    locked_by = null,
                    locked_until = null,
                    completed_at = coalesce(completed_at, ?)
                where id = ? and status in ('queued', 'running')
                """,
                (now, job_id),
            )
            self._conn.commit()
        return self.get(job_id)

    def expire_jobs(self) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                update rewrite_jobs
                set status = 'expired',
                    payload_nonce = null,
                    payload_ciphertext = null,
                    result_nonce = null,
                    result_ciphertext = null,
                    locked_by = null,
                    locked_until = null,
                    completed_at = coalesce(completed_at, ?)
                where expires_at < ?
                  and status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')
                """,
                (now, now),
            )
            self._conn.commit()

    def status_response(self, record: RewriteJobRecord) -> RewriteJobStatus:
        result = self.load_result(record)
        return RewriteJobStatus(
            jobId=record.job_id,
            requestId=record.request_id,
            status=record.status,
            rewriteMode=record.rewrite_mode,  # type: ignore[arg-type]
            textLength=record.text_length,
            attempts=record.attempts,
            maxAttempts=record.max_attempts,
            createdAt=_to_datetime(record.created_at),
            startedAt=_to_datetime(record.started_at),
            completedAt=_to_datetime(record.completed_at),
            expiresAt=_to_datetime(record.expires_at),
            latencyMs=record.latency_ms,
            errorCode=record.error_code,
            result=result,
        )

    def _connect(self, store_path: str) -> sqlite3.Connection:
        if store_path != ":memory:":
            Path(store_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(store_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode = wal")
        conn.execute("pragma foreign_keys = on")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                create table if not exists rewrite_jobs (
                  id text primary key,
                  request_id text not null unique,
                  status text not null,
                  rewrite_mode text not null,
                  text_length integer not null,
                  payload_key_id text,
                  payload_nonce blob,
                  payload_ciphertext blob,
                  result_key_id text,
                  result_nonce blob,
                  result_ciphertext blob,
                  attempts integer not null default 0,
                  max_attempts integer not null default 2,
                  locked_by text,
                  locked_until real,
                  error_code text,
                  input_tokens integer,
                  output_tokens integer,
                  latency_ms integer,
                  created_at real not null,
                  started_at real,
                  completed_at real,
                  expires_at real not null
                )
                """
            )
            self._conn.execute(
                "create index if not exists rewrite_jobs_claim_idx on rewrite_jobs(status, locked_until, created_at)"
            )
            self._conn.execute(
                "create index if not exists rewrite_jobs_expiry_idx on rewrite_jobs(expires_at)"
            )
            self._conn.commit()


class RewriteJobManager:
    def __init__(
        self,
        settings: Settings,
        graph_runner: RewriteGraphRunner,
        store: SqliteRewriteJobStore | None = None,
    ) -> None:
        self.settings = settings
        self.graph_runner = graph_runner
        self.debug_log = RewriteDebugLogger(settings)
        self._store = store
        self.worker_id = f"humanize-core-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._stop_event = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def store(self) -> SqliteRewriteJobStore:
        if self._store is None:
            self._store = SqliteRewriteJobStore(self.settings)
        return self._store

    async def start(self) -> None:
        if not self.settings.job_worker_enabled:
            return
        try:
            _ = self.store
        except RewriteJobError:
            logger.warning("rewrite job worker disabled because job storage is not configured")
            return
        self._stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._worker_task is not None:
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._store is not None:
            self._store.close()
            self._store = None

    async def enqueue(self, request_id: str, request: RewriteRequest) -> RewriteJobAccepted:
        record = self.store.create_job(request_id, request)
        self.debug_log.event(
            "job.enqueued",
            request_id=record.request_id,
            job_id=record.job_id,
            status=record.status,
            details={
                "rewrite_mode": record.rewrite_mode,
                "text_length": record.text_length,
                "attempts": record.attempts,
                "max_attempts": record.max_attempts,
                "poll_after_ms": max(250, int(self.settings.job_poll_interval_seconds * 1000)),
                "expires_at": _to_datetime(record.expires_at).isoformat() if record.expires_at else None,
            },
        )
        return RewriteJobAccepted(
            jobId=record.job_id,
            requestId=record.request_id,
            status=record.status,
            pollAfterMs=max(250, int(self.settings.job_poll_interval_seconds * 1000)),
        )

    def get_status(self, job_id: str) -> RewriteJobStatus | None:
        self.store.expire_jobs()
        record = self.store.get(job_id)
        if record is None:
            self.debug_log.event("job.status.not_found", job_id=job_id, status="not_found")
            return None
        return self.store.status_response(record)

    def cancel(self, job_id: str) -> RewriteJobStatus | None:
        record = self.store.cancel(job_id)
        if record is None:
            self.debug_log.event("job.cancel.not_found", job_id=job_id, status="not_found")
            return None
        self.debug_log.event(
            "job.cancelled",
            request_id=record.request_id,
            job_id=record.job_id,
            status=record.status,
            details=_job_debug_details(record),
        )
        return self.store.status_response(record)

    async def process_next(self) -> bool:
        record = self.store.claim_next(self.worker_id)
        if record is None:
            return False
        self.debug_log.event(
            "job.claimed",
            request_id=record.request_id,
            job_id=record.job_id,
            status=record.status,
            details={**_job_debug_details(record), "worker_id": self.worker_id},
        )
        await self._process_record(record)
        return True

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            processed = await self.process_next()
            if not processed:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.settings.job_poll_interval_seconds,
                    )
                except TimeoutError:
                    pass

    async def _process_record(self, record: RewriteJobRecord) -> None:
        started_at = time.perf_counter()
        self.debug_log.event(
            "job.processing.started",
            request_id=record.request_id,
            job_id=record.job_id,
            status="running",
            details={**_job_debug_details(record), "worker_id": self.worker_id},
        )
        try:
            request = self.store.load_request(record)
            self.debug_log.event(
                "job.payload.loaded",
                request_id=record.request_id,
                job_id=record.job_id,
                status="succeeded",
                details={
                    "rewrite_mode": request.rewrite_mode,
                    "text_length": len(request.text),
                    "protected_terms_count": len(request.protected_terms),
                    "user_intent_length": len(request.user_intent),
                    "source_text": request.text,
                    "user_intent": request.user_intent,
                    "protected_terms": request.protected_terms,
                },
            )
            response = await self.graph_runner.run(request, request_id=record.request_id, job_id=record.job_id)
            self.store.complete_success(record.job_id, response)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            self.debug_log.event(
                "job.succeeded",
                request_id=record.request_id,
                job_id=record.job_id,
                status="succeeded",
                duration_ms=duration_ms,
                details={
                    **_job_debug_details(record),
                    "latency_ms": response.usage.latencyMs,
                    "input_tokens": response.usage.inputTokens,
                    "output_tokens": response.usage.outputTokens,
                    "rounds": response.usage.rounds,
                    "changes_count": len(response.changes),
                    "summary_count": len(response.summary),
                    "warnings_count": len(response.warnings),
                    "revised_text_length": len(response.revisedText),
                    "revised_text": response.revisedText,
                    "changes": [change.model_dump(mode="json") for change in response.changes],
                    "summary": response.summary,
                    "warnings": response.warnings,
                },
            )
            logger.info(
                "rewrite job succeeded",
                extra={
                    "job_id": record.job_id,
                    "request_id": record.request_id,
                    "rewrite_mode": record.rewrite_mode,
                    "latency_ms": response.usage.latencyMs,
                },
            )
        except InputLimitError:
            self.store.fail_job(record.job_id, "input_limit_exceeded", retryable=False)
            self._log_job_failed(record, started_at, "input_limit_exceeded", retryable=False)
            logger.warning("rewrite job failed due to input limit", extra={"job_id": record.job_id})
        except LLMConfigurationError:
            self.store.fail_job(record.job_id, "model_not_configured", retryable=False)
            self._log_job_failed(record, started_at, "model_not_configured", retryable=False)
            logger.error("rewrite job failed due to model configuration", extra={"job_id": record.job_id})
        except LLMResponseError:
            self.store.fail_job(record.job_id, "invalid_model_response", retryable=True)
            self._log_job_failed(record, started_at, "invalid_model_response", retryable=True)
            logger.warning("rewrite job failed due to invalid model response", extra={"job_id": record.job_id})
        except Exception:  # noqa: BLE001 - never let one job kill the worker loop.
            self.store.fail_job(record.job_id, "internal_error", retryable=True)
            self._log_job_failed(record, started_at, "internal_error", retryable=True)
            logger.error("rewrite job failed", extra={"job_id": record.job_id, "error_code": "internal_error"})

    def _log_job_failed(
        self,
        record: RewriteJobRecord,
        started_at: float,
        error_code: str,
        *,
        retryable: bool,
    ) -> None:
        will_retry = retryable and record.attempts < record.max_attempts
        self.debug_log.event(
            "job.failed",
            request_id=record.request_id,
            job_id=record.job_id,
            status="queued" if will_retry else "failed",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=error_code,
            details={
                **_job_debug_details(record),
                "retryable": retryable,
                "will_retry": will_retry,
            },
        )


def _decode_or_derive_key(material: str) -> bytes:
    value = material.strip()
    if not value:
        raise RewriteJobError("empty job encryption key")

    if len(value) == 64:
        try:
            decoded = bytes.fromhex(value)
            if len(decoded) == 32:
                return decoded
        except ValueError:
            pass

    padded = value + ("=" * (-len(value) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        if len(decoded) == 32:
            return decoded
    except ValueError:
        pass

    return hashlib.sha256(value.encode("utf-8")).digest()


def _job_debug_details(record: RewriteJobRecord) -> dict[str, Any]:
    return {
        "rewrite_mode": record.rewrite_mode,
        "text_length": record.text_length,
        "attempts": record.attempts,
        "max_attempts": record.max_attempts,
        "latency_ms": record.latency_ms,
        "error_code": record.error_code,
        "created_at": _to_datetime(record.created_at).isoformat() if record.created_at else None,
        "started_at": _to_datetime(record.started_at).isoformat() if record.started_at else None,
        "completed_at": _to_datetime(record.completed_at).isoformat() if record.completed_at else None,
        "expires_at": _to_datetime(record.expires_at).isoformat() if record.expires_at else None,
    }


def _to_datetime(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)
