import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from humanize_core.config import Settings


logger = logging.getLogger("humanize_core")


_BLOCKED_KEY_PARTS = (
    "body",
    "change",
    "cipher",
    "diff",
    "finding",
    "intent",
    "nonce",
    "original",
    "payload",
    "prompt",
    "protected",
    "raw",
    "result",
    "revised",
    "source",
    "summary",
    "term",
    "text",
    "warning",
)
_METRIC_SUFFIXES = (
    "_at",
    "_count",
    "_enabled",
    "_length",
    "_ms",
    "_seconds",
    "_tokens",
    "attempts",
    "count",
    "length",
    "ms",
    "rounds",
    "tokens",
)
_SAFE_KEYS = {
    "attempts",
    "completed_at",
    "created_at",
    "duration_ms",
    "error_code",
    "error_type",
    "event",
    "expires_at",
    "job_id",
    "latency_ms",
    "max_attempts",
    "max_chars",
    "poll_after_ms",
    "preserve_formatting",
    "request_id",
    "result_status",
    "retryable",
    "rewrite_mode",
    "stage",
    "started_at",
    "status",
    "step",
    "tone",
    "will_retry",
    "worker_id",
}


class RewriteDebugLogger:
    """Append privacy-safe rewrite debug events to daily JSONL files."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.debug_log_enabled
        self.log_dir = Path(settings.debug_log_dir).expanduser()
        self._lock = RLock()

    def event(
        self,
        event: str,
        *,
        request_id: str | None = None,
        job_id: str | None = None,
        step: str | None = None,
        status: str | None = None,
        duration_ms: int | None = None,
        error_code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
        }
        if request_id:
            record["requestId"] = request_id
        if job_id:
            record["jobId"] = job_id
        if step:
            record["step"] = step
        if status:
            record["status"] = status
        if duration_ms is not None:
            record["durationMs"] = duration_ms
        if error_code:
            record["errorCode"] = error_code
        if details:
            record["details"] = _sanitize_mapping(details)

        self._write(record)

    def _write(self, record: Mapping[str, Any]) -> None:
        try:
            with self._lock:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                filename = datetime.now().astimezone().strftime("%Y-%m-%d.jsonl")
                with (self.log_dir / filename).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
        except OSError:
            logger.warning("rewrite debug log write failed", extra={"debug_log_dir": str(self.log_dir)})


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_value(str(key), item) for key, item in value.items()}


def _sanitize_value(key: str, value: Any) -> Any:
    if _should_redact_key(key):
        return _redacted_metadata(value)
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, str):
        return value if len(value) <= 256 else {"redacted": True, "length": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)


def _should_redact_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if normalized in _SAFE_KEYS or normalized.endswith(_METRIC_SUFFIXES):
        return False
    return any(part in normalized for part in _BLOCKED_KEY_PARTS)


def _redacted_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"redacted": True, "length": len(value)}
    if isinstance(value, Mapping):
        return {"redacted": True, "fieldCount": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {"redacted": True, "count": len(value)}
    if value is None:
        return {"redacted": True}
    return {"redacted": True, "type": type(value).__name__}
