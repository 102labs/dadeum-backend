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
    """Append privacy-safe rewrite debug events to daily text log files."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.debug_log_enabled
        self.log_dir = Path(settings.debug_log_dir).expanduser()
        self.include_plaintext = settings.debug_log_include_plaintext
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
        level: str | None = None,
        source: str | None = None,
        message: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": (level or _level_for_event(event)).upper(),
            "source": source or _source_for_event(event),
            "event": event,
            "message": message or _message_for_event(event),
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
            record["details"] = _sanitize_mapping(details, include_plaintext=self.include_plaintext)

        self._write(_format_log_line(record))

    def _write(self, line: str) -> None:
        try:
            with self._lock:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                filename = datetime.now().astimezone().strftime("%Y-%m-%d.log")
                with (self.log_dir / filename).open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
        except OSError:
            logger.warning("rewrite debug log write failed", extra={"debug_log_dir": str(self.log_dir)})


def _sanitize_mapping(value: Mapping[str, Any], *, include_plaintext: bool) -> dict[str, Any]:
    return {
        str(key): _sanitize_value(str(key), item, include_plaintext=include_plaintext)
        for key, item in value.items()
    }


def _sanitize_value(key: str, value: Any, *, include_plaintext: bool) -> Any:
    if not include_plaintext and _should_redact_key(key):
        return _redacted_metadata(value)
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, include_plaintext=include_plaintext)
    if isinstance(value, str):
        if include_plaintext:
            return value
        return value if len(value) <= 256 else {"redacted": True, "length": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_sanitize_value(key, item, include_plaintext=include_plaintext) for item in value]
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


def _format_log_line(record: Mapping[str, Any]) -> str:
    context: dict[str, Any] = {}
    for key in ("requestId", "jobId", "step", "status", "durationMs", "errorCode"):
        if key in record:
            context[key] = record[key]
    details = record.get("details")
    if isinstance(details, Mapping):
        context.update(details)

    suffix = _format_context(context)
    line = (
        f"{record['timestamp']} | {record['level']} | {record['source']} | "
        f"event={record['event']} | {record['message']}"
    )
    if suffix:
        line = f"{line} | {suffix}"
    return line


def _format_context(values: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in values.items():
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def _format_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("redacted") is True:
            redacted_parts = [f"{key}={item}" for key, item in value.items() if key != "redacted"]
            suffix = " " + " ".join(redacted_parts) if redacted_parts else ""
            return f"[REDACTED{suffix}]"
        return "{" + _format_context(value) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        return f'"{escaped}"'
    return text


def _level_for_event(event: str) -> str:
    if ".failed" in event:
        return "ERROR"
    if ".not_found" in event:
        return "WARNING"
    if event.endswith(".started") or event.endswith(".loaded"):
        return "DEBUG"
    return "INFO"


def _source_for_event(event: str) -> str:
    if event.startswith("api."):
        return "api.py"
    if event.startswith("graph."):
        return "graph.py"
    if event.startswith("job."):
        return "jobs.py"
    return "debug_log.py"


def _message_for_event(event: str) -> str:
    messages = {
        "api.rewrite.accepted": "rewrite 요청을 받았습니다.",
        "api.rewrite.succeeded": "rewrite 요청을 동기 처리했습니다.",
        "api.rewrite.failed": "rewrite 요청 처리에 실패했습니다.",
        "job.enqueued": "strict rewrite 작업을 큐에 등록했습니다.",
        "job.claimed": "worker가 strict rewrite 작업을 가져갔습니다.",
        "job.processing.started": "strict rewrite 작업 처리를 시작했습니다.",
        "job.payload.loaded": "암호화된 작업 payload를 복호화해 메타데이터를 확인했습니다.",
        "job.succeeded": "strict rewrite 작업이 정상 완료됐습니다.",
        "job.failed": "strict rewrite 작업이 실패했습니다.",
        "job.cancelled": "strict rewrite 작업을 취소했습니다.",
        "job.status.not_found": "조회한 strict rewrite 작업을 찾지 못했습니다.",
        "job.cancel.not_found": "취소할 strict rewrite 작업을 찾지 못했습니다.",
        "graph.stage.started": "그래프 단계를 시작했습니다.",
        "graph.stage.succeeded": "그래프 단계가 완료됐습니다.",
        "graph.stage.failed": "그래프 단계가 실패했습니다.",
    }
    return messages.get(event, "debug 이벤트를 기록했습니다.")
