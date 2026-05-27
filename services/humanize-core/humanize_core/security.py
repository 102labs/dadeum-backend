import hashlib
import hmac
import time
from datetime import datetime, timezone

from fastapi import HTTPException, Request, status

from humanize_core.config import Settings


REQUIRED_HEADERS = (
    "X-Core-Api-Key",
    "X-Request-Id",
    "X-Timestamp",
    "X-Body-SHA256",
    "X-Signature",
)


def sha256_hex(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def hmac_sha256_hex(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _parse_timestamp(timestamp: str) -> float:
    value = timestamp.strip()
    try:
        numeric = float(value)
    except ValueError:
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    if numeric > 10_000_000_000:
        return numeric / 1000
    return numeric


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


async def verify_core_request(request: Request, raw_body: bytes, settings: Settings) -> None:
    headers = request.headers
    if any(not headers.get(name) for name in REQUIRED_HEADERS):
        raise _unauthorized()

    configured_key = settings.core_api_key.encode("utf-8")
    provided_key = headers["X-Core-Api-Key"].encode("utf-8")
    if not configured_key or not hmac.compare_digest(provided_key, configured_key):
        raise _unauthorized()

    timestamp = headers["X-Timestamp"]
    request_time = _parse_timestamp(timestamp)
    if abs(time.time() - request_time) > settings.signature_tolerance_seconds:
        raise _unauthorized()

    body_hash = sha256_hex(raw_body)
    if not hmac.compare_digest(headers["X-Body-SHA256"].lower(), body_hash):
        raise _unauthorized()

    if not settings.signing_secret:
        raise _unauthorized()

    request_id = headers["X-Request-Id"]
    payload = f"{timestamp}.{request_id}.{body_hash}"
    expected_signature = hmac_sha256_hex(payload, settings.signing_secret)
    if not hmac.compare_digest(headers["X-Signature"].lower(), expected_signature):
        raise _unauthorized()

