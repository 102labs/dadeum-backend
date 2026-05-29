from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Tone = Literal["keep", "formal", "friendly"]
RewriteMode = Literal["fast", "strict"]
ChangeType = Literal[
    "clarity",
    "tone",
    "concision",
    "structure",
    "grammar",
    "meaning",
]
RiskLevel = Literal["low", "medium", "high"]
RewriteJobStatusValue = Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    user_intent: str = ""
    rewrite_mode: RewriteMode = "fast"
    tone: Tone = "keep"
    protected_terms: list[str] = Field(default_factory=list)
    max_rounds: int = Field(default=1, ge=1, le=3)
    preserve_formatting: bool = True

    @field_validator("user_intent")
    @classmethod
    def normalize_user_intent(cls, value: str) -> str:
        return value.strip()

    @field_validator("protected_terms")
    @classmethod
    def normalize_protected_terms(cls, value: list[str]) -> list[str]:
        return [term.strip() for term in value if term.strip()]


class Change(BaseModel):
    original: str
    revised: str
    reason: str
    type: ChangeType
    riskLevel: RiskLevel = "low"


class Usage(BaseModel):
    inputTokens: int = 0
    outputTokens: int = 0
    latencyMs: int
    rounds: int = 1


class RewriteResponse(BaseModel):
    revisedText: str
    changes: list[Change]
    summary: list[str]
    warnings: list[str]
    usage: Usage


class LLMRewriteResult(BaseModel):
    revisedText: str
    changes: list[Change]
    summary: list[str]
    inputTokens: int = 0
    outputTokens: int = 0


class RewriteJobAccepted(BaseModel):
    jobId: str
    requestId: str
    status: RewriteJobStatusValue
    pollAfterMs: int = 1000


class RewriteJobStatus(BaseModel):
    jobId: str
    requestId: str
    status: RewriteJobStatusValue
    rewriteMode: RewriteMode
    textLength: int
    attempts: int
    maxAttempts: int
    createdAt: datetime
    expiresAt: datetime
    startedAt: datetime | None = None
    completedAt: datetime | None = None
    latencyMs: int | None = None
    errorCode: str | None = None
    result: RewriteResponse | None = None
