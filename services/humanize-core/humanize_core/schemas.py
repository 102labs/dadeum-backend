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
    original: str = Field(description="원문에서 바뀐 구간. 원문에 그대로 들어 있는 짧은 조각이어야 한다.")
    revised: str = Field(description="같은 구간의 수정 결과. revisedText에 그대로 들어 있는 조각이어야 한다.")
    reason: str = Field(
        description=(
            "한국어 한 문장(40~80자). 무엇이 왜 어색했고 어떻게 바꿨는지를 일상어로 쓴다. "
            "룰 번호(A-1 등), '룰북', 'S1' 같은 내부 용어는 쓰지 않는다."
        )
    )
    type: ChangeType = Field(
        description=(
            "clarity=뜻이 더 분명해짐, tone=말투·격식 조정, concision=군더더기 삭제, "
            "structure=어순·문장 분리·연결 변경, grammar=조사·어미·피동·맞춤법, "
            "meaning=뜻이 미세하게 달라질 수 있는 수정"
        )
    )
    riskLevel: RiskLevel = Field(
        default="low",
        description="low=의미 동일, medium=뉘앙스가 달라질 수 있음, high=사실·수치·주장에 영향 가능",
    )


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
