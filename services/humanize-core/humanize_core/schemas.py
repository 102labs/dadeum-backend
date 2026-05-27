from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DocumentType = Literal[
    "auto",
    "report",
    "formal",
    "email",
    "proposal",
    "meeting_notes",
    "blog",
    "column",
]
Intensity = Literal["conservative", "standard", "strong"]
Concision = Literal["preserve", "tighten", "compact"]
Tone = Literal["keep", "neutral", "formal", "executive", "friendly"]
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


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    document_type: DocumentType
    intensity: Intensity
    concision: Concision
    tone: Tone
    intent: Literal["business_polish"]
    protected_terms: list[str] = Field(default_factory=list)
    quality_mode: Literal["balanced"] = "balanced"
    rewrite_mode: RewriteMode = "fast"
    focus_categories: list[str] = Field(default_factory=list)
    max_rounds: int = Field(default=1, ge=1, le=3)
    preserve_formatting: bool = True


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
