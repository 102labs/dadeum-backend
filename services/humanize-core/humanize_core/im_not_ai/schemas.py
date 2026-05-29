from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from humanize_core.schemas import Change


Severity = Literal["S1", "S2", "S3"]
FindingScope = Literal["span", "document"]
AuditStatus = Literal["full_pass", "conditional_pass", "fail"]
FlaggedEditAction = Literal[
    "rewrite_required",
    "restore_original",
    "preserve_exact",
    "warning",
    # Backward-compatible values accepted from older model prompts/tests.
    "rollback_required",
    "rewrite_with_hedge_preserved",
]


class SelfCheckItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    note: str


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    categoryLabel: str
    severity: Severity
    scope: FindingScope
    textSpan: str = ""
    start: int | None = None
    end: int | None = None
    reason: str
    suggestedFix: str


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentenceCount: int = 0
    sentenceLengthStats: dict[str, float | bool] = Field(default_factory=dict)
    detectedCount: int = 0
    aiTellDensity: float = 0.0
    severityWeightedScore: float = 0.0
    categorySummary: dict[str, int] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    inputTokens: int = 0
    outputTokens: int = 0


class RewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revisedText: str
    changes: list[Change]
    summary: list[str]
    warnings: list[str] = Field(default_factory=list)
    selfCheck: list[SelfCheckItem] = Field(default_factory=list)
    residualFindings: list[Finding] = Field(default_factory=list)
    qualityLevel: str = ""
    changeRate: float = 0.0
    rollbackRequired: bool = False
    inputTokens: int = 0
    outputTokens: int = 0


class FlaggedEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findingId: str = ""
    before: str = ""
    after: str = ""
    issue: str
    checklistFailed: list[int] = Field(default_factory=list)
    action: FlaggedEditAction
    correctionDirection: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AuditStatus
    warnings: list[str] = Field(default_factory=list)
    highRiskChangeIndexes: list[int] = Field(default_factory=list)
    flaggedEdits: list[FlaggedEdit] = Field(default_factory=list)
    rollbackRequired: int = 0
    editsPassed: int = 0
    editsFlagged: int = 0
    reason: str
    inputTokens: int = 0
    outputTokens: int = 0


class StrictReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revisedText: str
    changes: list[Change]
    summary: list[str]
    warnings: list[str] = Field(default_factory=list)
    auditCorrectionsApplied: list[str] = Field(default_factory=list)
    residualFindings: list[Finding] = Field(default_factory=list)
    finalAuditStatus: AuditStatus = "full_pass"
    finalAuditWarnings: list[str] = Field(default_factory=list)
    finalBlockingIssues: list[str] = Field(default_factory=list)
    qualityLevel: str = ""
    inputTokens: int = 0
    outputTokens: int = 0


class HumanizeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
