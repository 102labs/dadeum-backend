from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from humanize_core.schemas import Change, ChangeType, RiskLevel


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


class RulebookHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    categoryLabel: str
    severity: Severity
    scope: FindingScope
    suggestedFix: str
    occurrences: int = 1
    # Short sample expressions matched in the source. Never populated from
    # spans that overlap protected values; kept out of logs by count-only
    # stage details.
    matches: list[str] = Field(default_factory=list)


class HumanizeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detectedCount: int = 0
    severityWeightedScore: float = 0.0
    categorySummary: dict[str, int] = Field(default_factory=dict)
    rulebookHints: list[RulebookHint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model-facing output schemas. These are what the structured-output request
# actually asks the model to produce: only the fields the graph consumes.
# Internal bookkeeping (token usage, quality grades, residual findings) lives
# on the *Result models above and is filled in by code, never by the model.


class RewriteOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    revisedText: str
    changes: list[Change]
    summary: list[str]
    warnings: list[str] = Field(default_factory=list)


class AuditOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: AuditStatus
    reason: str
    warnings: list[str] = Field(default_factory=list)
    flaggedEdits: list[FlaggedEdit] = Field(default_factory=list)


class ReviewSegment(BaseModel):
    """One draft sentence the audit flagged, with the corrections to apply."""

    model_config = ConfigDict(extra="forbid")

    index: int
    draft_sentence: str
    original_sentence: str = ""
    corrections: list[FlaggedEdit] = Field(default_factory=list)


class RepairedSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    text: str


class SegmentReviewOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    repairedSegments: list[RepairedSegment]
    unresolved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SegmentReviewResult(SegmentReviewOutput):
    inputTokens: int = 0
    outputTokens: int = 0


class ChangeExplanation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    reason: str
    type: ChangeType
    riskLevel: RiskLevel


class ChangeExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ChangeExplanation]
    summary: list[str] = Field(default_factory=list)


class ChangeExplanationResult(ChangeExplanationOutput):
    inputTokens: int = 0
    outputTokens: int = 0
