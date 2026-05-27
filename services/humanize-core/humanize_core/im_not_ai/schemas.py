from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from humanize_core.schemas import Change


Severity = Literal["S1", "S2", "S3"]
FindingScope = Literal["span", "document"]
AuditStatus = Literal["full_pass", "conditional_pass", "fail"]
FlaggedEditAction = Literal["rollback_required", "rewrite_with_hedge_preserved", "warning"]
ReviewDecision = Literal[
    "accept",
    "accept_with_note",
    "rewrite_round_2",
    "rollback_and_rewrite",
    "hold_and_report",
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

    estimatedGenre: str
    sentenceCount: int = 0
    sentenceLengthStats: dict[str, float | bool] = Field(default_factory=dict)
    detectedCount: int = 0
    aiTellDensity: float = 0.0
    severityWeightedScore: float = 0.0
    categorySummary: dict[str, int] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    inputTokens: int = 0
    outputTokens: int = 0


class FastRewriteResult(BaseModel):
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


class RewriteEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findingId: str = ""
    before: str
    after: str
    category: str = ""
    reason: str
    action: str = ""
    changeRate: float = 0.0


class StrictRewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revisedText: str
    changes: list[Change]
    summary: list[str]
    appliedFindingIds: list[str] = Field(default_factory=list)
    unresolvedFindingIds: list[str] = Field(default_factory=list)
    charCountBefore: int = 0
    charCountAfter: int = 0
    changeRate: float = 0.0
    findingsResolved: list[str] = Field(default_factory=list)
    findingsUnresolved: list[str] = Field(default_factory=list)
    overPolishWarning: bool = False
    edits: list[RewriteEdit] = Field(default_factory=list)
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


class NaturalnessReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    warnings: list[str] = Field(default_factory=list)
    residualFindings: list[Finding] = Field(default_factory=list)
    scoreBefore: float = 0.0
    scoreAfter: float = 0.0
    scoreImprovement: float = 0.0
    s1Residual: int = 0
    s2Residual: int = 0
    overPolishSignals: list[str] = Field(default_factory=list)
    qualityLevel: str = ""
    overPolishFindings: list[str] = Field(default_factory=list)
    unclassifiedCandidates: list[str] = Field(default_factory=list)
    targetFindingIds: list[str] = Field(default_factory=list)
    reason: str
    inputTokens: int = 0
    outputTokens: int = 0


class HumanizeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimatedGenre: str
    metricsBefore: dict = Field(default_factory=dict)
    protectedTerms: list[str] = Field(default_factory=list)
    preservationTerms: list[str] = Field(default_factory=list)
