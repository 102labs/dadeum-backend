import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from humanize_core.config import Settings
from humanize_core.diff import build_display_safe_changes
from humanize_core.im_not_ai.audit import (
    build_audit_warnings,
    change_rate,
    finding_category_summary,
    finding_density,
    finding_score,
    local_detect,
    mark_high_risk_if_needed,
    over_polish_signals,
    quality_grade,
    self_check_items,
    split_sentences,
    sentence_length_stats,
)
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    DetectionResult,
    FlaggedEdit,
    FastRewriteResult,
    HumanizeContext,
    RewriteEdit,
    StrictReviewResult,
    StrictRewriteResult,
)
from humanize_core.llm import RewriteLLM
from humanize_core.schemas import Change, LLMRewriteResult, RewriteRequest, RewriteResponse, Usage


class InputLimitError(ValueError):
    pass


_INCOMPLETE_WARNING_RE = re.compile(
    r"truncat|incomplete|cut(?:s)?\s*off|mid-sentence|interrupted|"
    r"잘렸|잘림|중간에\s*끊|불완전|미완성|완료되지",
    re.IGNORECASE,
)


class RewriteState(TypedDict, total=False):
    request: RewriteRequest
    selected_mode: str
    humanize_context: dict
    warnings: list[str]
    detection: DetectionResult
    strict_rewrite_result: StrictRewriteResult
    audit_result: AuditResult
    final_audit_result: AuditResult
    review_result: StrictReviewResult
    round: int
    llm_result: LLMRewriteResult
    display_safe_llm_result: LLMRewriteResult
    response: RewriteResponse
    started_at: float


class RewriteGraphRunner:
    def __init__(self, settings: Settings, llm: RewriteLLM) -> None:
        self.settings = settings
        self.llm = llm
        self.graph = self._build_graph()

    async def run(self, request: RewriteRequest) -> RewriteResponse:
        state = await self.graph.ainvoke({"request": request, "started_at": time.perf_counter()})
        return state["response"]

    def _build_graph(self):
        builder = StateGraph(RewriteState)
        builder.add_node("prepare", self._prepare)
        builder.add_node("fast_rewrite", self._fast_rewrite)
        builder.add_node("fast_audit", self._fast_audit)
        builder.add_node("detect", self._detect)
        builder.add_node("strict_rewrite", self._strict_rewrite)
        builder.add_node("strict_audit", self._strict_audit)
        builder.add_node("review", self._review)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "prepare")
        builder.add_conditional_edges(
            "prepare",
            _route_after_prepare,
            {"fast": "fast_rewrite", "strict": "detect"},
        )
        builder.add_edge("fast_rewrite", "fast_audit")
        builder.add_edge("fast_audit", "finalize")
        builder.add_edge("detect", "strict_rewrite")
        builder.add_edge("strict_rewrite", "strict_audit")
        builder.add_edge("strict_audit", "review")
        builder.add_edge("review", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    async def _prepare(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        text_length = len(request.text)
        if text_length > self.settings.max_chars:
            raise InputLimitError(f"text length exceeds HUMANIZE_MAX_CHARS ({self.settings.max_chars})")

        selected_mode = request.rewrite_mode
        warnings: list[str] = []

        context = HumanizeContext().model_dump()

        return {
            "selected_mode": selected_mode,
            "humanize_context": context,
            "warnings": warnings,
            "round": 0,
        }

    async def _fast_rewrite(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        if hasattr(self.llm, "rewrite_fast"):
            fast_result: FastRewriteResult = await self.llm.rewrite_fast(  # type: ignore[attr-defined]
                request,
                state["humanize_context"],
            )
            residual = _local_detection(
                request,
                text=fast_result.revisedText,
            )
            self_check = fast_result.selfCheck or [
                item for item in self_check_items(
                    request.text,
                    fast_result.revisedText,
                    [],
                    residual.findings,
                )
            ]
            warnings = list(state.get("warnings", []))
            warnings.extend(fast_result.warnings)
            rate = fast_result.changeRate or change_rate(request.text, fast_result.revisedText)
            grade, grade_reason = quality_grade(request.text, fast_result.revisedText, residual.findings, self_check)
            failed_count = sum(1 for item in self_check if not item.passed)
            if failed_count:
                warnings.append(f"Fast 자체검증 미통과 항목이 {failed_count}건 있습니다.")
            if residual.findings:
                s1_count = sum(1 for finding in residual.findings if finding.severity == "S1")
                if s1_count:
                    warnings.append(f"Fast 결과에 S1 AI 티 패턴이 {s1_count}건 남아 있을 수 있습니다.")
            if grade in {"C", "D"}:
                warnings.append(f"Fast 등급 {grade}: {grade_reason}")
            rollback_required = fast_result.rollbackRequired or rate > 50
            revised_text = fast_result.revisedText
            changes = fast_result.changes
            summary = list(fast_result.summary)
            summary.append(f"Fast 등급 {grade}, 자체검증 {len(self_check) - failed_count}/{len(self_check)} 통과, 변경률 {rate:.2f}%.")
            if rollback_required:
                revised_text = request.text
                changes = [
                    Change(
                        original="fast_rewrite_draft",
                        revised="original_text",
                        reason="Fast 변경률이 50%를 초과해 원문 보존 버전으로 롤백했습니다.",
                        type="meaning",
                        riskLevel="high",
                    )
                ]
                warnings.append(f"Fast 변경률이 {rate:.2f}%로 50%를 초과해 결과를 원문으로 롤백했습니다.")
                summary.append("과윤문 가드가 작동해 LLM 초안을 폐기했습니다.")
            llm_result = LLMRewriteResult(
                revisedText=revised_text,
                changes=changes,
                summary=summary,
                inputTokens=fast_result.inputTokens,
                outputTokens=fast_result.outputTokens,
            )
            return {"llm_result": llm_result, "warnings": _dedupe(warnings), "round": 1}

        llm_result = await self.llm.rewrite(request)
        return {"llm_result": llm_result}

    async def _fast_audit(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        llm_result = state["llm_result"]
        warnings = list(state.get("warnings", []))

        local_warnings = build_audit_warnings(request.text, llm_result.revisedText, [])
        warnings.extend(local_warnings)
        llm_result.changes = mark_high_risk_if_needed(llm_result.changes, bool(local_warnings))

        return {"warnings": _dedupe(warnings), "llm_result": llm_result}

    async def _detect(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        local = _local_detection(request)
        if hasattr(self.llm, "detect"):
            model_detection: DetectionResult = await self.llm.detect(  # type: ignore[attr-defined]
                request,
                state["humanize_context"],
            )
            detection = _merge_detections(request.text, local, model_detection)
        else:
            detection = local
        return {"detection": detection}

    async def _strict_rewrite(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        round_number = 1

        if hasattr(self.llm, "rewrite_strict"):
            strict_result: StrictRewriteResult = await self.llm.rewrite_strict(  # type: ignore[attr-defined]
                request,
                state["humanize_context"],
                state["detection"],
            )
            strict_result = _enrich_strict_rewrite_result(strict_result, request, state["detection"])
            llm_result = LLMRewriteResult(
                revisedText=strict_result.revisedText,
                changes=strict_result.changes,
                summary=strict_result.summary,
                inputTokens=strict_result.inputTokens,
                outputTokens=strict_result.outputTokens,
            )
            return {
                "strict_rewrite_result": strict_result,
                "llm_result": llm_result,
                "round": round_number,
            }

        llm_result = await self.llm.rewrite(request)
        strict_result = StrictRewriteResult(
            revisedText=llm_result.revisedText,
            changes=llm_result.changes,
            summary=llm_result.summary,
            inputTokens=llm_result.inputTokens,
            outputTokens=llm_result.outputTokens,
        )
        strict_result = _enrich_strict_rewrite_result(strict_result, request, state["detection"])
        return {"strict_rewrite_result": strict_result, "llm_result": llm_result, "round": round_number}

    async def _strict_audit(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        llm_result = state["llm_result"]
        audit_result = await self._audit_candidate(
            request,
            state["humanize_context"],
            llm_result.revisedText,
            llm_result.changes,
        )
        llm_result.changes = mark_high_risk_if_needed(
            llm_result.changes,
            audit_result.status == "fail" or _strict_audit_requires_rollback(audit_result),
        )
        return {"audit_result": audit_result, "llm_result": llm_result}

    async def _audit_candidate(
        self,
        request: RewriteRequest,
        context: dict,
        revised_text: str,
        changes: list[Change],
    ) -> AuditResult:
        local_warnings = _strict_completion_warnings(request, revised_text)
        local_flagged = _local_flagged_edits_from_warnings(local_warnings)

        if hasattr(self.llm, "audit"):
            model_audit: AuditResult = await self.llm.audit(  # type: ignore[attr-defined]
                request,
                context,
                revised_text,
                [change.model_dump() for change in changes],
            )
            model_completion_warnings = _strict_completion_warnings(
                request,
                revised_text,
                model_audit.warnings,
            )
            local_warnings = _dedupe([*local_warnings, *model_completion_warnings])
            local_flagged = _local_flagged_edits_from_warnings(local_warnings)
            warnings = _dedupe([*local_warnings, *model_audit.warnings])
            flagged_edits = _merge_flagged_edits(local_flagged, model_audit.flaggedEdits)
            status = _audit_status(local_warnings, model_audit.status, flagged_edits)
            audit_result = model_audit.model_copy(
                update={
                    "warnings": warnings,
                    "status": status,
                    "flaggedEdits": flagged_edits,
                    "rollbackRequired": sum(1 for edit in flagged_edits if _flagged_edit_blocks(edit)),
                    "editsFlagged": len(flagged_edits),
                    "editsPassed": max(0, len(changes) - len(flagged_edits)),
                }
            )
            return audit_result

        status = _audit_status(local_warnings, "full_pass", local_flagged)
        return AuditResult(
            status=status,
            warnings=local_warnings,
            highRiskChangeIndexes=[0] if local_warnings and changes else [],
            flaggedEdits=local_flagged,
            rollbackRequired=sum(1 for edit in local_flagged if _flagged_edit_blocks(edit)),
            editsFlagged=len(local_flagged),
            editsPassed=max(0, len(changes) - len(local_flagged)),
            reason="로컬 완성도 감사 결과입니다.",
        )

    async def _review(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        llm_result = state["llm_result"]
        residual = _local_detection(
            request,
            text=llm_result.revisedText,
        )

        if hasattr(self.llm, "review"):
            review_result: StrictReviewResult = await self.llm.review(  # type: ignore[attr-defined]
                request,
                state["humanize_context"],
                state["detection"],
                llm_result.revisedText,
                state["audit_result"],
                residual,
            )
        else:
            review_result = _local_strict_review(
                request,
                llm_result,
                residual,
                state["audit_result"],
            )
        review_result = _augment_review_with_local_quality_signals(request, review_result)
        final_llm_result = LLMRewriteResult(
            revisedText=review_result.revisedText,
            changes=review_result.changes,
            summary=review_result.summary,
            inputTokens=review_result.inputTokens,
            outputTokens=review_result.outputTokens,
        )
        final_audit_result = await self._audit_candidate(
            request,
            state["humanize_context"],
            final_llm_result.revisedText,
            final_llm_result.changes,
        )
        review_result = _merge_final_audit_into_review(review_result, final_audit_result)
        final_llm_result.changes = mark_high_risk_if_needed(
            final_llm_result.changes,
            final_audit_result.status == "fail" or _strict_audit_requires_rollback(final_audit_result),
        )
        update: RewriteState = {
            "review_result": review_result,
            "final_audit_result": final_audit_result,
            "llm_result": final_llm_result,
        }
        if _strict_candidate_is_display_safe(request, final_llm_result, final_audit_result, review_result):
            update["display_safe_llm_result"] = final_llm_result
        return update

    async def _finalize(self, state: RewriteState) -> RewriteState:
        llm_result = state["llm_result"]
        warnings = list(state.get("warnings", []))
        audit_result = state.get("audit_result")
        final_audit_result = state.get("final_audit_result")
        review_result = state.get("review_result")
        if state.get("selected_mode") != "strict" and audit_result:
            warnings.extend(audit_result.warnings)
        if state.get("selected_mode") == "strict":
            if review_result:
                warnings.extend(review_result.warnings)
            if final_audit_result:
                warnings.extend(final_audit_result.warnings)
        elif review_result:
            warnings.extend(review_result.warnings)
        if state.get("selected_mode") == "strict":
            llm_result, warnings = _apply_strict_terminal_safety(state, llm_result, warnings)
        display_safe_changes = build_display_safe_changes(
            state["request"].text,
            llm_result.revisedText,
            llm_result.changes,
        )
        latency_ms = int((time.perf_counter() - state["started_at"]) * 1000)
        response = RewriteResponse(
            revisedText=llm_result.revisedText,
            changes=display_safe_changes,
            summary=llm_result.summary,
            warnings=_dedupe(warnings),
            usage=Usage(
                inputTokens=_sum_input_tokens(state),
                outputTokens=_sum_output_tokens(state),
                latencyMs=latency_ms,
                rounds=max(1, state.get("round", 1)),
            ),
        )
        return {"response": response}


def _route_after_prepare(state: RewriteState) -> str:
    return "strict" if state.get("selected_mode") == "strict" else "fast"


def _merge_detections(text: str, local: DetectionResult, model_detection: DetectionResult) -> DetectionResult:
    merged = list(local.findings)
    seen = {(finding.category, finding.start, finding.end, finding.textSpan) for finding in merged}
    for finding in model_detection.findings:
        key = (finding.category, finding.start, finding.end, finding.textSpan)
        if key not in seen:
            merged.append(finding)
            seen.add(key)
    return model_detection.model_copy(
        update={
            "findings": merged,
            "detectedCount": len(merged),
            "categorySummary": finding_category_summary(merged),
            "severityWeightedScore": finding_score(merged),
            "aiTellDensity": finding_density(text, merged),
            "sentenceCount": model_detection.sentenceCount or local.sentenceCount,
            "sentenceLengthStats": model_detection.sentenceLengthStats or local.sentenceLengthStats or sentence_length_stats(text),
            "inputTokens": model_detection.inputTokens,
            "outputTokens": model_detection.outputTokens,
        }
    )


def _local_detection(request: RewriteRequest, *, text: str | None = None) -> DetectionResult:
    return local_detect(
        request.text if text is None else text,
        None,
        [],
    )


def _audit_status(local_warnings: list[str], model_status: str, flagged_edits: list[FlaggedEdit]) -> str:
    if any(_is_strict_completion_warning(warning) for warning in local_warnings):
        return "fail"
    if any(_flagged_edit_blocks(edit) for edit in flagged_edits):
        return "conditional_pass" if model_status == "full_pass" else model_status
    if flagged_edits and model_status == "full_pass":
        return "conditional_pass"
    if local_warnings and model_status == "full_pass":
        return "conditional_pass"
    return model_status


def _enrich_strict_rewrite_result(
    strict_result: StrictRewriteResult,
    request: RewriteRequest,
    detection: DetectionResult,
) -> StrictRewriteResult:
    rate = strict_result.changeRate or change_rate(request.text, strict_result.revisedText)
    findings_resolved = strict_result.findingsResolved or strict_result.appliedFindingIds
    findings_unresolved = strict_result.findingsUnresolved or strict_result.unresolvedFindingIds
    if not findings_unresolved and findings_resolved:
        detected_ids = {finding.id for finding in detection.findings}
        findings_unresolved = sorted(detected_ids - set(findings_resolved))
    edits = strict_result.edits or [
        RewriteEdit(
            findingId=findings_resolved[index] if index < len(findings_resolved) else "",
            before=change.original,
            after=change.revised,
            category="",
            reason=change.reason,
            action="rewrite",
            changeRate=change_rate(change.original, change.revised),
        )
        for index, change in enumerate(strict_result.changes)
    ]
    summary = list(strict_result.summary)
    summary.append(f"Strict diff 메타: 변경률 {rate:.2f}%, resolved {len(findings_resolved)}, unresolved {len(findings_unresolved)}.")
    return strict_result.model_copy(
        update={
            "charCountBefore": strict_result.charCountBefore or len(request.text),
            "charCountAfter": strict_result.charCountAfter or len(strict_result.revisedText),
            "changeRate": rate,
            "findingsResolved": findings_resolved,
            "findingsUnresolved": findings_unresolved,
            "overPolishWarning": strict_result.overPolishWarning
            or _strict_over_polish_requires_retry(over_polish_signals(request.text, strict_result.revisedText)),
            "edits": edits,
            "summary": summary,
        }
    )


def _local_flagged_edits_from_warnings(local_warnings: list[str]) -> list[FlaggedEdit]:
    flagged: list[FlaggedEdit] = []
    for warning in local_warnings:
        if _is_strict_completion_warning(warning):
            flagged.append(
                FlaggedEdit(
                    issue=warning,
                    checklistFailed=[1, 4, 13],
                    action="restore_original",
                    correctionDirection="원문 전체를 다시 기준으로 삼아 누락 없이 완성본을 복원합니다.",
                    severity="high",
                )
            )
    return flagged


def _merge_flagged_edits(local_edits: list[FlaggedEdit], model_edits: list[FlaggedEdit]) -> list[FlaggedEdit]:
    merged: list[FlaggedEdit] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edit in [*local_edits, *model_edits]:
        key = (edit.findingId, edit.before, edit.after, edit.issue)
        if key in seen:
            continue
        merged.append(edit)
        seen.add(key)
    return merged


def _strict_over_polish_requires_retry(signals: list[str]) -> bool:
    blockers = _strict_over_polish_blockers(signals)
    return len(blockers) >= 2 or ("change_rate_over_50" in signals and bool(blockers))


def _strict_over_polish_blockers(signals: list[str]) -> list[str]:
    rate_only = {"change_rate_over_30", "change_rate_over_50"}
    return [signal for signal in signals if signal not in rate_only]


def _local_strict_review(
    request: RewriteRequest,
    llm_result: LLMRewriteResult,
    residual: DetectionResult,
    audit_result: AuditResult,
) -> StrictReviewResult:
    s1_count = sum(1 for finding in residual.findings if finding.severity == "S1")
    s2_count = sum(1 for finding in residual.findings if finding.severity == "S2")
    signals = over_polish_signals(request.text, llm_result.revisedText)
    warnings: list[str] = []
    if s1_count:
        warnings.append(f"잔존 S1 AI 티 패턴이 {s1_count}건 감지됐습니다.")
    if s2_count > 3:
        warnings.append(f"잔존 S2 AI 티 패턴이 {s2_count}건으로 strict 합격선을 넘었습니다.")
    if signals:
        warnings.append("과윤문 신호가 감지됐습니다: " + ", ".join(signals))
    blocking_issues: list[str] = []
    if audit_result.status == "fail" or _strict_audit_requires_rollback(audit_result):
        blocking_issues.append("초기 strict 감사에서 보수 수정이 필요한 항목이 남았습니다.")
    if _strict_over_polish_requires_retry(signals):
        blocking_issues.append("변경률 외 과윤문 신호가 함께 감지됐습니다.")
    summary = list(llm_result.summary)
    summary.append("Strict review: 감사 지시와 잔존 AI 티 패턴을 확인했습니다.")
    return StrictReviewResult(
        revisedText=llm_result.revisedText,
        changes=llm_result.changes,
        summary=summary,
        warnings=_dedupe(warnings),
        auditCorrectionsApplied=[
            edit.correctionDirection or edit.issue
            for edit in audit_result.flaggedEdits
            if edit.action != "warning"
        ],
        residualFindings=residual.findings,
        finalAuditStatus=audit_result.status,
        finalAuditWarnings=audit_result.warnings,
        finalBlockingIssues=blocking_issues,
        qualityLevel="B" if not blocking_issues and s1_count == 0 else "C",
        inputTokens=0,
        outputTokens=0,
    )


def _augment_review_with_local_quality_signals(
    request: RewriteRequest,
    review_result: StrictReviewResult,
) -> StrictReviewResult:
    residual = _local_detection(request, text=review_result.revisedText)
    s1_count = sum(1 for finding in residual.findings if finding.severity == "S1")
    s2_count = sum(1 for finding in residual.findings if finding.severity == "S2")
    signals = over_polish_signals(request.text, review_result.revisedText)

    warnings = list(review_result.warnings)
    if s1_count and not any("잔존 S1" in warning for warning in warnings):
        warnings.append(f"잔존 S1 AI 티 패턴이 {s1_count}건 감지됐습니다.")
    if s2_count > 3 and not any("잔존 S2" in warning for warning in warnings):
        warnings.append(f"잔존 S2 AI 티 패턴이 {s2_count}건으로 strict 합격선을 넘었습니다.")
    if signals and not any("과윤문 신호" in warning for warning in warnings):
        warnings.append("과윤문 신호가 감지됐습니다: " + ", ".join(signals))

    blocking = list(review_result.finalBlockingIssues)
    if _strict_over_polish_requires_retry(signals):
        blocking.append("변경률 외 과윤문 신호가 함께 감지됐습니다.")

    quality = review_result.qualityLevel
    if not quality:
        quality = "B" if not blocking and s1_count == 0 else "C"

    return review_result.model_copy(
        update={
            "warnings": _dedupe(warnings),
            "residualFindings": _merge_findings(review_result.residualFindings, residual.findings),
            "finalBlockingIssues": _dedupe(blocking),
            "qualityLevel": quality,
        }
    )


def _merge_findings(left: list, right: list) -> list:
    merged = []
    seen = set()
    for finding in [*left, *right]:
        key = (finding.category, finding.start, finding.end, finding.textSpan)
        if key in seen:
            continue
        merged.append(finding)
        seen.add(key)
    return merged


def _merge_final_audit_into_review(
    review_result: StrictReviewResult,
    final_audit_result: AuditResult,
) -> StrictReviewResult:
    blocking = list(review_result.finalBlockingIssues)
    if final_audit_result.status == "fail":
        blocking.append("최종 감사에서 원문 대비 누락 또는 의미 변화 가능성이 감지됐습니다.")
    if _strict_audit_requires_rollback(final_audit_result):
        blocking.extend(
            edit.correctionDirection or edit.issue
            for edit in final_audit_result.flaggedEdits
            if _flagged_edit_blocks(edit)
        )
    return review_result.model_copy(
        update={
            "finalAuditStatus": final_audit_result.status,
            "finalAuditWarnings": _dedupe([*review_result.finalAuditWarnings, *final_audit_result.warnings]),
            "finalBlockingIssues": _dedupe(blocking),
        }
    )


def _strict_candidate_is_display_safe(
    request: RewriteRequest,
    llm_result: LLMRewriteResult,
    audit_result: AuditResult,
    review_result: StrictReviewResult,
) -> bool:
    warnings = [*audit_result.warnings, *review_result.warnings, *review_result.finalAuditWarnings]
    if _has_strict_completion_warning(warnings):
        return False
    if _strict_completion_warnings(request, llm_result.revisedText):
        return False
    if audit_result.status == "fail" or _strict_audit_requires_rollback(audit_result):
        return False
    if review_result.finalAuditStatus == "fail" or review_result.finalBlockingIssues:
        return False
    return True


def _apply_strict_terminal_safety(
    state: RewriteState,
    llm_result: LLMRewriteResult,
    warnings: list[str],
) -> tuple[LLMRewriteResult, list[str]]:
    audit_result = state.get("final_audit_result") or state.get("audit_result")
    review_result = state.get("review_result")
    if audit_result is None:
        return llm_result, warnings

    completion_warnings = _strict_completion_warnings(
        state["request"],
        llm_result.revisedText,
        warnings,
    )
    all_warnings = _dedupe([*warnings, *completion_warnings])
    review_blocked = review_result is not None and bool(review_result.finalBlockingIssues)
    should_block = (
        bool(completion_warnings)
        or audit_result.status == "fail"
        or review_blocked
        or _strict_audit_requires_rollback(audit_result)
    )
    if not should_block:
        return llm_result, all_warnings

    fallback_warnings = _strict_fallback_warnings(state, all_warnings, review_blocked)
    request = state["request"]
    fallback = LLMRewriteResult(
        revisedText=request.text,
        changes=[
            Change(
                original="strict_rewrite_draft",
                revised="original_text",
                reason="Strict 최종 초안이 완성도/보존 안전 기준을 통과하지 못해 원문을 반환했습니다.",
                type="meaning",
                riskLevel="high",
            )
        ],
        summary=["Strict 최종 초안이 안전 기준을 통과하지 못해 원문을 반환했습니다."],
    )
    fallback_warnings.append("Strict가 안전한 윤문 결과를 만들지 못해 결과 노출을 차단하고 원문을 반환했습니다.")
    return fallback, _dedupe(fallback_warnings)


def _strict_fallback_warnings(
    state: RewriteState,
    discarded_warnings: list[str],
    review_blocked: bool,
) -> list[str]:
    warnings = list(state.get("warnings", []))
    if review_blocked:
        warnings.append("Strict 최종 후보가 감사/리뷰 기준을 통과하지 못해 사람 검토가 필요합니다.")
    if any(
        "누락" in warning
        or "의미" in warning
        or "고유명사" in warning
        or "인용" in warning
        for warning in discarded_warnings
    ):
        warnings.append("Strict 최종 후보에서 원문 대비 누락 또는 의미 보존 위험이 감지됐습니다.")
    if any(_is_strict_completion_warning(warning) for warning in discarded_warnings):
        warnings.append("Strict 최종 후보에서 출력 잘림 또는 문장·문단 누락 신호가 감지됐습니다.")
    if any("과윤문 신호" in warning for warning in discarded_warnings):
        warnings.append("Strict 최종 후보에서 과윤문 신호가 감지됐습니다.")
    if any("잔존 S1" in warning for warning in discarded_warnings):
        warnings.append("Strict 최종 후보에 잔존 S1 AI 티 패턴이 있었습니다.")
    return _dedupe(warnings)


def _strict_audit_requires_rollback(audit_result: AuditResult) -> bool:
    return audit_result.rollbackRequired > 0 or any(
        _flagged_edit_blocks(edit) for edit in audit_result.flaggedEdits
    )


def _flagged_edit_blocks(edit: FlaggedEdit) -> bool:
    return edit.action in {"restore_original", "preserve_exact", "rollback_required"} or (
        edit.action in {"rewrite_required", "rewrite_with_hedge_preserved"} and edit.severity == "high"
    )


def _strict_completion_warnings(
    request: RewriteRequest,
    revised_text: str,
    model_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    original = request.text.strip()
    revised = revised_text.strip()
    if not revised:
        warnings.append("Strict 결과가 비어 있어 완성도 검증을 통과하지 못했습니다.")
        return warnings

    if _model_reports_incomplete(model_warnings or []):
        warnings.append("Strict 결과가 중간에 잘렸거나 불완전하다는 감사 신호가 감지됐습니다.")

    if len(original) >= 200:
        length_ratio = len(revised) / max(len(original), 1)
        if length_ratio < 0.60:
            warnings.append("Strict 결과가 원문 대비 지나치게 짧아 누락 또는 출력 잘림 가능성이 큽니다.")

        original_sentences = split_sentences(original)
        revised_sentences = split_sentences(revised)
        if len(original_sentences) >= 4 and len(revised_sentences) / max(len(original_sentences), 1) < 0.50:
            warnings.append("Strict 결과의 문장 커버리지가 원문 대비 낮아 전체 내용을 반영하지 못했을 수 있습니다.")

        if request.preserve_formatting:
            original_paragraphs = _paragraphs(original)
            revised_paragraphs = _paragraphs(revised)
            if len(original_paragraphs) >= 3 and len(revised_paragraphs) / max(len(original_paragraphs), 1) < 0.50:
                warnings.append("Strict 결과의 문단 커버리지가 원문 대비 낮아 일부 문단이 누락됐을 수 있습니다.")

    if _looks_cut_off_mid_sentence(original, revised):
        warnings.append("Strict 결과가 문장 중간에서 끝난 것으로 보여 완성도 검증을 통과하지 못했습니다.")

    return _dedupe(warnings)


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _looks_cut_off_mid_sentence(original: str, revised: str) -> bool:
    if len(original) < 120 or len(revised) >= len(original) * 0.90:
        return False
    original_ends = original.rstrip().endswith((".", "!", "?", "。", "！", "？", "다.", "요."))
    revised_ends = revised.rstrip().endswith((".", "!", "?", "。", "！", "？"))
    return original_ends and not revised_ends


def _model_reports_incomplete(warnings: list[str]) -> bool:
    return any(_INCOMPLETE_WARNING_RE.search(warning) for warning in warnings)


def _has_strict_completion_warning(warnings: list[str]) -> bool:
    return any(_is_strict_completion_warning(warning) for warning in warnings)


def _is_strict_completion_warning(warning: str) -> bool:
    return warning.startswith("Strict 결과가") or _INCOMPLETE_WARNING_RE.search(warning) is not None


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _sum_input_tokens(state: RewriteState) -> int:
    llm_tokens = 0 if state.get("strict_rewrite_result") else _token_value(state.get("llm_result"), "inputTokens")
    return (
        _token_value(state.get("detection"), "inputTokens")
        + _token_value(state.get("strict_rewrite_result"), "inputTokens")
        + _token_value(state.get("audit_result"), "inputTokens")
        + _token_value(state.get("review_result"), "inputTokens")
        + _token_value(state.get("final_audit_result"), "inputTokens")
        + llm_tokens
    )


def _sum_output_tokens(state: RewriteState) -> int:
    llm_tokens = 0 if state.get("strict_rewrite_result") else _token_value(state.get("llm_result"), "outputTokens")
    return (
        _token_value(state.get("detection"), "outputTokens")
        + _token_value(state.get("strict_rewrite_result"), "outputTokens")
        + _token_value(state.get("audit_result"), "outputTokens")
        + _token_value(state.get("review_result"), "outputTokens")
        + _token_value(state.get("final_audit_result"), "outputTokens")
        + llm_tokens
    )


def _token_value(value: object | None, attr: str) -> int:
    return int(getattr(value, attr, 0) or 0)
