import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from humanize_core.config import Settings
from humanize_core.im_not_ai.audit import (
    build_audit_warnings,
    change_rate,
    collect_preservation_terms,
    compute_metrics,
    estimate_genre,
    finding_category_summary,
    finding_density,
    finding_score,
    local_detect,
    mark_high_risk_if_needed,
    over_polish_signals,
    quality_grade,
    score_improvement,
    self_check_items,
    sentence_length_stats,
    strict_quality_grade,
)
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    DetectionResult,
    FlaggedEdit,
    FastRewriteResult,
    HumanizeContext,
    NaturalnessReviewResult,
    RewriteEdit,
    StrictRewriteResult,
)
from humanize_core.llm import RewriteLLM
from humanize_core.schemas import Change, LLMRewriteResult, RewriteRequest, RewriteResponse, Usage


class InputLimitError(ValueError):
    pass


class RewriteState(TypedDict, total=False):
    request: RewriteRequest
    prepared_document_type: str
    selected_mode: str
    humanize_context: dict
    warnings: list[str]
    detection: DetectionResult
    strict_rewrite_result: StrictRewriteResult
    audit_result: AuditResult
    review_result: NaturalnessReviewResult
    round: int
    max_rounds: int
    llm_result: LLMRewriteResult
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
        builder.add_conditional_edges(
            "review",
            _route_after_review,
            {"rewrite": "strict_rewrite", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    async def _prepare(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        text_length = len(request.text)
        if text_length > self.settings.max_chars:
            raise InputLimitError(f"text length exceeds HUMANIZE_MAX_CHARS ({self.settings.max_chars})")

        document_type = request.document_type
        if document_type == "auto":
            document_type = _infer_document_type(request.text)

        warnings: list[str] = []
        selected_mode = request.rewrite_mode
        if text_length > 8_000 and selected_mode == "fast":
            selected_mode = "strict"
            warnings.append("입력 길이가 8,000자를 넘어 strict 모드로 자동 전환했습니다.")

        genre = estimate_genre(document_type, request.text)
        metrics_before = compute_metrics(request.text, genre)
        preservation_terms = collect_preservation_terms(request.text, request.protected_terms)
        context = HumanizeContext(
            estimatedGenre=genre,
            metricsBefore=metrics_before,
            protectedTerms=request.protected_terms,
            preservationTerms=preservation_terms,
        ).model_dump()
        max_rounds = self.settings.strict_max_rounds if selected_mode == "strict" else 1
        if request.max_rounds > 1:
            max_rounds = min(request.max_rounds, self.settings.strict_max_rounds)

        return {
            "prepared_document_type": document_type,
            "selected_mode": selected_mode,
            "humanize_context": context,
            "warnings": warnings,
            "round": 0,
            "max_rounds": max_rounds,
        }

    async def _fast_rewrite(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        document_type = state["prepared_document_type"]
        if hasattr(self.llm, "rewrite_fast"):
            fast_result: FastRewriteResult = await self.llm.rewrite_fast(  # type: ignore[attr-defined]
                request,
                document_type,
                state["humanize_context"],
            )
            residual = local_detect(
                fast_result.revisedText,
                state["humanize_context"]["estimatedGenre"],
                request.focus_categories,
                request.protected_terms,
            )
            self_check = fast_result.selfCheck or [
                item for item in self_check_items(
                    request.text,
                    fast_result.revisedText,
                    request.protected_terms,
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

        llm_result = await self.llm.rewrite(request, document_type)
        return {"llm_result": llm_result}

    async def _fast_audit(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        llm_result = state["llm_result"]
        warnings = list(state.get("warnings", []))

        local_warnings = build_audit_warnings(request.text, llm_result.revisedText, request.protected_terms)
        warnings.extend(local_warnings)
        llm_result.changes = mark_high_risk_if_needed(llm_result.changes, bool(local_warnings))

        return {"warnings": _dedupe(warnings), "llm_result": llm_result}

    async def _detect(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        local = local_detect(
            request.text,
            state["humanize_context"]["estimatedGenre"],
            request.focus_categories,
            request.protected_terms,
        )
        if hasattr(self.llm, "detect"):
            model_detection: DetectionResult = await self.llm.detect(  # type: ignore[attr-defined]
                request,
                state["prepared_document_type"],
                state["humanize_context"],
            )
            detection = _merge_detections(request.text, local, model_detection)
        else:
            detection = local
        return {"detection": detection}

    async def _strict_rewrite(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        document_type = state["prepared_document_type"]
        round_number = state.get("round", 0) + 1
        previous = state.get("llm_result").revisedText if state.get("llm_result") else None
        audit_feedback = state.get("audit_result").warnings if state.get("audit_result") else []
        review_feedback = state.get("review_result").warnings if state.get("review_result") else []
        use_escalation = round_number >= state.get("max_rounds", self.settings.strict_max_rounds)

        if hasattr(self.llm, "rewrite_strict"):
            strict_result: StrictRewriteResult = await self.llm.rewrite_strict(  # type: ignore[attr-defined]
                request,
                document_type,
                state["humanize_context"],
                state["detection"],
                previous,
                audit_feedback,
                review_feedback,
                use_escalation=use_escalation,
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

        llm_result = await self.llm.rewrite(request, document_type)
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
        local_warnings = build_audit_warnings(request.text, llm_result.revisedText, request.protected_terms)
        local_flagged = _local_flagged_edits(request, llm_result, local_warnings)

        if hasattr(self.llm, "audit"):
            model_audit: AuditResult = await self.llm.audit(  # type: ignore[attr-defined]
                request,
                state["prepared_document_type"],
                state["humanize_context"],
                llm_result.revisedText,
                [change.model_dump() for change in llm_result.changes],
            )
            warnings = _dedupe([*local_warnings, *model_audit.warnings])
            flagged_edits = _merge_flagged_edits(local_flagged, model_audit.flaggedEdits)
            status = _audit_status(local_warnings, model_audit.status, flagged_edits)
            audit_result = model_audit.model_copy(
                update={
                    "warnings": warnings,
                    "status": status,
                    "flaggedEdits": flagged_edits,
                    "rollbackRequired": sum(1 for edit in flagged_edits if edit.action == "rollback_required"),
                    "editsFlagged": len(flagged_edits),
                    "editsPassed": max(0, len(llm_result.changes) - len(flagged_edits)),
                }
            )
        else:
            status = _audit_status(local_warnings, "full_pass", local_flagged)
            audit_result = AuditResult(
                status=status,
                warnings=local_warnings,
                highRiskChangeIndexes=[0] if local_warnings and llm_result.changes else [],
                flaggedEdits=local_flagged,
                rollbackRequired=sum(1 for edit in local_flagged if edit.action == "rollback_required"),
                editsFlagged=len(local_flagged),
                editsPassed=max(0, len(llm_result.changes) - len(local_flagged)),
                reason="로컬 보존 감사 결과입니다.",
            )

        llm_result.changes = mark_high_risk_if_needed(
            llm_result.changes,
            audit_result.status != "full_pass" or bool(audit_result.highRiskChangeIndexes),
        )
        return {"audit_result": audit_result, "llm_result": llm_result}

    async def _review(self, state: RewriteState) -> RewriteState:
        request = state["request"]
        llm_result = state["llm_result"]
        residual = local_detect(
            llm_result.revisedText,
            state["humanize_context"]["estimatedGenre"],
            request.focus_categories,
            request.protected_terms,
        )
        audit_warnings = state["audit_result"].warnings
        signals = over_polish_signals(request.text, llm_result.revisedText, state["humanize_context"]["estimatedGenre"])

        if hasattr(self.llm, "review"):
            model_review: NaturalnessReviewResult = await self.llm.review(  # type: ignore[attr-defined]
                request,
                state["prepared_document_type"],
                state["humanize_context"],
                state["detection"],
                llm_result.revisedText,
                audit_warnings,
            )
            review_result = _combine_review(
                model_review,
                state["detection"],
                residual,
                state["audit_result"],
                state["round"],
                state["max_rounds"],
                signals,
            )
        else:
            review_result = _local_review(
                state["detection"],
                residual,
                state["audit_result"],
                state["round"],
                state["max_rounds"],
                signals,
            )
        return {"review_result": review_result}

    async def _finalize(self, state: RewriteState) -> RewriteState:
        llm_result = state["llm_result"]
        warnings = list(state.get("warnings", []))
        audit_result = state.get("audit_result")
        review_result = state.get("review_result")
        if audit_result:
            warnings.extend(audit_result.warnings)
        if review_result:
            warnings.extend(review_result.warnings)
            if review_result.decision == "hold_and_report":
                warnings.append("Strict 검증이 최대 라운드 안에 완료되지 않아 사람 검토가 필요합니다.")
        latency_ms = int((time.perf_counter() - state["started_at"]) * 1000)
        response = RewriteResponse(
            revisedText=llm_result.revisedText,
            changes=llm_result.changes,
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


def _infer_document_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bdear\b|안녕하세요|감사합니다|regards\b", lowered):
        return "email"
    if re.search(r"회의|참석자|안건|minutes|meeting", lowered):
        return "meeting_notes"
    if re.search(r"제안|견적|proposal|scope|deliverable", lowered):
        return "proposal"
    if re.search(r"보고|분석|결과|report|metric", lowered):
        return "report"
    if re.search(r"블로그|구독|독자|blog", lowered):
        return "blog"
    return "formal"


def _route_after_prepare(state: RewriteState) -> str:
    return "strict" if state.get("selected_mode") == "strict" else "fast"


def _route_after_review(state: RewriteState) -> str:
    review = state["review_result"]
    if review.decision in {"rewrite_round_2", "rollback_and_rewrite"} and state["round"] < state["max_rounds"]:
        return "rewrite"
    return "finalize"


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


def _audit_status(local_warnings: list[str], model_status: str, flagged_edits: list[FlaggedEdit]) -> str:
    if any("50%" in warning or "누락" in warning for warning in local_warnings):
        return "fail"
    if any(edit.action == "rollback_required" for edit in flagged_edits):
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
            "overPolishWarning": strict_result.overPolishWarning or rate > 30,
            "edits": edits,
            "summary": summary,
        }
    )


def _local_flagged_edits(
    request: RewriteRequest,
    llm_result: LLMRewriteResult,
    local_warnings: list[str],
) -> list[FlaggedEdit]:
    flagged: list[FlaggedEdit] = []
    for warning in local_warnings:
        if "누락" in warning:
            flagged.append(
                FlaggedEdit(
                    issue=warning,
                    checklistFailed=[1, 2, 3, 4, 13],
                    action="rollback_required",
                )
            )
        elif "50%" in warning:
            flagged.append(
                FlaggedEdit(
                    issue=warning,
                    checklistFailed=[7, 10, 13],
                    action="rollback_required",
                )
            )
        elif "30%" in warning:
            flagged.append(
                FlaggedEdit(
                    issue=warning,
                    checklistFailed=[13],
                    action="warning",
                )
            )
    if not flagged:
        rate = change_rate(request.text, llm_result.revisedText)
        if rate > 30:
            flagged.append(
                FlaggedEdit(
                    issue=f"변경률이 {rate:.2f}%로 원본 strict 경고 기준을 넘었습니다.",
                    checklistFailed=[13],
                    action="warning",
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


def _combine_review(
    model_review: NaturalnessReviewResult,
    original_detection: DetectionResult,
    residual: DetectionResult,
    audit_result: AuditResult,
    round_number: int,
    max_rounds: int,
    signals: list[str],
) -> NaturalnessReviewResult:
    s1_count = sum(1 for finding in residual.findings if finding.severity == "S1")
    s2_count = sum(1 for finding in residual.findings if finding.severity == "S2")
    improvement = score_improvement(original_detection.severityWeightedScore, residual.severityWeightedScore)
    quality = strict_quality_grade(
        s1_count=s1_count,
        s2_count=s2_count,
        improvement=improvement,
        over_polish_signal_count=len(signals),
    )
    warnings = list(model_review.warnings)
    if s1_count:
        warnings.append(f"잔존 S1 AI 티 패턴이 {s1_count}건 감지됐습니다.")
    if s2_count > 3:
        warnings.append(f"잔존 S2 AI 티 패턴이 {s2_count}건으로 strict 합격선을 넘었습니다.")
    if signals:
        warnings.append("과윤문 신호가 감지됐습니다: " + ", ".join(signals))
    if audit_result.status in {"fail", "conditional_pass"}:
        decision = "rollback_and_rewrite" if round_number < max_rounds else "hold_and_report"
    elif len(signals) >= 2 and round_number < max_rounds:
        decision = "rollback_and_rewrite"
    elif len(signals) >= 2:
        decision = "hold_and_report"
    elif s1_count and round_number < max_rounds:
        decision = "rewrite_round_2"
    elif s2_count > 3 and round_number < max_rounds:
        decision = "rewrite_round_2"
    elif s1_count:
        decision = "hold_and_report"
    elif s2_count > 3:
        decision = "hold_and_report"
    elif model_review.decision in {"rewrite_round_2", "rollback_and_rewrite"} and round_number >= max_rounds:
        decision = "hold_and_report"
    elif s2_count:
        decision = "accept_with_note"
    else:
        decision = model_review.decision
    residual_findings = _merge_findings(model_review.residualFindings, residual.findings)
    return model_review.model_copy(
        update={
            "decision": decision,
            "warnings": _dedupe(warnings),
            "residualFindings": residual_findings,
            "scoreBefore": original_detection.severityWeightedScore,
            "scoreAfter": residual.severityWeightedScore,
            "scoreImprovement": improvement,
            "s1Residual": s1_count,
            "s2Residual": s2_count,
            "overPolishSignals": _dedupe([*model_review.overPolishSignals, *signals]),
            "qualityLevel": model_review.qualityLevel or quality,
            "targetFindingIds": [finding.id for finding in residual.findings if finding.severity in {"S1", "S2"}],
        }
    )


def _local_review(
    original_detection: DetectionResult,
    residual: DetectionResult,
    audit_result: AuditResult,
    round_number: int,
    max_rounds: int,
    signals: list[str],
) -> NaturalnessReviewResult:
    s1_count = sum(1 for finding in residual.findings if finding.severity == "S1")
    s2_count = sum(1 for finding in residual.findings if finding.severity == "S2")
    improvement = score_improvement(original_detection.severityWeightedScore, residual.severityWeightedScore)
    quality = strict_quality_grade(
        s1_count=s1_count,
        s2_count=s2_count,
        improvement=improvement,
        over_polish_signal_count=len(signals),
    )
    warnings: list[str] = []
    if s1_count:
        warnings.append(f"잔존 S1 AI 티 패턴이 {s1_count}건 감지됐습니다.")
    if s2_count > 3:
        warnings.append(f"잔존 S2 AI 티 패턴이 {s2_count}건으로 strict 합격선을 넘었습니다.")
    if signals:
        warnings.append("과윤문 신호가 감지됐습니다: " + ", ".join(signals))
    if audit_result.status in {"fail", "conditional_pass"}:
        decision = "rollback_and_rewrite" if round_number < max_rounds else "hold_and_report"
        reason = "보존 감사 실패로 재윤문이 필요합니다."
    elif len(signals) >= 2 and round_number < max_rounds:
        decision = "rollback_and_rewrite"
        reason = "과윤문 신호가 2개 이상 감지돼 문제 edit 롤백 후 재윤문이 필요합니다."
    elif len(signals) >= 2:
        decision = "hold_and_report"
        reason = "최대 라운드 후에도 과윤문 신호가 남았습니다."
    elif s1_count and round_number < max_rounds:
        decision = "rewrite_round_2"
        reason = "잔존 S1 패턴이 있어 추가 윤문이 필요합니다."
    elif s2_count > 3 and round_number < max_rounds:
        decision = "rewrite_round_2"
        reason = "잔존 S2 패턴이 strict 합격선을 넘어 추가 윤문이 필요합니다."
    elif s1_count:
        decision = "hold_and_report"
        reason = "최대 라운드 후에도 잔존 S1 패턴이 남았습니다."
    elif s2_count > 3:
        decision = "hold_and_report"
        reason = "최대 라운드 후에도 잔존 S2 패턴이 많습니다."
    elif s2_count:
        decision = "accept_with_note"
        reason = "S2 패턴이 일부 남았지만 strict 허용 범위입니다."
    else:
        decision = "accept"
        reason = "로컬 자연스러움 검토 기준을 통과했습니다."
    return NaturalnessReviewResult(
        decision=decision,
        warnings=warnings,
        residualFindings=residual.findings,
        scoreBefore=original_detection.severityWeightedScore,
        scoreAfter=residual.severityWeightedScore,
        scoreImprovement=improvement,
        s1Residual=s1_count,
        s2Residual=s2_count,
        overPolishSignals=signals,
        qualityLevel=quality,
        targetFindingIds=[finding.id for finding in residual.findings if finding.severity in {"S1", "S2"}],
        reason=reason,
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
        + llm_tokens
    )


def _sum_output_tokens(state: RewriteState) -> int:
    llm_tokens = 0 if state.get("strict_rewrite_result") else _token_value(state.get("llm_result"), "outputTokens")
    return (
        _token_value(state.get("detection"), "outputTokens")
        + _token_value(state.get("strict_rewrite_result"), "outputTokens")
        + _token_value(state.get("audit_result"), "outputTokens")
        + _token_value(state.get("review_result"), "outputTokens")
        + llm_tokens
    )


def _token_value(value: object | None, attr: str) -> int:
    return int(getattr(value, attr, 0) or 0)
