import asyncio
import re
import time
from collections import Counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from humanize_core.config import Settings
from humanize_core.debug_log import RewriteDebugLogger
from humanize_core.diff import build_display_safe_changes
from humanize_core.im_not_ai.audit import (
    local_detect,
    mark_high_risk_if_needed,
    split_sentences,
)
from humanize_core.im_not_ai.preservation import (
    checklist_for_preservation_label,
    preserved_units_for_text,
)
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    Finding,
    FlaggedEdit,
    RewriteResult,
    HumanizeContext,
    RulebookHint,
    StrictReviewResult,
)
from humanize_core.llm import LLMConfigurationError, LLMResponseError, RewriteLLM
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
    request_id: str
    job_id: str
    humanize_context: dict
    warnings: list[str]
    audit_result: AuditResult
    review_result: StrictReviewResult
    round: int
    rewrite_result: LLMRewriteResult
    llm_result: LLMRewriteResult
    response: RewriteResponse
    started_at: float
    chunk_count: int
    style_gate_input_tokens: int
    style_gate_output_tokens: int


class RewriteGraphRunner:
    def __init__(
        self,
        settings: Settings,
        llm: RewriteLLM,
        debug_logger: RewriteDebugLogger | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.debug_log = debug_logger or RewriteDebugLogger(settings)
        self.graph = self._build_graph()

    async def run(
        self,
        request: RewriteRequest,
        *,
        request_id: str | None = None,
        job_id: str | None = None,
    ) -> RewriteResponse:
        initial_state: RewriteState = {"request": request, "started_at": time.perf_counter()}
        if request_id:
            initial_state["request_id"] = request_id
        if job_id:
            initial_state["job_id"] = job_id
        state = await self.graph.ainvoke(initial_state)
        return state["response"]

    def _build_graph(self):
        builder = StateGraph(RewriteState)
        builder.add_node("prepare", self._prepare)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("style_gate", self._style_gate)
        builder.add_node("audit", self._audit)
        builder.add_node("review", self._review)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "rewrite")
        builder.add_edge("rewrite", "style_gate")
        builder.add_edge("style_gate", "audit")
        builder.add_conditional_edges(
            "audit",
            _route_after_audit,
            {"review": "review", "finalize": "finalize"},
        )
        builder.add_edge("review", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _stage_started(self, state: RewriteState, step: str) -> float:
        started_at = time.perf_counter()
        self.debug_log.event(
            "graph.stage.started",
            request_id=state.get("request_id"),
            job_id=state.get("job_id"),
            step=step,
            status="running",
            details=_request_debug_details(state["request"]),
        )
        return started_at

    def _stage_succeeded(
        self,
        state: RewriteState,
        step: str,
        started_at: float,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        self.debug_log.event(
            "graph.stage.succeeded",
            request_id=state.get("request_id"),
            job_id=state.get("job_id"),
            step=step,
            status="succeeded",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            details=details,
        )

    def _stage_failed(self, state: RewriteState, step: str, started_at: float, exc: Exception) -> None:
        self.debug_log.event(
            "graph.stage.failed",
            request_id=state.get("request_id"),
            job_id=state.get("job_id"),
            step=step,
            status="failed",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=_error_code_from_exception(exc),
            details={"error_type": type(exc).__name__},
        )

    async def _prepare(self, state: RewriteState) -> RewriteState:
        stage_started_at = self._stage_started(state, "prepare")
        try:
            request = state["request"]
            text_length = len(request.text)
            if text_length > self.settings.max_chars:
                raise InputLimitError(f"text length exceeds HUMANIZE_MAX_CHARS ({self.settings.max_chars})")

            warnings: list[str] = []

            detection = local_detect(request.text, protected_terms=request.protected_terms)
            context_model = HumanizeContext(
                detectedCount=detection.detectedCount,
                severityWeightedScore=detection.severityWeightedScore,
                categorySummary=detection.categorySummary,
                rulebookHints=_compact_rulebook_hints(detection.findings),
            )
            context = context_model.model_dump()

            result: RewriteState = {
                "humanize_context": context,
                "warnings": warnings,
                "round": 0,
            }
        except Exception as exc:
            self._stage_failed(state, "prepare", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "prepare",
            stage_started_at,
            details={
                **_request_debug_details(request),
                "context_field_count": len(context),
                "rulebook_hint_count": len(context_model.rulebookHints),
                "rulebook_category_summary": context_model.categorySummary,
                "rulebook_detected_count": context_model.detectedCount,
                "rulebook_severity_score": context_model.severityWeightedScore,
                "warnings_count": len(warnings),
            },
        )
        return result

    async def _rewrite(self, state: RewriteState) -> RewriteState:
        stage_started_at = self._stage_started(state, "rewrite")
        try:
            request = state["request"]
            chunk_count = 1
            if hasattr(self.llm, "rewrite_once"):
                chunks = self._rewrite_chunks_for(request)
                chunk_count = len(chunks)
                if chunk_count > 1:
                    rewrite_result = await self._rewrite_chunked(request, chunks)
                    implementation = "rewrite_once_chunked"
                else:
                    rewrite_result: RewriteResult = await self.llm.rewrite_once(  # type: ignore[attr-defined]
                        request,
                        state["humanize_context"],
                    )
                    implementation = "rewrite_once"
                result = _rewrite_result_state(request, rewrite_result, state.get("warnings", []))
                llm_result = result["llm_result"]
            else:
                llm_result = await self.llm.rewrite(request)
                result = {"llm_result": llm_result, "rewrite_result": llm_result, "round": 1}
                implementation = "rewrite"
            result["chunk_count"] = chunk_count
        except Exception as exc:
            self._stage_failed(state, "rewrite", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "rewrite",
            stage_started_at,
            details={
                "implementation": implementation,
                "chunk_count": chunk_count,
                "revised_text_length": len(llm_result.revisedText),
                "changes_count": len(llm_result.changes),
                "summary_count": len(llm_result.summary),
                "input_tokens": llm_result.inputTokens,
                "output_tokens": llm_result.outputTokens,
                "rounds": result.get("round", 1),
                "revised_text": llm_result.revisedText,
                "changes": [change.model_dump(mode="json") for change in llm_result.changes],
                "summary": llm_result.summary,
            },
        )
        return result

    def _rewrite_chunks_for(self, request: RewriteRequest) -> list[str]:
        if len(request.text) < self.settings.chunk_min_chars:
            return [request.text]
        return _split_text_chunks(request.text, self.settings.chunk_target_chars)

    async def _rewrite_chunked(self, request: RewriteRequest, chunks: list[str]) -> RewriteResult:
        async def rewrite_chunk(chunk: str) -> tuple[str, RewriteResult]:
            body = chunk.rstrip()
            separator = chunk[len(body):]
            if not body.strip():
                return chunk, RewriteResult(revisedText="", changes=[], summary=[])
            chunk_request = request.model_copy(update={"text": body})
            detection = local_detect(body, protected_terms=request.protected_terms)
            chunk_context = HumanizeContext(
                detectedCount=detection.detectedCount,
                severityWeightedScore=detection.severityWeightedScore,
                categorySummary=detection.categorySummary,
                rulebookHints=_compact_rulebook_hints(detection.findings),
            ).model_dump()
            chunk_result: RewriteResult = await self.llm.rewrite_once(  # type: ignore[attr-defined]
                chunk_request,
                chunk_context,
            )
            return separator, chunk_result

        chunk_results = await asyncio.gather(*(rewrite_chunk(chunk) for chunk in chunks))

        revised_parts: list[str] = []
        changes: list[Change] = []
        summary: list[str] = []
        warnings: list[str] = []
        input_tokens = 0
        output_tokens = 0
        for separator, chunk_result in chunk_results:
            revised_parts.append(chunk_result.revisedText.rstrip() + separator)
            changes.extend(chunk_result.changes)
            summary.extend(chunk_result.summary)
            warnings.extend(chunk_result.warnings)
            input_tokens += chunk_result.inputTokens
            output_tokens += chunk_result.outputTokens
        return RewriteResult(
            revisedText="".join(revised_parts),
            changes=changes,
            summary=_dedupe(summary),
            warnings=_dedupe(warnings),
            inputTokens=input_tokens,
            outputTokens=output_tokens,
        )

    async def _style_gate(self, state: RewriteState) -> RewriteState:
        stage_started_at = self._stage_started(state, "style_gate")
        try:
            request = state["request"]
            llm_result = state["llm_result"]
            chunked = state.get("chunk_count", 1) > 1
            max_rounds = self.settings.style_gate_max_rounds
            supported = hasattr(self.llm, "style_repair") and max_rounds > 0

            detection = local_detect(llm_result.revisedText, protected_terms=request.protected_terms)
            initial_score = detection.severityWeightedScore
            rounds_used = 0
            input_tokens = 0
            output_tokens = 0
            extra_warnings: list[str] = []
            implementation = "style_repair" if supported else "skipped"

            while supported and rounds_used < max_rounds:
                residual_s1 = _severity_count(detection.findings, "S1")
                residual_s2 = _severity_count(detection.findings, "S2")
                needs_style = residual_s1 > 0 or residual_s2 >= self.settings.style_gate_s2_threshold
                needs_transition = chunked and rounds_used == 0
                if not needs_style and not needs_transition:
                    break
                residual_hints = [
                    hint.model_dump() for hint in _compact_rulebook_hints(detection.findings)
                ]
                if not residual_hints and not needs_transition:
                    break

                repair: RewriteResult = await self.llm.style_repair(  # type: ignore[attr-defined]
                    request,
                    llm_result.revisedText,
                    residual_hints,
                    needs_transition,
                )
                rounds_used += 1
                input_tokens += repair.inputTokens
                output_tokens += repair.outputTokens

                repaired_text = repair.revisedText
                if not repaired_text.strip():
                    break
                if _style_repair_regressions(request, llm_result.revisedText, repaired_text):
                    break
                repaired_detection = local_detect(repaired_text, protected_terms=request.protected_terms)
                if needs_style and repaired_detection.severityWeightedScore > detection.severityWeightedScore:
                    break

                llm_result = llm_result.model_copy(
                    update={
                        "revisedText": repaired_text,
                        "changes": [*llm_result.changes, *repair.changes],
                        "summary": _dedupe([*llm_result.summary, *repair.summary]),
                    }
                )
                detection = repaired_detection

            final_s1 = _severity_count(detection.findings, "S1")
            if rounds_used and final_s1:
                extra_warnings.append(
                    f"스타일 게이트 수리 후에도 S1 잔존 신호가 {final_s1}건 감지됐습니다."
                )

            result: RewriteState = {
                "llm_result": llm_result,
                "warnings": _dedupe([*state.get("warnings", []), *extra_warnings]),
                "round": state.get("round", 1) + rounds_used,
                "style_gate_input_tokens": input_tokens,
                "style_gate_output_tokens": output_tokens,
            }
        except Exception as exc:
            self._stage_failed(state, "style_gate", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "style_gate",
            stage_started_at,
            details={
                "implementation": implementation,
                "chunked": chunked,
                "repair_rounds": rounds_used,
                "initial_severity_score": initial_score,
                "final_severity_score": detection.severityWeightedScore,
                "residual_s1_count": final_s1,
                "residual_s2_count": _severity_count(detection.findings, "S2"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "warnings_count": len(extra_warnings),
            },
        )
        return result

    async def _audit(self, state: RewriteState) -> RewriteState:
        stage_started_at = self._stage_started(state, "audit")
        try:
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
                audit_result.status == "fail" or _audit_requires_preservation_repair(audit_result),
            )
            result: RewriteState = {"audit_result": audit_result, "llm_result": llm_result}
        except Exception as exc:
            self._stage_failed(state, "audit", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "audit",
            stage_started_at,
            details={
                "result_status": audit_result.status,
                "warnings_count": len(audit_result.warnings),
                "flagged_edits_count": len(audit_result.flaggedEdits),
                "high_risk_change_count": len(audit_result.highRiskChangeIndexes),
                "rollback_required_count": audit_result.rollbackRequired,
                "edits_flagged_count": audit_result.editsFlagged,
                "edits_passed_count": audit_result.editsPassed,
                "review_required": _audit_requires_review(audit_result),
                "warnings": audit_result.warnings,
                "flagged_edits": [edit.model_dump(mode="json") for edit in audit_result.flaggedEdits],
            },
        )
        return result

    async def _audit_candidate(
        self,
        request: RewriteRequest,
        context: dict,
        revised_text: str,
        changes: list[Change],
    ) -> AuditResult:
        completion_warnings = _completion_warnings(request, revised_text)
        local_flagged = [
            *_local_flagged_edits_from_warnings(completion_warnings),
            *_local_preservation_flagged_edits(request, revised_text),
        ]
        local_warnings = _dedupe([*completion_warnings, *(edit.issue for edit in local_flagged)])

        if hasattr(self.llm, "audit"):
            model_audit: AuditResult = await self.llm.audit(  # type: ignore[attr-defined]
                request,
                context,
                revised_text,
                [change.model_dump() for change in changes],
            )
            model_completion_warnings = _completion_warnings(
                request,
                revised_text,
                model_audit.warnings,
            )
            local_warnings = _dedupe([*local_warnings, *model_completion_warnings])
            local_flagged = _merge_flagged_edits(
                [
                    *_local_flagged_edits_from_warnings(local_warnings),
                    *_local_preservation_flagged_edits(request, revised_text),
                ],
                [],
            )
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
        stage_started_at = self._stage_started(state, "review")
        try:
            request = state["request"]
            llm_result = state["llm_result"]

            if hasattr(self.llm, "review"):
                review_result: StrictReviewResult = await self.llm.review(  # type: ignore[attr-defined]
                    request,
                    state["humanize_context"],
                    llm_result.revisedText,
                    state["audit_result"],
                )
                implementation = "review"
                regressions = _review_output_regressions(
                    request,
                    llm_result.revisedText,
                    review_result.revisedText,
                )
                if regressions:
                    fallback = _local_repair_review(request, llm_result, state["audit_result"])
                    review_result = fallback.model_copy(
                        update={
                            "warnings": _dedupe(
                                [
                                    "모델 review 출력이 완성도·보존 재검증에 실패해 로컬 복원 결과로 대체했습니다.",
                                    *regressions,
                                    *fallback.warnings,
                                ]
                            ),
                            "inputTokens": review_result.inputTokens,
                            "outputTokens": review_result.outputTokens,
                        }
                    )
                    implementation = "local_repair_review_fallback"
            else:
                review_result = _local_repair_review(
                    request,
                    llm_result,
                    state["audit_result"],
                )
                implementation = "local_repair_review"
            final_llm_result = LLMRewriteResult(
                revisedText=review_result.revisedText,
                changes=review_result.changes,
                summary=review_result.summary,
                inputTokens=review_result.inputTokens,
                outputTokens=review_result.outputTokens,
            )
            result: RewriteState = {
                "review_result": review_result,
                "llm_result": final_llm_result,
            }
        except Exception as exc:
            self._stage_failed(state, "review", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "review",
            stage_started_at,
            details={
                "implementation": implementation,
                "result_status": review_result.finalAuditStatus,
                "quality_level": review_result.qualityLevel,
                "revised_text_length": len(review_result.revisedText),
                "changes_count": len(review_result.changes),
                "summary_count": len(review_result.summary),
                "warnings_count": len(review_result.warnings),
                "final_warnings_count": len(review_result.finalAuditWarnings),
                "blocking_issues_count": len(review_result.finalBlockingIssues),
                "input_tokens": review_result.inputTokens,
                "output_tokens": review_result.outputTokens,
                "revised_text": review_result.revisedText,
                "changes": [change.model_dump(mode="json") for change in review_result.changes],
                "summary": review_result.summary,
                "warnings": review_result.warnings,
                "final_audit_warnings": review_result.finalAuditWarnings,
                "final_blocking_issues": review_result.finalBlockingIssues,
            },
        )
        return result

    async def _finalize(self, state: RewriteState) -> RewriteState:
        stage_started_at = self._stage_started(state, "finalize")
        try:
            llm_result = state["llm_result"]
            warnings = list(state.get("warnings", []))
            audit_result = state.get("audit_result")
            review_result = state.get("review_result")
            if audit_result and not review_result:
                warnings.extend(audit_result.warnings)
            if review_result:
                warnings.extend(review_result.warnings)
                warnings.extend(review_result.finalAuditWarnings)
                warnings.extend(review_result.finalBlockingIssues)
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
            result: RewriteState = {"response": response}
        except Exception as exc:
            self._stage_failed(state, "finalize", stage_started_at, exc)
            raise

        self._stage_succeeded(
            state,
            "finalize",
            stage_started_at,
            details={
                "revised_text_length": len(response.revisedText),
                "changes_count": len(response.changes),
                "summary_count": len(response.summary),
                "warnings_count": len(response.warnings),
                "input_tokens": response.usage.inputTokens,
                "output_tokens": response.usage.outputTokens,
                "latency_ms": response.usage.latencyMs,
                "rounds": response.usage.rounds,
                "revised_text": response.revisedText,
                "changes": [change.model_dump(mode="json") for change in response.changes],
                "summary": response.summary,
                "warnings": response.warnings,
            },
        )
        return result


def _route_after_audit(state: RewriteState) -> str:
    audit_result = state.get("audit_result")
    if audit_result is not None and _audit_requires_review(audit_result):
        return "review"
    return "finalize"


def _request_debug_details(request: RewriteRequest) -> dict[str, object]:
    return {
        "rewrite_mode": request.rewrite_mode,
        "tone": request.tone,
        "max_rounds": request.max_rounds,
        "preserve_formatting": request.preserve_formatting,
        "text_length": len(request.text),
        "protected_terms_count": len(request.protected_terms),
        "user_intent_length": len(request.user_intent),
        "source_text": request.text,
        "user_intent": request.user_intent,
        "protected_terms": request.protected_terms,
    }


def _error_code_from_exception(exc: Exception) -> str:
    if isinstance(exc, InputLimitError):
        return "input_limit_exceeded"
    if isinstance(exc, LLMConfigurationError):
        return "model_not_configured"
    if isinstance(exc, LLMResponseError):
        return "invalid_model_response"
    return "internal_error"


def _audit_requires_review(audit_result: AuditResult) -> bool:
    if audit_result.status == "fail":
        return True
    if audit_result.status == "conditional_pass":
        return True
    if _audit_requires_preservation_repair(audit_result):
        return True
    return any(edit.action != "warning" for edit in audit_result.flaggedEdits)


def _rewrite_result_state(
    request: RewriteRequest,
    rewrite_result: RewriteResult,
    warnings: list[str],
) -> RewriteState:
    summary = list(rewrite_result.summary)
    llm_result = LLMRewriteResult(
        revisedText=rewrite_result.revisedText,
        changes=rewrite_result.changes,
        summary=summary,
        inputTokens=rewrite_result.inputTokens,
        outputTokens=rewrite_result.outputTokens,
    )
    return {
        "llm_result": llm_result,
        "rewrite_result": llm_result,
        "warnings": _dedupe([*warnings, *rewrite_result.warnings]),
        "round": 1,
    }


_MAX_RULEBOOK_HINTS = 24
_MAX_HINT_MATCHES = 3
_MATCH_SNIPPET_MAX_CHARS = 40
# Findings for these rules may legitimately overlap protected values (numbers,
# quotes), so their spans never travel as match samples.
_MATCH_EXEMPT_RULE_IDS = frozenset({"B-1", "C-9"})


def _compact_rulebook_hints(findings: list[Finding]) -> list[RulebookHint]:
    severity_order = {"S1": 0, "S2": 1, "S3": 2}
    scope_order = {"span": 0, "document": 1}
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        if finding.severity not in {"S1", "S2"}:
            continue
        grouped.setdefault(finding.category, []).append(finding)

    for rule_findings in grouped.values():
        rule_findings.sort(
            key=lambda finding: (
                severity_order.get(finding.severity, 9),
                scope_order.get(finding.scope, 9),
                finding.id,
            )
        )

    def rule_sort_key(item: tuple[str, list[Finding]]) -> tuple[int, str]:
        rule_id, rule_findings = item
        return (severity_order.get(rule_findings[0].severity, 9), rule_id)

    hints: list[RulebookHint] = []
    for rule_id, rule_findings in sorted(grouped.items(), key=rule_sort_key):
        first = rule_findings[0]
        matches: list[str] = []
        if rule_id not in _MATCH_EXEMPT_RULE_IDS:
            for finding in rule_findings:
                span = (finding.textSpan or "").strip()[:_MATCH_SNIPPET_MAX_CHARS]
                if span and span not in matches:
                    matches.append(span)
                if len(matches) >= _MAX_HINT_MATCHES:
                    break
        hints.append(
            RulebookHint(
                id=first.id,
                category=rule_id,
                categoryLabel=first.categoryLabel,
                severity=first.severity,
                scope=first.scope,
                suggestedFix=first.suggestedFix,
                occurrences=len(rule_findings),
                matches=matches,
            )
        )
        if len(hints) >= _MAX_RULEBOOK_HINTS:
            break
    return hints


def _split_text_chunks(text: str, target_chars: int) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    segments: list[str] = []
    for index in range(0, len(parts), 2):
        paragraph = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if paragraph or separator:
            segments.append(paragraph + separator)

    chunks: list[str] = []
    current = ""
    for segment in segments:
        if current and len(current) + len(segment) > target_chars:
            chunks.append(current)
            current = segment
        else:
            current += segment
    if current:
        chunks.append(current)
    return chunks or [text]


def _severity_count(findings: list[Finding], severity: str) -> int:
    return sum(1 for finding in findings if finding.severity == severity)


def _style_repair_regressions(
    request: RewriteRequest,
    draft_text: str,
    repaired_text: str,
) -> list[str]:
    draft_completion = set(_completion_warnings(request, draft_text))
    reasons = [
        warning
        for warning in _completion_warnings(request, repaired_text)
        if warning not in draft_completion
    ]
    draft_flagged = _local_preservation_flagged_edits(request, draft_text)
    repaired_flagged = _local_preservation_flagged_edits(request, repaired_text)
    if len(repaired_flagged) > len(draft_flagged):
        reasons.append("스타일 수리 출력에서 보존 대상 훼손이 초안보다 늘었습니다.")
    return _dedupe(reasons)


def _audit_status(local_warnings: list[str], model_status: str, flagged_edits: list[FlaggedEdit]) -> str:
    if any(_is_completion_warning(warning) for warning in local_warnings):
        return "fail"
    if any(_flagged_edit_blocks(edit) for edit in flagged_edits):
        return "conditional_pass" if model_status == "full_pass" else model_status
    if flagged_edits and model_status == "full_pass":
        return "conditional_pass"
    if local_warnings and model_status == "full_pass":
        return "conditional_pass"
    return model_status


def _local_flagged_edits_from_warnings(local_warnings: list[str]) -> list[FlaggedEdit]:
    flagged: list[FlaggedEdit] = []
    for warning in local_warnings:
        if _is_completion_warning(warning):
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


def _local_preservation_flagged_edits(request: RewriteRequest, revised_text: str) -> list[FlaggedEdit]:
    flagged: list[FlaggedEdit] = []
    original_units = dict(preserved_units_for_text(request.text, request.protected_terms))
    revised_units = dict(preserved_units_for_text(revised_text, request.protected_terms))
    for label, values in original_units.items():
        original_counts = Counter(values)
        for value, expected_count in original_counts.items():
            observed_count = revised_text.count(value)
            if observed_count >= expected_count:
                continue
            flagged.append(
                FlaggedEdit(
                    before=value,
                    after="",
                    issue=f"{label} 보존 대상이 원문보다 적게 남았습니다: {value}",
                    checklistFailed=checklist_for_preservation_label(label),
                    action="preserve_exact",
                    correctionDirection=f"{label} '{value}'를 원문 그대로 복원합니다.",
                    severity="high",
                )
            )
    for label, values in revised_units.items():
        original_counts = Counter(original_units.get(label, []))
        revised_counts = Counter(values)
        for value, observed_count in revised_counts.items():
            expected_count = original_counts.get(value, 0)
            if observed_count <= expected_count:
                continue
            flagged.append(
                FlaggedEdit(
                    before="",
                    after=value,
                    issue=f"{label} 값이 원문보다 많이 추가됐습니다: {value}",
                    checklistFailed=checklist_for_preservation_label(label),
                    action="rewrite_required",
                    correctionDirection=f"원문에 없는 {label} '{value}'를 제거하거나 원문 표현으로 복원합니다.",
                    severity="high",
                )
            )
    return _merge_flagged_edits(flagged, [])


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


def _local_repair_review(
    request: RewriteRequest,
    llm_result: LLMRewriteResult,
    audit_result: AuditResult,
) -> StrictReviewResult:
    summary = list(llm_result.summary)
    repair_edits = [edit for edit in audit_result.flaggedEdits if edit.action != "warning"]
    corrections = [edit.correctionDirection or edit.issue for edit in repair_edits]
    if corrections:
        repaired_text, applied, unresolved = _apply_local_audit_repairs(
            request,
            llm_result.revisedText,
            repair_edits,
        )
        summary.append("Audit repair: 감사 지적 항목만 원문 기준으로 부분 복원했습니다.")
        warnings = [f"감사 지적 {len(applied)}건을 로컬에서 부분 복원했습니다."] if applied else []
        if unresolved:
            warnings.append(
                "감사 지적 중 로컬 자동 복원이 어려운 항목이 있습니다: "
                + "; ".join(unresolved[:3])
            )
        return StrictReviewResult(
            revisedText=repaired_text,
            changes=build_display_safe_changes(request.text, repaired_text, llm_result.changes),
            summary=summary,
            warnings=warnings,
            auditCorrectionsApplied=applied,
            finalAuditStatus="full_pass" if not unresolved else audit_result.status,
            finalBlockingIssues=unresolved,
            qualityLevel="B",
            inputTokens=0,
            outputTokens=0,
        )

    summary.append("Audit repair: 감사 단계에서 복원할 항목이 없어 초안을 유지했습니다.")
    return StrictReviewResult(
        revisedText=llm_result.revisedText,
        changes=llm_result.changes,
        summary=summary,
        warnings=[],
        finalAuditStatus=audit_result.status,
        finalAuditWarnings=audit_result.warnings,
        qualityLevel="B",
        inputTokens=0,
        outputTokens=0,
    )


def _apply_local_audit_repairs(
    request: RewriteRequest,
    revised_text: str,
    flagged_edits: list[FlaggedEdit],
) -> tuple[str, list[str], list[str]]:
    repaired = revised_text
    applied: list[str] = []
    unresolved: list[str] = []
    for edit in flagged_edits:
        label = edit.correctionDirection or edit.issue
        if _local_repair_satisfies_edit(request, repaired, edit):
            applied.append(label)
            continue

        next_text = repaired
        if edit.before and edit.after and edit.after in next_text:
            next_text = next_text.replace(edit.after, edit.before, 1)
        elif edit.after and not edit.before and edit.after in next_text:
            next_text = _clean_repaired_text(next_text.replace(edit.after, "", 1))
        elif edit.before and repaired.count(edit.before) < request.text.count(edit.before):
            next_text = _restore_original_sentence_for_value(request.text, repaired, edit.before)

        if next_text != repaired:
            repaired = next_text
            applied.append(label)
        else:
            unresolved.append(label)
    return repaired, _dedupe(applied), _dedupe(unresolved)


def _local_repair_satisfies_edit(
    request: RewriteRequest,
    repaired_text: str,
    edit: FlaggedEdit,
) -> bool:
    if edit.before and not edit.after:
        return repaired_text.count(edit.before) >= request.text.count(edit.before)
    if edit.before and edit.after:
        return edit.before in repaired_text and edit.after not in repaired_text
    if edit.after and not edit.before:
        return edit.after not in repaired_text
    return False


def _restore_original_sentence_for_value(original: str, revised: str, value: str) -> str:
    if value in revised:
        return revised

    original_sentences = split_sentences(original)
    revised_sentences = split_sentences(revised)
    for index, original_sentence in enumerate(original_sentences):
        if value not in original_sentence:
            continue
        if index < len(revised_sentences):
            return revised.replace(revised_sentences[index], original_sentence, 1)
        separator = "" if revised.endswith(("\n", " ")) else " "
        return f"{revised}{separator}{original_sentence}".strip()
    return revised


def _clean_repaired_text(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def _review_output_regressions(
    request: RewriteRequest,
    draft_text: str,
    review_text: str,
) -> list[str]:
    # Flag only damage the review step introduced relative to the draft; a
    # draft that was already incomplete ships with warnings instead of being
    # stitched back together locally.
    draft_completion = set(_completion_warnings(request, draft_text))
    reasons = [
        warning
        for warning in _completion_warnings(request, review_text)
        if warning not in draft_completion
    ]
    draft_flagged = _local_preservation_flagged_edits(request, draft_text)
    review_flagged = _local_preservation_flagged_edits(request, review_text)
    if len(review_flagged) > len(draft_flagged):
        reasons.append("Review 출력에서 보존 대상 훼손이 초안보다 늘었습니다.")
    return _dedupe(reasons)


def _audit_requires_preservation_repair(audit_result: AuditResult) -> bool:
    return audit_result.rollbackRequired > 0 or any(
        _flagged_edit_blocks(edit) for edit in audit_result.flaggedEdits
    )


def _flagged_edit_blocks(edit: FlaggedEdit) -> bool:
    return edit.action in {"restore_original", "preserve_exact", "rollback_required"} or (
        edit.action in {"rewrite_required", "rewrite_with_hedge_preserved"} and edit.severity == "high"
    )


def _completion_warnings(
    request: RewriteRequest,
    revised_text: str,
    model_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    original = request.text.strip()
    revised = revised_text.strip()
    if not revised:
        warnings.append("Rewrite 결과가 비어 있어 완성도 검증을 통과하지 못했습니다.")
        return warnings

    if _model_reports_incomplete(model_warnings or []):
        warnings.append("Rewrite 결과가 중간에 잘렸거나 불완전하다는 감사 신호가 감지됐습니다.")

    if len(original) >= 200:
        length_ratio = len(revised) / max(len(original), 1)
        if length_ratio < 0.60:
            warnings.append("Rewrite 결과가 원문 대비 지나치게 짧아 누락 또는 출력 잘림 가능성이 큽니다.")

        original_sentences = split_sentences(original)
        revised_sentences = split_sentences(revised)
        if len(original_sentences) >= 4 and len(revised_sentences) / max(len(original_sentences), 1) < 0.50:
            warnings.append("Rewrite 결과의 문장 커버리지가 원문 대비 낮아 전체 내용을 반영하지 못했을 수 있습니다.")

        if request.preserve_formatting:
            original_paragraphs = _paragraphs(original)
            revised_paragraphs = _paragraphs(revised)
            if len(original_paragraphs) >= 3 and len(revised_paragraphs) / max(len(original_paragraphs), 1) < 0.50:
                warnings.append("Rewrite 결과의 문단 커버리지가 원문 대비 낮아 일부 문단이 누락됐을 수 있습니다.")

    if _looks_cut_off_mid_sentence(original, revised):
        warnings.append("Rewrite 결과가 문장 중간에서 끝난 것으로 보여 완성도 검증을 통과하지 못했습니다.")

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


def _is_completion_warning(warning: str) -> bool:
    return warning.startswith(("Rewrite 결과가", "Strict 결과가")) or _INCOMPLETE_WARNING_RE.search(warning) is not None


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _sum_input_tokens(state: RewriteState) -> int:
    rewrite_result = state.get("rewrite_result") or state.get("llm_result")
    return (
        _token_value(rewrite_result, "inputTokens")
        + int(state.get("style_gate_input_tokens") or 0)
        + _token_value(state.get("audit_result"), "inputTokens")
        + _token_value(state.get("review_result"), "inputTokens")
    )


def _sum_output_tokens(state: RewriteState) -> int:
    rewrite_result = state.get("rewrite_result") or state.get("llm_result")
    return (
        _token_value(rewrite_result, "outputTokens")
        + int(state.get("style_gate_output_tokens") or 0)
        + _token_value(state.get("audit_result"), "outputTokens")
        + _token_value(state.get("review_result"), "outputTokens")
    )


def _token_value(value: object | None, attr: str) -> int:
    return int(getattr(value, attr, 0) or 0)
