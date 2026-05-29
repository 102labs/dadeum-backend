import hashlib
import hmac
import json
import logging
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from humanize_core.api import create_app
from humanize_core.config import Settings
from humanize_core.graph import RewriteGraphRunner
from humanize_core.im_not_ai import prompts, resources
from humanize_core.im_not_ai.audit import (
    SUPPORTED_QUICK_RULE_IDS,
    finding_density,
    finding_score,
    local_detect,
)
from humanize_core.im_not_ai.metrics_v2 import (
    compute_all_v2,
    deul_overuse_rate,
    double_passive_count,
    pronoun_density,
)
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    DetectionResult,
    FastRewriteResult,
    Finding,
    StrictReviewResult,
    StrictRewriteResult,
)
from humanize_core.llm import (
    OpenAIRewriteLLM,
    OpenRouterRewriteLLM,
    StubRewriteLLM,
    _openai_rewrite_text_format,
    _openrouter_response_format,
)
from humanize_core.schemas import Change
from humanize_core.schemas import RewriteRequest as RewriteRequestForTest


def _settings() -> Settings:
    return Settings(
        core_api_key="test-core-key",
        signing_secret="test-signing-secret",
        model_provider="stub",
        model_name="stub",
        max_chars=10_000,
    )


def _payload(**overrides):
    body = {
        "text": "안녕하세요.  2026년 5월 보고서 문장을 더 명확하게 정리해주세요.",
        "user_intent": "",
        "rewrite_mode": "fast",
        "tone": "keep",
        "protected_terms": ["2026년"],
        "max_rounds": 1,
        "preserve_formatting": True,
    }
    body.update(overrides)
    return body


def _signed_headers(raw_body: bytes, *, timestamp: str | None = None, request_id: str = "req_test"):
    timestamp = timestamp or str(int(time.time()))
    body_hash = hashlib.sha256(raw_body).hexdigest()
    signature_payload = f"{timestamp}.{request_id}.{body_hash}"
    signature = hmac.new(
        b"test-signing-secret",
        signature_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "X-Core-Api-Key": "test-core-key",
        "X-Request-Id": request_id,
        "X-Timestamp": timestamp,
        "X-Body-SHA256": body_hash,
        "X-Signature": signature,
    }


def _strict_review_result(request, revised_text, *, warnings=None, blocking=None, status="full_pass"):
    return StrictReviewResult(
        revisedText=revised_text,
        changes=[
            Change(
                original=request.text,
                revised=revised_text,
                reason="strict review 최종 후보입니다.",
                type="clarity",
                riskLevel="low",
            )
        ],
        summary=["strict review가 최종 후보를 구성했습니다."],
        warnings=warnings or [],
        finalAuditStatus=status,
        finalBlockingIssues=blocking or [],
    )


def _client() -> TestClient:
    return TestClient(create_app(_settings()))


def test_health_returns_ok():
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_api_key_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)
    headers.pop("X-Core-Api-Key")

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_bad_signature_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)
    headers["X-Signature"] = "bad"

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_expired_timestamp_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body, timestamp=str(int(time.time()) - 600))

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_body_hash_mismatch_returns_401():
    client = _client()
    signed_body = json.dumps(_payload(text="signed"), ensure_ascii=False).encode("utf-8")
    sent_body = json.dumps(_payload(text="sent"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(signed_body)

    response = client.post("/v1/rewrite", content=sent_body, headers=headers)

    assert response.status_code == 401


def test_invalid_enum_returns_422_after_auth_passes():
    client = _client()
    raw_body = json.dumps(_payload(tone="executive"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_old_contract_fields_return_422_after_auth_passes():
    client = _client()
    body = _payload()
    body["intensity"] = "standard"
    body["concision"] = "tighten"
    body["intent"] = "business_polish"
    body["quality_mode"] = "balanced"
    body["focus_categories"] = []
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_rewrite_request_defaults_match_new_contract():
    request = RewriteRequestForTest.model_validate({"text": "문장을 정리합니다."})

    assert request.user_intent == ""
    assert request.rewrite_mode == "fast"
    assert request.tone == "keep"
    assert request.protected_terms == []
    assert request.max_rounds == 1
    assert request.preserve_formatting is True


def test_rewrite_request_normalizes_intent_and_protected_terms():
    request = RewriteRequestForTest.model_validate(
        {
            "text": "API v1은 2026년에 유지됩니다.",
            "user_intent": "  더 명확하게  ",
            "protected_terms": [" API v1 ", "", " 2026년 "],
        }
    )

    assert request.user_intent == "더 명확하게"
    assert request.protected_terms == ["API v1", "2026년"]


def test_rewrite_success_returns_structured_response():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["revisedText"]
    assert data["changes"]
    assert data["summary"]
    assert data["usage"]["rounds"] == 1
    assert isinstance(data["warnings"], list)


def test_stub_rewrite_reflects_user_selected_controls():
    client = _client()
    raw_body = json.dumps(
        _payload(
            text="보고 문장",
            user_intent="더 단정하게 정리",
            tone="formal",
            preserve_formatting=True,
        ),
        ensure_ascii=False,
    ).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["revisedText"] == "보고 문장."
    assert "격식 있는 톤" in data["summary"][0]
    assert "사용자 요청 방향" in data["summary"][0]
    assert "형식을 보존" in data["summary"][0]


async def test_long_fast_request_stays_on_fast_path():
    text = "보고 문장입니다. " * 801
    request = RewriteRequestForTest.model_validate(_payload(text=text, rewrite_mode="fast", max_rounds=3))

    class CapturingFastLLM:
        def __init__(self) -> None:
            self.fast_calls = 0

        async def rewrite_fast(self, request, context):
            self.fast_calls += 1
            return FastRewriteResult(
                revisedText=request.text,
                changes=[
                    Change(
                        original="보고 문장입니다.",
                        revised="보고 문장입니다.",
                        reason="긴 fast 요청도 사용자 선택 모드를 유지합니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["fast 모드를 유지했습니다."],
            )

        async def detect(self, request, context):
            raise AssertionError("long fast request should not switch to strict detection")

    llm = CapturingFastLLM()
    response = await RewriteGraphRunner(_settings(), llm).run(request)

    assert len(text) > 8_000
    assert llm.fast_calls == 1
    assert response.usage.rounds == 1
    assert not any("strict 모드로 자동 전환" in warning for warning in response.warnings)


def test_valid_strict_request_uses_structured_response():
    client = _client()
    raw_body = json.dumps(_payload(rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["revisedText"]
    assert data["changes"]
    assert data["usage"]["rounds"] >= 1


def test_strict_request_ignores_max_rounds_and_runs_single_routine():
    client = _client()
    text = "결론적으로 성과를 냈습니다. 따라서 개선됩니다. 이를 통해 정리합니다. 그러므로 유지합니다."
    raw_body = json.dumps(_payload(text=text, rewrite_mode="strict", max_rounds=3), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["usage"]["rounds"] == 1
    assert not any("최대 라운드" in warning for warning in data["warnings"])


def test_strict_request_defaults_to_single_round_when_max_rounds_omitted():
    client = _client()
    text = "결론적으로 성과를 냈습니다. 따라서 개선됩니다. 이를 통해 정리합니다. 그러므로 유지합니다."
    body = _payload(text=text, rewrite_mode="strict")
    body.pop("max_rounds")
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["usage"]["rounds"] == 1
    assert not any("최대 라운드" in warning for warning in data["warnings"])


def test_prompts_pass_user_selected_rewrite_controls():
    request = RewriteRequestForTest.model_validate(
        _payload(
            user_intent="문장을 더 부드럽게 다듬어 주세요.",
            rewrite_mode="strict",
            tone="friendly",
            max_rounds=3,
            preserve_formatting=True,
        )
    )

    payload = json.loads(
        prompts.fast_user_prompt(
            request,
            {},
        )
    )

    assert payload["settings"] == {
        "user_intent": "문장을 더 부드럽게 다듬어 주세요.",
        "rewrite_mode": "strict",
        "tone": "friendly",
        "preserve_formatting": True,
    }
    assert "우선 반영" in payload["rewrite_guidance"]["user_intent"]
    assert "친근" in payload["rewrite_guidance"]["tone"]
    assert "protected_terms" not in payload["rewrite_guidance"]
    assert "max_rounds" not in payload["rewrite_guidance"]
    assert "줄바꿈" in payload["rewrite_guidance"]["formatting"]


def _strict_prompt_detection() -> DetectionResult:
    return DetectionResult(
        sentenceCount=4,
        detectedCount=2,
        aiTellDensity=0.12,
        severityWeightedScore=7.0,
        categorySummary={"A": 1, "F": 1},
        findings=[
            Finding(
                id="f-a2",
                category="A-2",
                categoryLabel="추상 연결어",
                severity="S1",
                scope="span",
                textSpan="데이터를 통해 성장합니다",
                start=10,
                end=24,
                reason="'통해' 중심의 번역투 연결입니다.",
                suggestedFix="행위 주체와 결과가 보이도록 동사형으로 정리합니다.",
            ),
            Finding(
                id="f-f4",
                category="F-4",
                categoryLabel="명사형 추상어",
                severity="S2",
                scope="document",
                textSpan="전략적 실행성과 구조화",
                start=None,
                end=None,
                reason="추상 명사어가 겹쳐 읽기 어렵습니다.",
                suggestedFix="실제 행동과 대상이 드러나게 풀어 씁니다.",
            ),
        ],
    )


def test_strict_rewrite_prompt_targets_advanced_quality_not_finding_only():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="이 기능은 데이터를 통해 성장을 지원합니다. 전략적 실행성과 구조화가 필요합니다.",
            rewrite_mode="strict",
            max_rounds=2,
        )
    )
    detection = _strict_prompt_detection()

    system_prompt = prompts.strict_rewrite_system_prompt()
    payload = json.loads(
        prompts.strict_rewrite_user_prompt(
            request,
            {},
            detection,
        )
    )
    rendered_payload = json.dumps(payload, ensure_ascii=False)

    assert "Rewrite only finding-backed" not in system_prompt
    assert "only finding-backed" not in system_prompt
    assert "priority hints" in system_prompt
    assert "quick_rules" not in payload
    assert "rewriting_playbook" not in payload
    assert "findings" not in payload
    assert "advanced_rewrite_objective" in payload
    assert "rewrite_strategy" in payload
    assert "completion_contract" in payload
    assert "structured_output_contract" in payload
    assert "single_pass_initial_draft" in payload["rewrite_strategy"]
    assert payload["completion_contract"]["originalCharCount"] == len(request.text)
    assert "complete rewritten passage" in payload["completion_contract"]["scope"]
    assert any("revisedText is the single canonical final answer" in item for item in payload["structured_output_contract"])
    assert any("changes[].original and changes[].revised are local diff snippets only" in item for item in payload["structured_output_contract"])
    assert "rewrite_plan" in payload
    assert "priority_findings" in payload
    assert "do_not_edit_spans" in payload
    assert payload["priority_findings"][0]["id"] == "f-a2"
    assert payload["rewrite_plan"][0].startswith("A-2")
    assert any(recipe["category"] == "A" for recipe in payload["targeted_rewrite_recipes"])
    assert any("직접 인용" in rule for rule in payload["do_not_edit_spans"])
    assert "수정 범위를 finding span으로 제한하지 말고" in rendered_payload
    assert "문장 흐름" in rendered_payload
    assert "리듬" in rendered_payload
    assert "명확성" in rendered_payload
    assert "변경률은 품질 목표가 아니라" in payload["rewrite_guidance"]["rewrite_mode"]


def test_review_prompt_applies_audit_corrections_and_reaudits():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="이 기능은 데이터를 통해 성장을 지원합니다.",
            rewrite_mode="strict",
        )
    )
    detection = _strict_prompt_detection()
    audit_result = AuditResult(
        status="conditional_pass",
        flaggedEdits=[
            {
                "before": "데이터를 통해",
                "after": "데이터로",
                "issue": "번역투 연결입니다.",
                "checklistFailed": [13],
                "action": "rewrite_required",
                "correctionDirection": "'통해'를 자연스러운 조사로 줄입니다.",
                "severity": "medium",
            }
        ],
        reason="수정 지시가 있습니다.",
    )

    payload = json.loads(
        prompts.review_user_prompt(
            request,
            {},
            detection,
            "이 기능은 데이터를 통해 성장을 지원합니다.",
            audit_result,
            detection,
        )
    )

    rendered = json.dumps(payload, ensure_ascii=False)
    assert "strict_audit" in payload
    assert "fixed_review_routine" in payload
    assert "다시 점검" in rendered
    assert "finalAuditStatus" in rendered


def test_strict_detect_prompt_uses_compact_stric_rules_only_in_detect():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="결론적으로 성과를 통해 결과를 냈습니다.",
            rewrite_mode="strict",
        )
    )
    context = {}
    detection = _strict_prompt_detection()

    detect_payload = json.loads(prompts.detect_user_prompt(request, context))
    rewrite_payload = json.loads(prompts.strict_rewrite_user_prompt(request, context, detection))

    assert detect_payload["rulebook"] == "stric-rules.md"
    assert "strict_rules" in detect_payload
    assert "A-1" in detect_payload["strict_rules"]
    assert "Operational Notes" in detect_payload["strict_rules"]
    assert "Do not flag" in detect_payload["strict_rules"]
    assert "A-2 example" in detect_payload["strict_rules"]
    assert "Detector Contract" in detect_payload["strict_rules"]
    assert "strict_rules" not in rewrite_payload
    assert "stric-rules.md" not in json.dumps(rewrite_payload, ensure_ascii=False)
    assert not hasattr(resources, "ai_tell_taxonomy")
    assert hasattr(resources, "strict_rules")


def test_strict_rewrite_and_review_prompts_stay_compact():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text=(
                "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 "
                "묶어서 폴더를 만들고 자동으로 정리합니다. 두 번째는 소스 검색 기능입니다. 더 이상 "
                "사용자가 직접 피드를 스크롤하지 않아도 필요한 자료를 찾아 작은 묶음으로 반환합니다."
            ),
            rewrite_mode="strict",
            max_rounds=2,
        )
    )
    context = {}
    detection = _strict_prompt_detection()

    fast_prompt = prompts.fast_user_prompt(request, context)
    strict_prompt = prompts.strict_rewrite_user_prompt(request, context, detection)
    audit_result = AuditResult(status="full_pass", reason="통과")
    review_prompt = prompts.review_user_prompt(request, context, detection, request.text, audit_result, detection)

    assert len(strict_prompt) < len(fast_prompt)
    assert len(review_prompt) < len(fast_prompt)


def test_strict_audit_prompt_does_not_embed_scholarship_reference():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="AI 에이전트가 저장된 자료를 정리하고 필요한 자료를 찾아줍니다.",
            rewrite_mode="strict",
            max_rounds=2,
        )
    )
    context = {}

    payload = json.loads(prompts.audit_user_prompt(request, context, request.text, []))
    rendered_payload = json.dumps(payload, ensure_ascii=False)

    assert "scholarship_constraints" not in payload
    assert "번역학계" not in rendered_payload
    assert "checklist_13" in payload


def test_removed_reference_resources_have_no_runtime_loaders():
    assert not hasattr(resources, "scholarship")
    assert not hasattr(resources, "rewriting_playbook")
    assert not hasattr(resources, "ai_tell_taxonomy")
    assert hasattr(resources, "strict_rules")


async def test_stub_rewrite_uses_preserve_formatting_switch():
    preserving = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=True)
    )
    normalizing = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=False)
    )
    llm = StubRewriteLLM()

    preserved = await llm.rewrite(preserving)
    normalized = await llm.rewrite(normalizing)

    assert preserved.revisedText == "첫 문장.  둘째 문장."
    assert normalized.revisedText == "첫 문장. 둘째 문장."


async def test_protected_terms_are_accepted_but_not_used_by_prepare_or_audit():
    class DropsProtectedTermLLM:
        async def rewrite(self, request):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, context):
            return FastRewriteResult(
                revisedText="정책은 유지됩니다.",
                changes=[
                    Change(
                        original="API v1 정책은 유지됩니다.",
                        revised="정책은 유지됩니다.",
                        reason="테스트용 누락입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["표현을 정리했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="API v1 정책은 유지됩니다.", protected_terms=["API v1"])
    )

    response = await RewriteGraphRunner(_settings(), DropsProtectedTermLLM()).run(request)

    assert not any("API v1" in warning and "누락" in warning for warning in response.warnings)
    assert response.changes[0].riskLevel == "low"


async def test_prepare_context_is_empty():
    captured = {}

    class CapturingLLM:
        async def rewrite(self, request):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, context):
            captured["context"] = context
            return FastRewriteResult(
                revisedText=request.text,
                changes=[Change(original="", revised="", reason="유지했습니다.", type="clarity", riskLevel="low")],
                summary=["유지했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="분기별 보고 결과와 분석 내용입니다.")
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    assert captured["context"] == {}


def test_local_quick_rules_cover_fast_source_rule_ids():
    expected = {
        "A-1",
        "A-2",
        "A-3",
        "A-4",
        "A-5",
        "A-6",
        "A-7",
        "A-8",
        "A-9",
        "A-10",
        "A-11",
        "A-15",
        "A-16",
        "A-18",
        "A-19",
        "B-1",
        "B-2",
        "C-5",
        "C-7",
        "C-8",
        "C-9",
        "C-10",
        "C-11",
        "D-1",
        "D-2",
        "D-3",
        "D-4",
        "D-5",
        "D-6",
        "D-7",
        "E-1",
        "E-2",
        "E-7",
        "F-4",
        "F-5",
        "G-1",
        "G-2",
        "G-3",
        "H-1",
        "H-3",
        "H-4",
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "J-1",
        "J-2",
        "J-3",
    }

    assert SUPPORTED_QUICK_RULE_IDS == expected


def test_local_detect_covers_reinforced_quick_rule_cases():
    cases = [
        ("A-16", "그는 말했다. 그는 다시 말했다. 그것은 반복됐다."),
        ("B-1", "소버린 AI(Sovereign AI)는 중요한 전략이다."),
        ("C-9", "(1) 계획을 세운다. (2) 실행한다."),
        ("E-2", "첫째다. 둘째다. 셋째다. 넷째다."),
        ("E-7", "우리는 실행한다. 그런데 이건 좋아요. 다음 단계입니다."),
        ("F-4", "전략적 실행성과 구조화가 필요하다."),
        ("G-3", "균형 있게 보고 신중하게 판단하며 양쪽 모두와 두 가지 모두의 장점도 있지만 균형이 필요하다."),
        ("J-3", "- 첫 번째 항목\n- 두 번째 항목"),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        assert rule_id in {finding.category for finding in detection.findings}


def test_local_detection_uses_rule_score_and_density():
    detection = local_detect("성과를 통해 결과를 냈다.", focus_categories=["A-2"])

    assert detection.severityWeightedScore == 5.0
    assert detection.aiTellDensity == finding_density("성과를 통해 결과를 냈다.", detection.findings)
    assert finding_score(detection.findings) == 5.0


def test_local_detector_severity_matches_strict_rulebook_for_non_decisive_style_rules():
    cases = [
        ("C-10", "전략: 실행으로 전환"),
        ("H-1", "또한 우리는 실행합니다."),
        ("J-2", '"하나" "둘" "셋" "넷" "다섯" "여섯"'),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        severities = {finding.severity for finding in detection.findings if finding.category == rule_id}
        assert severities == {"S2"}


def test_local_detection_excludes_do_not_spans():
    text = '"데이터를 통해 성장한다"라고 말했다. API는 유지한다.'
    detection = local_detect(text)

    assert "A-2" not in {finding.category for finding in detection.findings}


def test_a16_pronoun_literal_translation_golden_case():
    literal = "메리는 그녀가 그녀를 그리워해서 그녀의 어머니에게 전화했다."
    natural = "메리는 어머니가 그리워서 전화를 걸었다."

    literal_detection = local_detect(literal, focus_categories=["A-16"])
    natural_detection = local_detect(natural, focus_categories=["A-16"])

    assert "A-16" in {finding.category for finding in literal_detection.findings}
    assert "A-16" not in {finding.category for finding in natural_detection.findings}
    assert pronoun_density(literal) > pronoun_density(natural)


def test_a17_deul_overuse_stays_metric_only():
    text = "이러한 데이터들과 정보들과 결과들이 중요한 아이디어들을 보여준다."

    detection = local_detect(text)

    assert "A-17" not in SUPPORTED_QUICK_RULE_IDS
    assert "A-17" not in {finding.category for finding in detection.findings}
    assert deul_overuse_rate(text) > 0


def test_a8_double_passive_golden_case():
    text = "이 문제는 분석되어진다."

    detection = local_detect(text, focus_categories=["A-8"])

    assert "A-8" in {finding.category for finding in detection.findings}
    assert double_passive_count(text) >= 1


async def test_fast_mode_rolls_back_when_change_rate_exceeds_half():
    class OverRewriteLLM:
        async def rewrite(self, request):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, context):
            return FastRewriteResult(
                revisedText="완전히 다른 결론과 새로운 주장으로 바뀐 문장입니다.",
                changes=[
                    Change(
                        original="원문",
                        revised="완전히 다른 문장",
                        reason="테스트용 과윤문입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["과도하게 바꿨습니다."],
                inputTokens=3,
                outputTokens=4,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="2026년 5월 보고서 문장을 명확하게 정리합니다.", rewrite_mode="fast")
    )
    response = await RewriteGraphRunner(_settings(), OverRewriteLLM()).run(request)

    assert response.revisedText == request.text
    assert response.usage.rounds == 1
    assert any("원문으로 롤백" in warning for warning in response.warnings)


async def test_strict_change_rate_alone_is_review_signal_not_rollback():
    class HighChangeStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=1)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            return StrictRewriteResult(
                revisedText="나가 라다 바마 아사 차자 타카 하파 파하 카타 자차 사아 마바 다라 가나.",
                changes=[
                    Change(
                        original=request.text,
                        revised="나가 라다 바마 아사 차자 타카 하파 파하 카타 자차 사아 마바 다라 가나.",
                        reason="테스트용 고변경률 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["변경률은 높지만 보존 누락은 없습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(request, revised_text)

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="가나 다라 마바 사아 자차 카타 파하 하파 타카 차자 아사 바마 라다 나가.",
            rewrite_mode="strict",
            max_rounds=2,
            protected_terms=[],
        )
    )

    response = await RewriteGraphRunner(_settings(), HighChangeStrictLLM()).run(request)

    assert response.revisedText != request.text
    assert response.usage.rounds == 1
    assert any("change_rate_over_50" in warning for warning in response.warnings)
    assert not any("원문을 반환" in warning for warning in response.warnings)


async def test_strict_graph_calls_detect_rewrite_audit_review_nodes():
    calls = []

    class StrictFakeLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            calls.append("detect")
            return DetectionResult(
                sentenceCount=1,
                inputTokens=1,
                outputTokens=2,
            )

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            calls.append("rewrite")
            return StrictRewriteResult(
                revisedText="2026년 보고서입니다.",
                changes=[
                    Change(
                        original="2026년 보고서입니다.",
                        revised="2026년 보고서입니다.",
                        reason="의미를 유지했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["정밀 파이프라인을 통과했습니다."],
                inputTokens=3,
                outputTokens=4,
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.", inputTokens=5, outputTokens=6)

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            calls.append("review")
            return _strict_review_result(request, revised_text).model_copy(
                update={"inputTokens": 7, "outputTokens": 8}
            )

    runner = RewriteGraphRunner(_settings(), StrictFakeLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict")
        )
    )

    assert calls == ["detect", "rewrite", "audit", "review", "audit"]
    assert response.usage.rounds == 1
    assert response.usage.inputTokens == 21
    assert response.usage.outputTokens == 26


async def test_strict_conditional_audit_is_handled_by_review_without_rewrite_loop():
    calls = []
    audit_calls = 0

    class ConditionalAuditLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            calls.append("detect")
            return DetectionResult(sentenceCount=1)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            calls.append("rewrite")
            return StrictRewriteResult(
                revisedText="2026년 보고서입니다.",
                changes=[
                    Change(
                        original="2026년 보고서입니다.",
                        revised="2026년 보고서입니다.",
                        reason="의미를 유지했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["정밀 파이프라인을 통과했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            nonlocal audit_calls
            audit_calls += 1
            calls.append("audit")
            status = "conditional_pass" if audit_calls == 1 else "full_pass"
            return AuditResult(status=status, reason="조건부 감사 결과입니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            calls.append("review")
            return _strict_review_result(request, revised_text)

    runner = RewriteGraphRunner(_settings(), ConditionalAuditLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict", max_rounds=2)
        )
    )

    assert calls == ["detect", "rewrite", "audit", "review", "audit"]
    assert response.usage.rounds == 1
    assert response.revisedText == "2026년 보고서입니다."


async def test_strict_blocks_truncated_final_draft_without_retry_loop():
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 묶어 "
        "폴더를 만들고 자동으로 정리하는 기능입니다.\n\n"
        "두 번째는 소스 검색 기능입니다. 사용자가 직접 피드를 스크롤하지 않아도 에이전트가 필요한 "
        "자료를 찾아 작은 묶음으로 반환합니다.\n\n"
        "이 기능이 제대로 작동하면 앱은 단순한 캡처 도구를 넘어 작업에 바로 쓰이는 컨텍스트 "
        "시스템이 됩니다."
    )

    class TruncatedStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=6)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            return StrictRewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트가",
                changes=[
                    Change(
                        original="source",
                        revised="truncated",
                        reason="테스트용 잘림 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["중간에서 잘린 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(request, revised_text)

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=2, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), TruncatedStrictLLM()).run(request)

    assert response.revisedText == request.text
    assert response.usage.rounds == 1
    assert any("결과 노출을 차단" in warning for warning in response.warnings)
    assert any("Strict 최종 후보에서 출력 잘림" in warning for warning in response.warnings)


async def test_strict_fallback_warnings_do_not_describe_returned_original_as_missing_terms():
    text = (
        '첫 번째는 소스 정리 기능입니다. "소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 '
        '새로운 에이전트 스킬을 만드는 데 사용해줘"라는 요청을 보존해야 합니다. '
        "이 기능은 저장된 자료를 바탕으로 작업 컨텍스트를 만듭니다."
    )

    class MissingQuoteStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=3)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            return StrictRewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다.",
                changes=[
                    Change(
                        original="source",
                        revised="truncated",
                        reason="테스트용 보존 누락 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["보존 문구가 빠진 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            if "소셜미디어 성장" not in revised_text:
                return AuditResult(
                    status="fail",
                    warnings=["직접 인용 또는 핵심 구절이 누락됐습니다."],
                    flaggedEdits=[
                        {
                            "issue": "직접 인용 또는 핵심 구절 누락",
                            "checklistFailed": [4, 13],
                            "action": "restore_original",
                            "correctionDirection": "누락된 직접 인용을 원문 그대로 복원합니다.",
                            "severity": "high",
                        }
                    ],
                    reason="누락 감지",
                )
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(request, revised_text)

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=1, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), MissingQuoteStrictLLM()).run(request)

    assert response.revisedText == request.text
    assert not any("보존되어야 하는 표현이 결과" in warning for warning in response.warnings)
    assert any("누락" in warning for warning in response.warnings)
    assert any("원문을 반환" in warning for warning in response.warnings)


async def test_strict_review_blocking_issues_block_current_draft_without_audit_rollback():
    class HoldReviewLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=1)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            return StrictRewriteResult(
                revisedText="문장을 더 세련되고 자연스럽게 정리합니다.",
                changes=[
                    Change(
                        original=request.text,
                        revised="문장을 더 세련되고 자연스럽게 정리합니다.",
                        reason="테스트용 hold 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["hold 대상 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="감사는 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(
                request,
                revised_text,
                warnings=["사람 검토가 필요합니다."],
                blocking=["최종 리뷰에서 의미 보존 위험이 남았습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="문장을 조금 더 자연스럽게 정리합니다.",
            rewrite_mode="strict",
            max_rounds=1,
            protected_terms=[],
        )
    )
    response = await RewriteGraphRunner(_settings(), HoldReviewLLM()).run(request)

    assert response.revisedText == request.text
    assert any("사람 검토" in warning for warning in response.warnings)


async def test_strict_rewrite_does_not_repair_incomplete_revised_text_from_change_candidate():
    audited_texts = []
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 제가 저장해둔 모든 소스를 살펴보고, "
        "관련 있는 자료끼리 묶어서 폴더를 만들고 자동으로 정리하도록 하는 기능입니다. "
        "두 번째는 소스 검색 기능입니다. 더 이상 제가 직접 피드를 스크롤하면서 자료를 찾는 대신, "
        "에이전트에게 “소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 새로운 에이전트 스킬을 만드는 데 사용해줘”라고 "
        "요청할 수 있게 만드는 것입니다. 그러면 에이전트는 현재 작업에 가장 관련성이 높은 자료들을 골라내고, "
        "유용도에 따라 순위를 매긴 뒤 작은 묶음으로 반환해줍니다."
    )
    complete_revised = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 분석해 관련 자료를 묶고 "
        "폴더를 자동으로 구성하는 기능입니다. 두 번째는 소스 검색 기능입니다. 더 이상 피드를 직접 "
        "스크롤하며 자료를 찾는 대신, 에이전트에게 “소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 "
        "새로운 에이전트 스킬을 만드는 데 사용해줘”라고 요청하면 됩니다. 그러면 에이전트는 해당 작업에 "
        "맞는 자료를 추려 유용도 순으로 정렬한 뒤 작은 묶음으로 돌려줍니다."
    )

    class InconsistentStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=4)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            return StrictRewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트에게 ",
                changes=[
                    Change(
                        original=request.text,
                        revised=complete_revised,
                        reason="전체 윤문입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["revisedText 필드만 불완전한 structured output입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            audited_texts.append(revised_text)
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(request, revised_text)

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=1, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), InconsistentStrictLLM()).run(request)

    assert audited_texts == [
        "첫 번째는 소스 정리 기능입니다. 에이전트에게 ",
        "첫 번째는 소스 정리 기능입니다. 에이전트에게 ",
    ]
    assert response.revisedText == request.text
    assert not any("revisedText가 불완전" in item for item in response.summary)
    assert any("결과 노출을 차단" in warning for warning in response.warnings)


async def test_strict_review_can_return_safe_final_candidate_without_rewrite_loop():
    rewrite_calls = 0
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 묶어 "
        "폴더를 만들고 자동으로 정리하는 기능입니다.\n\n"
        "두 번째는 소스 검색 기능입니다. 사용자가 직접 피드를 스크롤하지 않아도 에이전트가 필요한 "
        "자료를 찾아 작은 묶음으로 반환합니다.\n\n"
        "이 기능이 제대로 작동하면 앱은 단순한 캡처 도구를 넘어 작업에 바로 쓰이는 컨텍스트 "
        "시스템이 됩니다."
    )
    safe_text = text.replace("자료를 찾아 작은", "자료를 찾아서 작은")

    class ReviewRepairsLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, context):
            return DetectionResult(sentenceCount=6)

        async def rewrite_strict(
            self,
            request,
            context,
            detection,
        ):
            nonlocal rewrite_calls
            rewrite_calls += 1
            return StrictRewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트가",
                changes=[
                    Change(
                        original="자료",
                        revised="자료",
                        reason="테스트용 strict 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=[f"{rewrite_calls}라운드 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, detection, revised_text, audit_result, residual_detection):
            return _strict_review_result(request, safe_text)

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=2, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), ReviewRepairsLLM()).run(request)

    assert response.revisedText == safe_text
    assert response.usage.rounds == 1
    assert rewrite_calls == 1


def test_text_above_core_max_chars_returns_422():
    client = _client()
    raw_body = json.dumps(_payload(text="가" * 10_001), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_logs_do_not_include_source_or_rewrite_result(caplog):
    client = _client()
    source = "PRIVACY_SENTINEL_2026 원문입니다."
    raw_body = json.dumps(_payload(text=source), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with caplog.at_level(logging.INFO, logger="humanize_core"):
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    revised = response.json()["revisedText"]
    assert source not in caplog.text
    assert revised not in caplog.text


def test_openai_text_format_uses_strict_json_schema():
    text_format = _openai_rewrite_text_format()

    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    schema = text_format["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["revisedText", "changes", "summary"]
    change_schema = schema["properties"]["changes"]["items"]
    assert change_schema["additionalProperties"] is False
    assert change_schema["required"] == ["original", "revised", "reason", "type", "riskLevel"]


async def test_openai_rewrite_uses_responses_structured_output(monkeypatch):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "revisedText": "개선된 문장입니다.",
                        "changes": [
                            {
                                "original": "원문",
                                "revised": "개선된 문장",
                                "reason": "명확성을 높였습니다.",
                                "type": "clarity",
                                "riskLevel": "low",
                            }
                        ],
                        "summary": ["표현을 정리했습니다."],
                    },
                    ensure_ascii=False,
                ),
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenAIRewriteLLM(api_key="test-key", model_name="gpt-5-mini")
    result = await llm.rewrite(
        RewriteRequestForTest.model_validate(_payload()),
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 11
    assert result.outputTokens == 7
    assert calls[0]["model"] == "gpt-5-mini"
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert "response_format" not in calls[0]


async def test_openrouter_fast_rewrite_uses_json_schema_and_usage(monkeypatch):
    calls = []
    init_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "revisedText": "개선된 문장입니다.",
                                    "changes": [
                                        {
                                            "original": "원문",
                                            "revised": "개선된 문장",
                                            "reason": "AI 티를 줄였습니다.",
                                            "type": "clarity",
                                            "riskLevel": "low",
                                        }
                                    ],
                                    "summary": ["표현을 정리했습니다."],
                                    "warnings": [],
                                    "selfCheck": [],
                                    "residualFindings": [],
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=13, completion_tokens=8),
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenRouterRewriteLLM(
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        app_title="Test App",
        site_url="https://example.test",
        model_name="stub",
        fast_model_name="openai/gpt-5-mini",
        fast_fallback_model_name="~anthropic/claude-haiku-latest",
        strict_detect_model_name="openai/gpt-5-mini",
        strict_rewrite_model_name="~anthropic/claude-sonnet-latest",
        strict_audit_model_name="openai/gpt-5",
        strict_review_model_name="~anthropic/claude-haiku-latest",
        strict_escalation_model_name="~anthropic/claude-opus-latest",
    )

    result = await llm.rewrite_fast(
        RewriteRequestForTest.model_validate(_payload()),
        context={},
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 13
    assert result.outputTokens == 8
    assert init_calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert init_calls[0]["default_headers"]["X-Title"] == "Test App"
    assert calls[0]["model"] == "openai/gpt-5-mini"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert "temperature" not in calls[0]


async def test_openrouter_strict_detect_omits_temperature_for_parameter_routing(monkeypatch):
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"sentenceCount": 1}, ensure_ascii=False)
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenRouterRewriteLLM(
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        app_title="Test App",
        site_url=None,
        model_name="openai/gpt-5-mini",
        fast_model_name="openai/gpt-5-mini",
        fast_fallback_model_name="openai/gpt-5-mini",
        strict_detect_model_name="openai/gpt-5-mini",
        strict_rewrite_model_name="openai/gpt-5-mini",
        strict_audit_model_name="openai/gpt-5-mini",
        strict_review_model_name="openai/gpt-5-mini",
        strict_escalation_model_name="openai/gpt-5-mini",
    )

    result = await llm.detect(
        RewriteRequestForTest.model_validate(_payload(rewrite_mode="strict")),
        context={},
    )

    assert result.inputTokens == 5
    assert result.outputTokens == 3
    assert calls[0]["model"] == "openai/gpt-5-mini"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert "temperature" not in calls[0]


def test_openrouter_schema_marks_pydantic_default_fields_required():
    response_format = _openrouter_response_format(
        "detection_result",
        DetectionResult.model_json_schema(),
    )

    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"].keys())
    assert "default" not in json.dumps(schema)
    assert "title" not in json.dumps(schema)

    finding_schema = schema["$defs"]["Finding"]
    assert finding_schema["required"] == list(finding_schema["properties"].keys())
    assert "textSpan" in finding_schema["required"]
    assert "start" in finding_schema["required"]
    assert "end" in finding_schema["required"]

    assert schema["properties"]["sentenceLengthStats"] == {
        "additionalProperties": False,
        "properties": {},
        "required": [],
        "type": "object",
    }
    assert schema["properties"]["categorySummary"] == {
        "additionalProperties": False,
        "properties": {},
        "required": [],
        "type": "object",
    }


def test_metrics_v2_computes_im_not_ai_signal_keys():
    metrics = compute_all_v2("결론적으로, API를 통해 성과를 확인할 수 있다.")

    assert metrics["version"] == "v2.0"
    assert "v2_metrics" in metrics
    assert "normalisation_score" in metrics["v2_metrics"]
    assert "v2_interference_index" in metrics
