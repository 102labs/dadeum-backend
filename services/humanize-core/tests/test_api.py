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
from humanize_core.im_not_ai import prompts
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
    NaturalnessReviewResult,
    StrictRewriteResult,
)
from humanize_core.llm import OpenAIRewriteLLM, OpenRouterRewriteLLM, StubRewriteLLM, _openai_rewrite_text_format
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


def test_strict_request_loops_until_hold_when_residual_s1_remains():
    client = _client()
    text = "결론적으로 성과를 냈습니다. 따라서 개선됩니다. 이를 통해 정리합니다. 그러므로 유지합니다."
    raw_body = json.dumps(_payload(text=text, rewrite_mode="strict", max_rounds=3), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["usage"]["rounds"] == 3
    assert any("사람 검토" in warning for warning in data["warnings"])


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
    assert any("사람 검토" in warning for warning in data["warnings"])


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
            "report",
            {"estimatedGenre": "report", "metricsBefore": {}, "preservationTerms": []},
        )
    )

    assert payload["settings"] == {
        "user_intent": "문장을 더 부드럽게 다듬어 주세요.",
        "rewrite_mode": "strict",
        "tone": "friendly",
        "protected_terms": ["2026년"],
        "max_rounds": 3,
        "preserve_formatting": True,
    }
    assert "우선 반영" in payload["rewrite_guidance"]["user_intent"]
    assert "친근" in payload["rewrite_guidance"]["tone"]
    assert "2026년" in payload["rewrite_guidance"]["protected_terms"]
    assert "최대 3라운드" in payload["rewrite_guidance"]["max_rounds"]
    assert "줄바꿈" in payload["rewrite_guidance"]["formatting"]


async def test_stub_rewrite_uses_preserve_formatting_switch():
    preserving = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=True)
    )
    normalizing = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=False)
    )
    llm = StubRewriteLLM()

    preserved = await llm.rewrite(preserving, "report")
    normalized = await llm.rewrite(normalizing, "report")

    assert preserved.revisedText == "첫 문장.  둘째 문장."
    assert normalized.revisedText == "첫 문장. 둘째 문장."


async def test_protected_terms_are_audited_after_rewrite():
    class DropsProtectedTermLLM:
        async def rewrite(self, request, genre_hint):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, genre_hint, context):
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

    assert any("API v1" in warning and "누락" in warning for warning in response.warnings)
    assert response.changes[0].riskLevel == "high"


async def test_internal_genre_hint_is_inferred_from_text():
    captured = {}

    class CapturingLLM:
        async def rewrite(self, request, genre_hint):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, genre_hint, context):
            captured["genre_hint"] = genre_hint
            return FastRewriteResult(
                revisedText=request.text,
                changes=[Change(original="", revised="", reason="유지했습니다.", type="clarity", riskLevel="low")],
                summary=["유지했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="분기별 보고 결과와 분석 내용입니다.")
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    assert captured["genre_hint"] == "report"


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
        ("A-16", "그는 말했다. 그는 다시 말했다. 그것은 반복됐다.", "report"),
        ("B-1", "소버린 AI(Sovereign AI)는 중요한 전략이다.", "report"),
        ("C-9", "(1) 계획을 세운다. (2) 실행한다.", "report"),
        ("E-2", "첫째다. 둘째다. 셋째다. 넷째다.", "report"),
        ("E-7", "우리는 실행한다. 그런데 이건 좋아요. 다음 단계입니다.", "blog"),
        ("F-4", "전략적 실행성과 구조화가 필요하다.", "report"),
        ("G-3", "균형 있게 보고 신중하게 판단하며 양쪽 모두와 두 가지 모두의 장점도 있지만 균형이 필요하다.", "report"),
        ("J-3", "- 첫 번째 항목\n- 두 번째 항목", "report"),
    ]

    for rule_id, text, genre in cases:
        detection = local_detect(text, genre, focus_categories=[rule_id])
        assert rule_id in {finding.category for finding in detection.findings}


def test_local_detection_uses_original_taxonomy_score_and_density():
    detection = local_detect("성과를 통해 결과를 냈다.", "report", focus_categories=["A-2"])

    assert detection.severityWeightedScore == 5.0
    assert detection.aiTellDensity == finding_density("성과를 통해 결과를 냈다.", detection.findings)
    assert finding_score(detection.findings) == 5.0


def test_local_detection_excludes_do_not_spans():
    text = '"데이터를 통해 성장한다"라고 말했다. API는 유지한다.'
    detection = local_detect(text, "report")

    assert "A-2" not in {finding.category for finding in detection.findings}


def test_a16_pronoun_literal_translation_golden_case():
    literal = "메리는 그녀가 그녀를 그리워해서 그녀의 어머니에게 전화했다."
    natural = "메리는 어머니가 그리워서 전화를 걸었다."

    literal_detection = local_detect(literal, "report", focus_categories=["A-16"])
    natural_detection = local_detect(natural, "report", focus_categories=["A-16"])

    assert "A-16" in {finding.category for finding in literal_detection.findings}
    assert "A-16" not in {finding.category for finding in natural_detection.findings}
    assert pronoun_density(literal) > pronoun_density(natural)


def test_a17_deul_overuse_stays_metric_only():
    text = "이러한 데이터들과 정보들과 결과들이 중요한 아이디어들을 보여준다."

    detection = local_detect(text, "report")

    assert "A-17" not in SUPPORTED_QUICK_RULE_IDS
    assert "A-17" not in {finding.category for finding in detection.findings}
    assert deul_overuse_rate(text) > 0


def test_a8_double_passive_golden_case():
    text = "이 문제는 분석되어진다."

    detection = local_detect(text, "report", focus_categories=["A-8"])

    assert "A-8" in {finding.category for finding in detection.findings}
    assert double_passive_count(text) >= 1


async def test_fast_mode_rolls_back_when_change_rate_exceeds_half():
    class OverRewriteLLM:
        async def rewrite(self, request, genre_hint):
            raise AssertionError("fast path should call rewrite_fast")

        async def rewrite_fast(self, request, genre_hint, context):
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


async def test_strict_graph_calls_detect_rewrite_audit_review_nodes():
    calls = []

    class StrictFakeLLM:
        async def rewrite(self, request, genre_hint):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, genre_hint, context):
            calls.append("detect")
            return DetectionResult(
                estimatedGenre="report",
                sentenceCount=1,
                inputTokens=1,
                outputTokens=2,
            )

        async def rewrite_strict(
            self,
            request,
            genre_hint,
            context,
            detection,
            previous_revised_text,
            audit_feedback,
            review_feedback,
            *,
            use_escalation=False,
        ):
            calls.append(("rewrite", use_escalation))
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

        async def audit(self, request, genre_hint, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.", inputTokens=5, outputTokens=6)

        async def review(self, request, genre_hint, context, detection, revised_text, audit_warnings):
            calls.append("review")
            return NaturalnessReviewResult(decision="accept", reason="잔존 패턴이 없습니다.", inputTokens=7, outputTokens=8)

    runner = RewriteGraphRunner(_settings(), StrictFakeLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict")
        )
    )

    assert calls == ["detect", ("rewrite", False), "audit", "review"]
    assert response.usage.rounds == 1
    assert response.usage.inputTokens == 16
    assert response.usage.outputTokens == 20


async def test_strict_conditional_audit_retries_rewrite_round():
    calls = []
    audit_calls = 0

    class ConditionalAuditLLM:
        async def rewrite(self, request, genre_hint):
            raise AssertionError("strict graph should call node-specific methods")

        async def detect(self, request, genre_hint, context):
            calls.append("detect")
            return DetectionResult(estimatedGenre="report", sentenceCount=1)

        async def rewrite_strict(
            self,
            request,
            genre_hint,
            context,
            detection,
            previous_revised_text,
            audit_feedback,
            review_feedback,
            *,
            use_escalation=False,
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

        async def audit(self, request, genre_hint, context, revised_text, changes):
            nonlocal audit_calls
            audit_calls += 1
            calls.append("audit")
            status = "conditional_pass" if audit_calls == 1 else "full_pass"
            return AuditResult(status=status, reason="조건부 감사 결과입니다.")

        async def review(self, request, genre_hint, context, detection, revised_text, audit_warnings):
            calls.append("review")
            return NaturalnessReviewResult(decision="accept", reason="잔존 패턴이 없습니다.")

    runner = RewriteGraphRunner(_settings(), ConditionalAuditLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict", max_rounds=2)
        )
    )

    assert calls == ["detect", "rewrite", "audit", "review", "rewrite", "audit", "review"]
    assert response.usage.rounds == 2
    assert response.revisedText == "2026년 보고서입니다."


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
        genre_hint="report",
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
        genre_hint="report",
        context={"estimatedGenre": "report", "metricsBefore": {}, "preservationTerms": []},
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


def test_metrics_v2_computes_im_not_ai_signal_keys():
    metrics = compute_all_v2("결론적으로, API를 통해 성과를 확인할 수 있다.", genre="essay")

    assert metrics["version"] == "v2.0"
    assert "v2_metrics" in metrics
    assert "normalisation_score" in metrics["v2_metrics"]
    assert "v2_interference_index" in metrics
