"""SARIF 2.1.0 export: required structure, level mapping, locations, fingerprints, determinism."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jettae.check.registry import rule_catalog
from jettae.domain import DiagnosisReport, DocumentRef, Finding, Severity
from jettae.reporting.sarif import (
    FINGERPRINT_KEY,
    INFORMATION_URI,
    LEVEL_BY_SEVERITY,
    SARIF_SCHEMA,
    SARIF_VERSION,
    TOOL_NAME,
    artifact_uri,
    to_sarif,
)
from test_reporting_sample import (
    F_DEPARTED,
    F_HELD,
    F_HOSTILE,
    F_INFO,
    F_LEGAL,
    F_ONCALL,
    FINDINGS,
    GRAPH_HASH,
    PAY_DOC,
    SEC_DOC,
    SPAN_MULTILINE,
    SPAN_SENTENCE,
    documents,
    empty_report,
    sample_graph,
    sample_report,
)

PAY_URI = "runbooks/01-%EA%B2%B0%EC%A0%9C.md"
SEC_URI = "runbooks/03-%EB%B3%B4%EC%95%88%20%EC%8B%A0%EA%B3%A0.md"


def _sarif(
    report: DiagnosisReport | None = None, *, documents_by_id: dict[str, DocumentRef] | None = None
) -> dict[str, Any]:
    docs = {d.doc_id: d for d in documents()} if documents_by_id is None else documents_by_id
    return to_sarif(report or sample_report(), sample_graph(), documents=docs, tool_version="0.1.0")


def _run(log: dict[str, Any]) -> dict[str, Any]:
    run: dict[str, Any] = log["runs"][0]
    return run


def _result(log: dict[str, Any], finding: Finding) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [
        r for r in _run(log)["results"] if r["partialFingerprints"][FINGERPRINT_KEY] == finding.id
    ]
    assert len(matches) == 1
    return matches[0]


def test_log_has_the_required_top_level_structure() -> None:
    log = _sarif()
    assert log["$schema"] == SARIF_SCHEMA
    assert log["version"] == SARIF_VERSION == "2.1.0"
    assert len(log["runs"]) == 1
    driver = _run(log)["tool"]["driver"]
    assert driver["name"] == TOOL_NAME == "JETTAE Ops"
    assert driver["informationUri"] == INFORMATION_URI
    assert driver["version"] == "0.1.0"
    assert driver["language"] == "ko-KR"
    run = _run(log)
    assert run["automationDetails"]["id"] == "jettae-ops/ws-1/run-42"
    assert run["columnKind"] == "unicodeCodePoints"
    assert run["invocations"] == [{"executionSuccessful": True, "endTimeUtc": "2026-10-20T01:02:03Z"}]
    assert run["properties"]["graphHash"] == GRAPH_HASH
    assert run["properties"]["readiness"] == 37.2
    assert run["properties"]["judgementPathLlmCalls"] == 0


def test_rules_publish_the_whole_catalog_in_korean() -> None:
    rules = _run(_sarif())["tool"]["driver"]["rules"]
    catalog = sorted(rule_catalog(), key=lambda i: i.rule_id)
    assert [r["id"] for r in rules] == [i.rule_id for i in catalog]
    for rule, info in zip(rules, catalog, strict=True):
        assert set(rule) == {
            "id",
            "shortDescription",
            "fullDescription",
            "help",
            "defaultConfiguration",
            "properties",
        }
        assert rule["shortDescription"]["text"]
        assert rule["fullDescription"]["text"]
        assert rule["help"]["text"].startswith(info.title.replace("[", "\\[").replace("]", "\\]"))
        assert "판정 근거: " in rule["help"]["text"]
        assert "판정에 생성형 AI를 쓰지 않습니다" in rule["help"]["text"]
        assert rule["defaultConfiguration"]["level"] == LEVEL_BY_SEVERITY[info.severity]
        tags = rule["properties"]["tags"]
        assert tags[:3] == ["jettae", "runbook", "incident-response"]
        assert len(tags) == len(set(tags))
        assert ("legal" in tags) == bool(info.legal_basis)
    by_id = {r["id"]: r for r in rules}
    assert "kr-regulation" in by_id["JT-KR-001"]["properties"]["tags"]
    assert by_id["JT-KR-001"]["properties"]["legalBasis"].startswith("정보통신망법")
    assert "response-path" in by_id["JT-PATH-001"]["properties"]["tags"]


@pytest.mark.parametrize(
    ("severity", "level"),
    [
        (Severity.CRITICAL, "error"),
        (Severity.HIGH, "error"),
        (Severity.MEDIUM, "warning"),
        (Severity.LOW, "note"),
        (Severity.INFO, "note"),
    ],
)
def test_severity_maps_to_sarif_level(severity: Severity, level: str) -> None:
    assert LEVEL_BY_SEVERITY[severity] == level
    finding = F_ONCALL.model_copy(update={"severity": severity})
    (result,) = _run(_sarif(sample_report(findings=(finding,))))["results"]
    assert result["level"] == level


def test_every_finding_becomes_one_result_in_published_order() -> None:
    results = _run(_sarif())["results"]
    assert [r["partialFingerprints"][FINGERPRINT_KEY] for r in results] == [
        F_LEGAL.id,
        F_DEPARTED.id,
        F_HOSTILE.id,
        F_HELD.id,
        F_ONCALL.id,
        F_INFO.id,
    ]
    rules = _run(_sarif())["tool"]["driver"]["rules"]
    for result in results:
        assert {"ruleId", "ruleIndex", "kind", "level", "message", "locations", "partialFingerprints"} <= set(
            result
        )
        assert rules[result["ruleIndex"]]["id"] == result["ruleId"]
        assert result["kind"] == "fail"
        assert set(result["properties"]) >= {"service", "element", "status"}


def test_result_properties_and_message() -> None:
    result = _result(_sarif(), F_DEPARTED)
    assert result["message"]["text"] == (
        "퇴사자에게 대응이 걸려 있음 — 결제: '결제' 서비스의 호출이 퇴사자 '김민준'에게 걸려 있습니다."
    )
    props = result["properties"]
    assert props["service"] == "service:결제"
    assert props["serviceName"] == "결제"
    assert props["element"] is None
    assert props["status"] == "broken"
    assert props["statusLabel"] == "끊김"
    assert props["fixKind"] == "replace_departed"
    assert props["severity"] == "critical"
    held = _result(_sarif(), F_HELD)
    assert held["properties"]["status"] == "unknown"
    assert "[확인 불가: 정상으로 보지 않습니다]" not in held["message"]["text"]
    assert "\\[확인 불가: 정상으로 보지 않습니다\\]" in held["message"]["text"]
    legal = _result(_sarif(), F_LEGAL)
    assert legal["properties"]["service"] is None
    assert legal["properties"]["serviceName"] == "조직 전체"
    assert legal["properties"]["legalBasis"].startswith("정보통신망법")


def test_locations_come_from_distinct_evidence_spans() -> None:
    locations = _result(_sarif(), F_DEPARTED)["locations"]
    assert len(locations) == 2  # the word span sits inside the sentence span
    first, second = (loc["physicalLocation"] for loc in locations)
    assert first["artifactLocation"] == {"uri": PAY_URI, "index": 0}
    assert first["region"] == {"startLine": 3, "snippet": {"text": SPAN_SENTENCE.quote}}
    assert second["region"] == {"startLine": 5, "endLine": 6, "snippet": {"text": SPAN_MULTILINE.quote}}


def test_spans_without_a_line_use_character_offsets_with_a_note() -> None:
    first, second = _result(_sarif(), F_LEGAL)["locations"]
    assert first["physicalLocation"]["artifactLocation"] == {"uri": SEC_URI, "index": 1}
    assert first["physicalLocation"]["region"]["startLine"] == 13
    assert second["physicalLocation"]["region"] == {
        "charOffset": 200,
        "charLength": 10,
        "snippet": {"text": "줄 없음 인용"},
    }
    assert "문자 위치" in second["message"]["text"]


def test_findings_without_evidence_point_at_line_one_of_the_first_document() -> None:
    (location,) = _result(_sarif(), F_HOSTILE)["locations"]
    assert location["physicalLocation"] == {
        "artifactLocation": {"uri": PAY_URI, "index": 0},
        "region": {"startLine": 1},
    }
    assert "인용할 원문이 없습니다" in location["message"]["text"]
    # documents known only to the graph are used when the caller passes none
    (graph_only,) = _result(_sarif(documents_by_id={}), F_ONCALL)["locations"]
    assert graph_only["physicalLocation"]["artifactLocation"] == {"uri": PAY_URI, "index": 0}


def test_without_any_document_results_have_no_locations() -> None:
    graph = sample_graph().model_copy(update={"docs": {}})
    log = to_sarif(sample_report(findings=(F_HOSTILE,)), graph, documents={}, tool_version="1.0")
    (result,) = _run(log)["results"]
    assert result["locations"] == []
    assert _run(log)["artifacts"] == []


def test_evidence_from_an_unlisted_document_is_cited_by_id() -> None:
    graph = sample_graph().model_copy(update={"docs": {}})
    finding = F_DEPARTED.model_copy(
        update={"evidence": (SPAN_SENTENCE.model_copy(update={"doc_id": "doc x"}),)}
    )
    log = to_sarif(sample_report(findings=(finding,)), graph, documents={}, tool_version="1.0")
    (location,) = _run(log)["results"][0]["locations"]
    assert location["physicalLocation"]["artifactLocation"] == {"uri": "doc%20x"}


def test_artifacts_list_documents_with_titles_and_normalized_hashes() -> None:
    artifacts = _run(_sarif())["artifacts"]
    assert [a["location"]["uri"] for a in artifacts] == [PAY_URI, SEC_URI]
    assert artifacts[1]["description"]["text"] == "보안 | 신고 <절차>"
    assert artifacts[0]["properties"] == {"docId": PAY_DOC, "normalizedTextSha256": "a" * 64}
    assert "hashes" not in artifacts[0]


def test_passed_documents_take_precedence_over_graph_documents() -> None:
    renamed = DocumentRef(doc_id=SEC_DOC, title="신고 절차", sha256="d" * 64, path="docs/report.md")
    artifacts = _run(_sarif(documents_by_id={SEC_DOC: renamed}))["artifacts"]
    assert [a["location"]["uri"] for a in artifacts] == ["docs/report.md", PAY_URI]


@pytest.mark.parametrize(
    ("path", "title", "uri"),
    [
        ("runbooks/01-결제.md", "t", PAY_URI),
        ("runbooks\\sub\\a b.md", "t", "runbooks/sub/a%20b.md"),
        ("./runbooks//a.md", "t", "runbooks/a.md"),
        ("/srv/runbooks/a.md", "t", "file:///srv/runbooks/a.md"),
        ("C:\\runbooks\\결제.md", "t", "file:///C:/runbooks/%EA%B2%B0%EC%A0%9C.md"),
        ("C:/runbooks/a.md", "t", "file:///C:/runbooks/a.md"),
        (None, "결제 런북", "%EA%B2%B0%EC%A0%9C%20%EB%9F%B0%EB%B6%81"),
        ("  ", "", "doc-1"),
    ],
)
def test_artifact_uri(path: str | None, title: str, uri: str) -> None:
    assert artifact_uri(DocumentRef(doc_id="doc-1", title=title, sha256="0" * 64, path=path)) == uri


def test_message_strings_escape_sarif_syntax_and_drop_control_characters() -> None:
    text = _result(_sarif(), F_HOSTILE)["message"]["text"]
    assert "\\[링크\\](http://evil.example)" in text
    assert "{{0}}" in text
    assert "\x1b" not in text
    assert "\u202e" not in text
    assert text.startswith("대체 채널 없음 — 주문|정산: <script>")


def test_findings_of_rules_outside_the_catalog_have_no_rule_index() -> None:
    custom = F_ONCALL.model_copy(update={"rule_id": "JT-XYZ-001", "id": "JT-XYZ-001:1"})
    (result,) = _run(_sarif(sample_report(findings=(custom,))))["results"]
    assert result["ruleId"] == "JT-XYZ-001"
    assert "ruleIndex" not in result


def test_naive_timestamps_are_not_converted() -> None:
    run = _run(to_sarif(empty_report(), sample_graph(), documents={}, tool_version="1"))
    assert run["invocations"] == [{"executionSuccessful": True}]
    assert run["results"] == []


@pytest.mark.parametrize("version", ["", "   "])
def test_tool_version_is_required(version: str) -> None:
    with pytest.raises(ValueError, match="tool_version"):
        to_sarif(sample_report(), sample_graph(), documents={}, tool_version=version)


def test_log_is_json_serialisable_and_deterministic() -> None:
    first = json.dumps(_sarif(), ensure_ascii=False, sort_keys=False)
    assert first == json.dumps(_sarif(), ensure_ascii=False, sort_keys=False)
    assert json.loads(first) == _sarif()


@settings(max_examples=20)
@given(st.permutations(FINDINGS))
def test_finding_order_does_not_change_the_log(findings: list[Finding]) -> None:
    assert _sarif(sample_report(findings=tuple(findings))) == _sarif()


def _walk(value: object) -> list[object]:
    if isinstance(value, dict):
        return [value, *(x for v in value.values() for x in _walk(v))]
    if isinstance(value, list):
        return [value, *(x for v in value for x in _walk(v))]
    return [value]


def test_only_json_types_are_emitted() -> None:
    for value in _walk(_sarif()):
        assert value is None or isinstance(value, (dict, list, str, int, float, bool))
