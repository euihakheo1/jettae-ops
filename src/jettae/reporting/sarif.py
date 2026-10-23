"""SARIF 2.1.0 export — findings as code-scanning alerts on the runbooks themselves.

Why SARIF: GitHub code scanning, GitLab, Azure DevOps and most IDE viewers ingest it as-is, so a
runbook repository can run ``jettae scan`` in CI and see broken response paths annotated on the
exact runbook line, next to ordinary code alerts. Only the fixed rules' verdicts are exported (P1);
nothing is re-judged here.

Design notes:

* ``partialFingerprints['jettaeFindingId/v1']`` is the deterministic :attr:`Finding.id`, so an alert
  keeps its identity across runs and is closed automatically once an approved patch fixes it.
* Locations come only from the finding's evidence spans (P2). A finding *about something missing*
  ("감지 경로 없음") has no sentence to cite; it is placed on line 1 of the first document with a
  location message saying so, because code-scanning consumers drop results without a location.
* Document hashes in :class:`DocumentRef` are of the *normalized* text, not of the file bytes, so
  they are published as a property rather than in SARIF ``hashes`` (whose consumers would compare
  them against the raw file and report a mismatch).
* The output is a plain ``dict`` of JSON types, built in a fixed order from sorted inputs (P6):
  ``json.dumps`` of two exports of the same report is byte-identical.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from datetime import UTC
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Final
from urllib.parse import quote

from jettae.check.registry import rule_catalog, sort_findings
from jettae.domain import (
    ELEMENT_LABEL_KO,
    DiagnosisReport,
    DocumentRef,
    Finding,
    FindingStatus,
    OpsGraph,
    RuleInfo,
    Severity,
    SourceSpan,
)
from jettae.reporting.evidence import (
    FINDING_STATUS_LABEL_KO,
    SEVERITY_LABEL_KO,
    distinct_spans,
    printable,
)

__all__ = [
    "FINGERPRINT_KEY",
    "INFORMATION_URI",
    "LEVEL_BY_SEVERITY",
    "SARIF_SCHEMA",
    "SARIF_VERSION",
    "TOOL_NAME",
    "artifact_uri",
    "to_sarif",
]

SARIF_SCHEMA: Final = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
)
SARIF_VERSION: Final = "2.1.0"
TOOL_NAME: Final = "JETTAE Ops"
TOOL_FULL_NAME: Final = "제때옵스(JETTAE Ops) 런북 준비도 진단"
INFORMATION_URI: Final = "https://github.com/euihakheo1/jettae-ops"
FINGERPRINT_KEY: Final = "jettaeFindingId/v1"

#: critical/high block a merge (error), medium asks for attention (warning), the rest informs (note).
LEVEL_BY_SEVERITY: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}

#: Rule-id family (``JT-<FAMILY>-nnn``) → tag shown in code-scanning filters.
FAMILY_TAGS: Final[dict[str, str]] = {
    "PATH": "response-path",
    "PPL": "people",
    "ESC": "escalation",
    "CHN": "channel",
    "DDL": "deadline",
    "KR": "kr-regulation",
    "ONC": "on-call",
    "CAT": "catalog",
    "INT": "integrity",
    "SPF": "single-point-of-failure",
}

_NO_EVIDENCE_NOTE: Final = (
    "이 판정은 인용할 원문이 없습니다(런북에 해당 내용이 없거나 명부·카탈로그 대조로 판정). "
    "위치는 문서 첫 줄로 표시합니다."
)
_NO_LINE_NOTE: Final = "줄 번호 기록이 없어 정규화된 문서 기준 문자 위치로 표시합니다."
_WINDOWS_ABSOLUTE: Final = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


def _complete_drive(drive: str) -> bool:
    """A drive letter, or a UNC drive naming both server and share (``\\\\server\\share``).

    Python 3.11's pathlib calls ``\\\\server\\`` absolute (and makes ``file://server//`` of it) while
    3.12 does not; requiring the share keeps the URI the same on every supported version.
    """
    if not drive.startswith(("\\\\", "//")):
        return True
    return len([part for part in re.split(r"[\\/]", drive) if part]) == 2


#: SARIF §3.11.5/§3.11.6: literal braces are doubled (they would read as ``{0}`` placeholders) and
#: literal brackets are backslash-escaped (they would read as embedded ``[text](link)`` links).
_MESSAGE_ESCAPES: Final = str.maketrans({"[": "\\[", "]": "\\]", "{": "{{", "}": "}}"})


def _text(value: str, *, multiline: bool = False) -> str:
    """A SARIF message string: no control or bidi characters (P5), message syntax escaped.

    Whitespace is collapsed per line; ``multiline`` keeps the line breaks (rule help text).
    """
    lines = value.split("\n") if multiline else [value]
    return "\n".join(" ".join(printable(line).split()) for line in lines).translate(_MESSAGE_ESCAPES)


def artifact_uri(ref: DocumentRef) -> str:
    """The SARIF ``artifactLocation.uri`` of a document.

    A relative path becomes a relative URI with forward slashes, which code-scanning services
    resolve against the repository root. ``..`` segments cannot climb above that root: an uploaded
    file name such as ``../../etc/x.md`` is untrusted (P5) and a URI escaping the repository can
    never point at the runbook, so it is confined the way a web server confines request paths. An
    absolute path becomes a ``file:`` URI (honest, even if no repository can map it). Without a
    usable path the title stands in. Non-ASCII characters (Korean file names) are percent-encoded,
    because a SARIF ``uri`` must be a valid URI reference.
    """
    path = printable(ref.path or "").strip()
    if _WINDOWS_ABSOLUTE.match(path):
        windows = PureWindowsPath(path)
        if windows.is_absolute() and _complete_drive(windows.drive):
            return windows.as_uri()
        # "\\host" without a share (or an upload named "\\x.md") is no complete UNC path and has
        # no file URI — pathlib raises for it — so it is named like any other relative path.
        posix = path.replace("\\", "/").lstrip("/")
    else:
        posix = path.replace("\\", "/")
        if posix.startswith("/"):
            return PurePosixPath(posix).as_uri()
    # normpath of "/"-anchored text drops every ".." that would leave the root.
    confined = posixpath.normpath("/" + posix).lstrip("/") if posix else ""
    if not confined:
        return quote(" ".join(printable(ref.title).split()) or ref.doc_id, safe="")
    return quote(confined, safe="/")


def _rule_family(rule_id: str) -> str:
    parts = rule_id.split("-")
    return parts[1] if len(parts) >= 3 else rule_id


def _rule_help(info: RuleInfo) -> str:
    lines = [info.title, "", f"판정 근거: {info.rationale}"]
    if info.legal_basis:
        lines.append(f"법적 근거: {info.legal_basis}")
    element = ELEMENT_LABEL_KO[info.element] if info.element is not None else "해당 없음"
    lines += [
        f"기본 심각도: {SEVERITY_LABEL_KO[info.severity]} · 점검 요소: {element}",
        "",
        "고정 규칙으로만 판정하며 판정에 생성형 AI를 쓰지 않습니다. "
        "LLM을 켠 진단에서는 역번역 대조의 AI 판정이 요소를 판단 보류로 돌릴 수 있고, "
        "그 요소는 JT-INT-001로 알립니다. "
        "확인할 수 없는 항목은 정상으로 보지 않고 '확인 불가'로 알립니다.",
    ]
    return "\n".join(lines)


def _rule_descriptor(info: RuleInfo) -> dict[str, Any]:
    family = _rule_family(info.rule_id)
    tags = ["jettae", "runbook", "incident-response", FAMILY_TAGS.get(family, family.lower())]
    if info.element is not None:
        tags.append(info.element.value)
    if info.legal_basis:
        tags.append("legal")
    # The schema requires unique tags; a future family named like a fixed tag ("JT-LEGAL-…") or an
    # element must not make the whole log invalid.
    properties: dict[str, Any] = {"tags": list(dict.fromkeys(tags)), "severity": info.severity.value}
    if info.legal_basis:
        properties["legalBasis"] = info.legal_basis
    return {
        "id": info.rule_id,
        "shortDescription": {"text": _text(info.title)},
        "fullDescription": {"text": _text(info.rationale)},
        "help": {"text": _text(_rule_help(info), multiline=True)},
        "defaultConfiguration": {"level": LEVEL_BY_SEVERITY[info.severity]},
        "properties": properties,
    }


class _Artifacts:
    """Documents as SARIF artifacts, indexed in a deterministic order (by URI, then id)."""

    def __init__(self, docs: Mapping[str, DocumentRef]) -> None:
        ordered = sorted(docs.values(), key=lambda d: (artifact_uri(d), d.doc_id))
        self.refs = ordered
        self.index = {d.doc_id: i for i, d in enumerate(ordered)}

    def entries(self) -> list[dict[str, Any]]:
        return [
            {
                "location": {"uri": artifact_uri(d)},
                "description": {"text": _text(d.title)},
                "properties": {"docId": d.doc_id, "normalizedTextSha256": d.sha256},
            }
            for d in self.refs
        ]

    def location(self, doc_id: str) -> dict[str, Any]:
        position = self.index.get(doc_id)
        if position is None:
            # Evidence from a document the caller did not list: cite it by id rather than drop it.
            return {"uri": quote(doc_id, safe="")}
        return {"uri": artifact_uri(self.refs[position]), "index": position}


def _snippet(quote_text: str) -> str:
    """The quote as written (line breaks and spacing kept) minus control and bidi characters.

    Not a message string, so no SARIF escaping; but viewers print snippets verbatim, in a browser
    or a terminal, so an escape sequence or bidi override from a document must not reach them (P5).
    """
    return "\n".join(printable(line) for line in quote_text.split("\n"))


def _span_location(span: SourceSpan, artifacts: _Artifacts) -> dict[str, Any]:
    snippet = {"text": _snippet(span.quote)}
    location: dict[str, Any] = {"physicalLocation": {"artifactLocation": artifacts.location(span.doc_id)}}
    if span.line is not None:
        region: dict[str, Any] = {"startLine": span.line}
        extra_lines = span.quote.count("\n")
        if extra_lines:
            region["endLine"] = span.line + extra_lines
        region["snippet"] = snippet
    else:
        region = {"charOffset": span.start, "charLength": span.end - span.start, "snippet": snippet}
        location["message"] = {"text": _NO_LINE_NOTE}
    location["physicalLocation"]["region"] = region
    return location


def _locations(finding: Finding, artifacts: _Artifacts) -> list[dict[str, Any]]:
    spans = distinct_spans(finding.evidence)
    if spans:
        return [_span_location(span, artifacts) for span in spans]
    if not artifacts.refs:
        return []
    first = artifacts.refs[0]
    return [
        {
            "physicalLocation": {
                "artifactLocation": artifacts.location(first.doc_id),
                "region": {"startLine": 1},
            },
            "message": {"text": _NO_EVIDENCE_NOTE},
        }
    ]


def _message(finding: Finding, service_name: str) -> str:
    status = " [확인 불가: 정상으로 보지 않습니다]" if finding.status is FindingStatus.UNKNOWN else ""
    return _text(f"{finding.title} — {service_name}{status}: {finding.message}")


def _result(
    finding: Finding,
    rule_index: Mapping[str, int],
    service_names: Mapping[str, str],
    artifacts: _Artifacts,
) -> dict[str, Any]:
    service_name = (
        "조직 전체"
        if finding.service_id is None
        else service_names.get(finding.service_id, finding.service_id)
    )
    properties: dict[str, Any] = {
        "service": finding.service_id,
        "serviceName": service_name,
        "element": finding.element.value if finding.element is not None else None,
        "status": finding.status.value,
        "statusLabel": FINDING_STATUS_LABEL_KO[finding.status],
        "severity": finding.severity.value,
        "subjectIds": list(finding.subject_ids),
    }
    if finding.fix is not None:
        properties["fixKind"] = finding.fix.kind.value
    if finding.legal_basis:
        properties["legalBasis"] = finding.legal_basis
    result: dict[str, Any] = {"ruleId": finding.rule_id}
    if finding.rule_id in rule_index:
        result["ruleIndex"] = rule_index[finding.rule_id]
    result.update(
        {
            "kind": "fail",
            "level": LEVEL_BY_SEVERITY[finding.severity],
            "message": {"text": _message(finding, service_name)},
            "locations": _locations(finding, artifacts),
            "partialFingerprints": {FINGERPRINT_KEY: finding.id},
            "properties": properties,
        }
    )
    return result


def _service_names(report: DiagnosisReport, graph: OpsGraph) -> dict[str, str]:
    names = {sid: node.name for sid, node in graph.nodes.items()}
    names.update({s.service_id: s.service_name for s in report.readiness.services})
    return names


def to_sarif(
    report: DiagnosisReport,
    graph: OpsGraph,
    *,
    documents: Mapping[str, DocumentRef],
    tool_version: str,
) -> dict[str, Any]:
    """Export ``report.findings`` as a SARIF 2.1.0 log (one run, every published rule).

    ``documents`` (keyed by ``doc_id``) supply the runbook paths; documents only known to
    ``graph.docs`` are included too, with the passed mapping taking precedence.
    """
    version = tool_version.strip()
    if not version:
        raise ValueError("tool_version이 비어 있습니다. SARIF에는 도구 버전이 필요합니다.")
    catalog = sorted(rule_catalog(), key=lambda info: info.rule_id)
    rule_index = {info.rule_id: i for i, info in enumerate(catalog)}
    artifacts = _Artifacts({**graph.docs, **dict(documents)})
    service_names = _service_names(report, graph)

    invocation: dict[str, Any] = {"executionSuccessful": True}
    created = report.created_at
    if created.tzinfo is not None and created.utcoffset() is not None:
        invocation["endTimeUtc"] = created.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": TOOL_NAME,
                "fullName": TOOL_FULL_NAME,
                "informationUri": INFORMATION_URI,
                "version": version,
                "language": "ko-KR",
                "rules": [_rule_descriptor(info) for info in catalog],
            }
        },
        "automationDetails": {"id": f"jettae-ops/{report.workspace_id}/{report.run_id}"},
        "columnKind": "unicodeCodePoints",
        "artifacts": artifacts.entries(),
        "invocations": [invocation],
        "results": [
            _result(finding, rule_index, service_names, artifacts)
            for finding in sort_findings(report.findings)
        ],
        "properties": {
            "workspaceId": report.workspace_id,
            "runId": report.run_id,
            "graphVersion": report.graph_version,
            "graphHash": report.graph_hash,
            "readiness": report.readiness.overall,
            "weightsVersion": report.readiness.weights_version,
            "llmMode": report.llm.mode.value,
            "llmCalls": report.llm.calls,
            "judgementPathLlmCalls": report.llm.judgement_path_calls,
        },
    }
    return {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}
