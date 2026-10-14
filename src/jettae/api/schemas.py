"""Request bodies and thin response DTOs of the REST API.

Domain models (``DiagnosisReport``, ``Patch``, ``GraphVersion``, ``RehearsalResult`` …) are returned
as they are — the API must not become a second, drifting definition of the contract. The models
here exist only where a screen needs a different *shape* of the same facts:

* :class:`GraphView` — the graph as id-sorted node/edge lists with spans flattened out of the
  provenance and the findings that point at each element, so the web app can draw it without
  re-implementing graph walks;
* :class:`GraphDiffView` — a version comparison with the elements themselves (before/after and
  per-attribute changes), not just their ids;
* :class:`WorkspaceSummary`, :class:`ApprovalResult`, :class:`DemoResult`,
  :class:`FreeScanResult`, :class:`RequestInfo` — small envelopes.

Builders (:func:`graph_view`, :func:`graph_diff_view`, :func:`workspace_summary`) are pure and
deterministic (P6): the same graph and report always give byte-identical JSON.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, get_args

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)

from jettae.domain import (
    Attr,
    AttrValue,
    DiagnosisReport,
    DocumentRef,
    Edge,
    EdgeKind,
    GraphVersion,
    LlmMode,
    Node,
    NodeKind,
    OpsGraph,
    Patch,
    Provenance,
    SourceKind,
    SourceSpan,
    Verification,
)
from jettae.improve import GraphDiff
from jettae.orchestrator.service import ScenarioGeneration
from jettae.store import Workspace

__all__ = [
    "EVENT_NAMES",
    "EVENT_PROPS",
    "EVENT_SCREEN_PATHS",
    "TEAM_SIZES",
    "Accepted",
    "ApprovalResult",
    "AttrChange",
    "CreateWorkspaceRequest",
    "DemoRequest",
    "DemoResult",
    "DiagnoseRequest",
    "DraftRequest",
    "EdgeChange",
    "EventName",
    "EventRequest",
    "Flag",
    "FreeScanResult",
    "GraphDiffView",
    "GraphView",
    "GraphViewEdge",
    "GraphViewNode",
    "HealthStatus",
    "IntRange",
    "NodeChange",
    "OneOf",
    "PropRule",
    "ReadinessStatus",
    "RehearsalRequest",
    "RejectRequest",
    "RequestInfo",
    "SessionInfo",
    "SignupDeleted",
    "SignupRequest",
    "WorkspaceSummary",
    "graph_diff_view",
    "graph_view",
    "workspace_summary",
]

_LLM_MODE_HELP: Final = "off(규칙만, LLM 0회) · live(실제 LLM) · replay(녹화 재생). 비우면 서버 기본값"
_RUN_LLM_MODE_HELP: Final = (
    f"{_LLM_MODE_HELP}. 보기 전용·익명 요청은 이 값을 무시하고 언제나 서버 기본값(JETTAE_LLM_MODE)을 씁니다"
)


# --------------------------------------------------------------------------- workspaces


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"name": "결제 플랫폼팀"}]})

    name: str = Field(
        min_length=1,
        max_length=200,
        description="워크스페이스 이름(1~200자, 한 줄). 앞뒤 공백·보이지 않는 문자는 지우고 NFC로 맞춥니다",
    )


class WorkspaceSummary(BaseModel):
    """A workspace and, when it has been diagnosed, the headline of its latest report."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "ws-b7212db3a067507c",
                    "name": "별빛상점 (DMC 데모)",
                    "created_at": "2026-10-28T14:00:00Z",
                    "latest_version": 1,
                    "graph_hash": "883b5d40d60c09ebc5d73329f370baa77f9c7bec38081ebed3cd208673ceec90",
                    "overall": 55.7,
                    "finding_count": 38,
                    "is_demo": True,
                }
            ]
        },
    )

    id: str
    name: str
    created_at: datetime
    latest_version: int | None = Field(
        default=None, description="최신 진단이 판정한 그래프 버전(진단 전이면 null)"
    )
    graph_hash: str | None = Field(default=None, description="그 그래프의 내용 해시(SHA-256)")
    overall: float | None = Field(default=None, description="최신 진단의 준비도 점수(0~100)")
    finding_count: int | None = Field(default=None, description="최신 진단의 끊김 수")
    is_demo: bool = Field(default=False, description="별빛상점 예제 워크스페이스인지")


def workspace_summary(
    workspace: Workspace, report: DiagnosisReport | None, *, demo_name: str
) -> WorkspaceSummary:
    return WorkspaceSummary(
        id=workspace.id,
        name=workspace.name,
        created_at=workspace.created_at,
        latest_version=report.graph_version if report is not None else None,
        graph_hash=report.graph_hash if report is not None else None,
        overall=report.readiness.overall if report is not None else None,
        finding_count=len(report.findings) if report is not None else None,
        is_demo=workspace.name == demo_name,
    )


# --------------------------------------------------------------------------- run requests


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"llm_mode": "off"}]})

    llm_mode: LlmMode | None = Field(default=None, description=_RUN_LLM_MODE_HELP)


class DemoRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"llm_mode": "replay"}]})

    llm_mode: LlmMode | None = Field(default=None, description=_RUN_LLM_MODE_HELP)


class RehearsalRequest(BaseModel):
    """Which drills to replay on the head graph.

    ``scenario_ids`` runs exactly those drills. Otherwise ``generate`` picks the sources:
    ``template``/``human``/``llm``/``all``; ``true`` means ``all`` and ``false`` replays the
    existing drills without asking an LLM for new ones (LLM mode forced off).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"scenario_ids": ["human:dmc-s1-chuseok-payment"]}, {"generate": "all"}]
        }
    )

    scenario_ids: list[str] | None = Field(
        default=None, min_length=1, max_length=200, description="이 시나리오만 재생(비우면 generate로 고름)"
    )
    generate: ScenarioGeneration | bool = Field(
        default="all",
        description="template·human·llm·all 중 출처. true는 all, false는 LLM에 새 시나리오를 묻지 않고 재생",
    )
    llm_mode: LlmMode | None = Field(default=None, description=_LLM_MODE_HELP)

    def resolved(self) -> tuple[ScenarioGeneration, LlmMode | None]:
        """``generate`` as the service's vocabulary, and the LLM mode to use for it."""
        if self.generate is False:
            return "all", LlmMode.OFF
        if self.generate is True:
            return "all", self.llm_mode
        return self.generate, self.llm_mode


class DraftRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"finding_ids": ["JT-PATH-020:08e9c34f06"]}]})

    finding_ids: list[str] | None = Field(
        default=None, min_length=1, max_length=500, description="이 끊김들에 대해서만(비우면 모든 끊김)"
    )
    llm_mode: LlmMode | None = Field(default=None, description=_LLM_MODE_HELP)


class RejectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"reason": "김서연 사원은 10월 말 휴직 예정이라 담당으로 둘 수 없습니다."}]
        }
    )

    reason: str = Field(min_length=1, max_length=2000, description="거절 사유(감사 로그에 남습니다)")

    @field_validator("reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("거절 사유를 적어 주세요.")
        return value


# --------------------------------------------------------------------------- results


class ApprovalResult(BaseModel):
    """What an approval or a revert produced: the decided patch (if any), the new version and the
    report re-judged on it by the fixed rules (no LLM)."""

    model_config = ConfigDict(frozen=True)

    patch: Patch | None
    version: GraphVersion
    report: DiagnosisReport


class DemoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    workspace_name: str
    report: DiagnosisReport


class FreeScanResult(BaseModel):
    """A finished anonymous free scan (무료 검진): its private workspace, the key that opens it, and
    the headline of the report (the full report: ``GET /workspaces/{workspace_id}/report`` with the
    key in ``X-Jettae-Workspace-Key``)."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "workspace_id": "ws-3f9c2a1b7d4e5f60",
                    "access_key": "q8Zr3N0vYk2bW5mD7xTfLh1sPcJ9uEaG4oRiK6nVwQy",
                    "readiness_overall": 41.2,
                    "findings": 17,
                }
            ]
        },
    )

    workspace_id: str = Field(description="무료 검진이 만든 비공개 워크스페이스 id")
    access_key: str = Field(
        description=(
            "이 워크스페이스를 여는 접근 키(다시 보여 주지 않으므로 보관하세요). "
            "요청 헤더 X-Jettae-Workspace-Key로 보냅니다"
        )
    )
    readiness_overall: float = Field(description="준비도 점수(0~100)")
    findings: int = Field(description="찾은 끊김 수")


# --------------------------------------------------------------------------- graph view


class GraphViewNode(BaseModel):
    """A node for drawing: ``attrs`` keep each value's spans and source, ``values`` are the bare
    values of the same attributes, ``spans`` gather every evidence span of the node."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: NodeKind
    name: str
    attrs: dict[str, Attr]
    values: dict[str, AttrValue]
    verification: Verification
    hold_reason: str | None
    spans: tuple[SourceSpan, ...]
    sources: tuple[SourceKind, ...]


class GraphViewEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: EdgeKind
    src: str
    dst: str
    attrs: dict[str, Attr]
    values: dict[str, AttrValue]
    verification: Verification
    hold_reason: str | None
    spans: tuple[SourceSpan, ...]
    sources: tuple[SourceKind, ...]


class GraphView(BaseModel):
    """One graph version ready to draw.

    ``findings_by_element`` maps a node/edge id to the findings that name it, taken from the latest
    report *only if that report judged this very graph* (``findings_run_id``); for any other
    version it is empty rather than borrowed from a different graph (P3).
    """

    model_config = ConfigDict(frozen=True)

    version: int
    graph_hash: str
    nodes: tuple[GraphViewNode, ...]
    edges: tuple[GraphViewEdge, ...]
    findings_by_element: dict[str, tuple[str, ...]]
    findings_run_id: str | None
    docs: tuple[DocumentRef, ...]


def _sources(prov: Iterable[Provenance]) -> tuple[SourceKind, ...]:
    return tuple(sorted({p.source for p in prov}, key=lambda s: s.value))


def _values(attrs: Mapping[str, Attr]) -> dict[str, AttrValue]:
    return {key: attrs[key].value for key in sorted(attrs)}


def _view_node(node: Node) -> GraphViewNode:
    return GraphViewNode(
        id=node.id,
        kind=node.kind,
        name=node.name,
        attrs=dict(sorted(node.attrs.items())),
        values=_values(node.attrs),
        verification=node.verification,
        hold_reason=node.hold_reason,
        spans=node.spans,
        sources=_sources(node.prov),
    )


def _view_edge(edge: Edge) -> GraphViewEdge:
    return GraphViewEdge(
        id=edge.id,
        kind=edge.kind,
        src=edge.src,
        dst=edge.dst,
        attrs=dict(sorted(edge.attrs.items())),
        values=_values(edge.attrs),
        verification=edge.verification,
        hold_reason=edge.hold_reason,
        spans=edge.spans,
        sources=_sources(edge.prov),
    )


def graph_view(version: GraphVersion, graph: OpsGraph, report: DiagnosisReport | None) -> GraphView:
    """Build the :class:`GraphView` of ``version`` (``graph`` is its content)."""
    judged = report is not None and report.graph_hash == version.graph_hash
    by_element: dict[str, set[str]] = {}
    if report is not None and judged:
        for finding in report.findings:
            for subject in finding.subject_ids:
                by_element.setdefault(subject, set()).add(finding.id)
    return GraphView(
        version=version.number,
        graph_hash=version.graph_hash,
        nodes=tuple(_view_node(graph.nodes[nid]) for nid in sorted(graph.nodes)),
        edges=tuple(_view_edge(graph.edges[eid]) for eid in sorted(graph.edges)),
        findings_by_element={key: tuple(sorted(ids)) for key, ids in sorted(by_element.items())},
        findings_run_id=report.run_id if report is not None and judged else None,
        docs=tuple(graph.docs[doc_id] for doc_id in sorted(graph.docs)),
    )


# --------------------------------------------------------------------------- version diff


class AttrChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    before: Attr | None
    after: Attr | None


class NodeChange(BaseModel):
    """A node present in both versions with different content (``attrs`` lists value changes;
    it is empty when only provenance, name or verification changed)."""

    model_config = ConfigDict(frozen=True)

    id: str
    before: Node
    after: Node
    attrs: tuple[AttrChange, ...]


class EdgeChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    before: Edge
    after: Edge
    attrs: tuple[AttrChange, ...]


class GraphDiffView(BaseModel):
    """What changed from version ``a`` to version ``b`` (every list sorted by id)."""

    model_config = ConfigDict(frozen=True)

    a: int
    b: int
    added_nodes: tuple[Node, ...]
    removed_nodes: tuple[Node, ...]
    changed_nodes: tuple[NodeChange, ...]
    added_edges: tuple[Edge, ...]
    removed_edges: tuple[Edge, ...]
    changed_edges: tuple[EdgeChange, ...]


def _attr_changes(before: Mapping[str, Attr], after: Mapping[str, Attr]) -> tuple[AttrChange, ...]:
    keys = sorted(set(before) | set(after))
    return tuple(
        AttrChange(key=key, before=before.get(key), after=after.get(key))
        for key in keys
        if before.get(key) != after.get(key)
    )


def graph_diff_view(a: int, b: int, diff: GraphDiff, before: OpsGraph, after: OpsGraph) -> GraphDiffView:
    """Resolve the ids of ``diff`` (computed by :mod:`jettae.improve`) against the two graphs."""
    return GraphDiffView(
        a=a,
        b=b,
        added_nodes=tuple(after.nodes[nid] for nid in diff.added_nodes),
        removed_nodes=tuple(before.nodes[nid] for nid in diff.removed_nodes),
        changed_nodes=tuple(
            NodeChange(
                id=nid,
                before=before.nodes[nid],
                after=after.nodes[nid],
                attrs=_attr_changes(before.nodes[nid].attrs, after.nodes[nid].attrs),
            )
            for nid in diff.changed_nodes
        ),
        added_edges=tuple(after.edges[eid] for eid in diff.added_edges),
        removed_edges=tuple(before.edges[eid] for eid in diff.removed_edges),
        changed_edges=tuple(
            EdgeChange(
                id=eid,
                before=before.edges[eid],
                after=after.edges[eid],
                attrs=_attr_changes(before.edges[eid].attrs, after.edges[eid].attrs),
            )
            for eid in diff.changed_edges
        ),
    )


# --------------------------------------------------------------------------- growth


class SignupRequest(BaseModel):
    """Pre-launch sign-up (사전 신청). ``consent_privacy`` must be true — personal data is never
    accepted without consent. ``org_name`` is accepted as an alias of ``organization``."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        coerce_numbers_to_str=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "email": "sre-lead@example.com",
                    "organization": "별빛상점",
                    "team_size": "10-49",
                    "message": "추석 대비 결제 런북을 점검하고 싶습니다.",
                    "consent_privacy": True,
                    "consent_marketing": False,
                    "source": "landing",
                }
            ]
        },
    )

    email: str = Field(min_length=3, max_length=254)
    organization: str | None = Field(
        default=None, max_length=200, validation_alias=AliasChoices("organization", "org_name")
    )
    team_size: str | None = Field(default=None, max_length=50)
    message: str | None = Field(default=None, max_length=2000)
    consent_privacy: bool = Field(description="개인정보 수집·이용 동의(필수, true여야 저장)")
    consent_marketing: bool = False
    source: str | None = Field(default=None, max_length=64)


#: Demand events the landing page and the app may record (anything else is refused).
#: ``landing_first_view`` is sent once per browser (first visit); ``runbook_upload`` is recorded by
#: the server for each free scan and is deliberately *not* accepted from clients.
EventName = Literal[
    "landing_view", "landing_first_view", "upload_click", "signup_click", "demo_start", "report_download"
]
EVENT_NAMES: Final[tuple[str, ...]] = get_args(EventName)

#: Team sizes the landing page's sign-up form offers (``web/src/pages/landing/SignupForm.tsx``).
TEAM_SIZES: Final[tuple[str, ...]] = ("1~10명", "11~50명", "51~200명", "201~1,000명", "1,000명 이상")


@dataclass(frozen=True, slots=True)
class OneOf:
    """A prop that must be one of a few fixed words."""

    values: tuple[str, ...]

    def problem(self, value: object) -> str | None:
        if isinstance(value, str) and value in self.values:
            return None
        return f"{', '.join(self.values)} 중 하나여야 합니다."


@dataclass(frozen=True, slots=True)
class IntRange:
    """A prop that must be a whole number in ``low..high`` (inclusive)."""

    low: int
    high: int

    def problem(self, value: object) -> str | None:
        if type(value) is int and self.low <= value <= self.high:
            return None
        return f"{self.low}~{self.high} 사이의 정수여야 합니다."


@dataclass(frozen=True, slots=True)
class Flag:
    """A prop that must be true or false."""

    def problem(self, value: object) -> str | None:
        return None if isinstance(value, bool) else "true 또는 false여야 합니다."


PropRule = OneOf | IntRange | Flag

_REFERRER: Final = OneOf(("direct", "external"))
#: The props each event may carry: key → rule; every key is optional, any other key is refused.
#: Only counts are ever reported, so props are kept to a few enumerated or bounded values: an
#: analytics row can then hold no free text, hence no personal data, and stays a few bytes long.
EVENT_PROPS: Final[Mapping[str, Mapping[str, PropRule]]] = {
    "landing_view": {"referrer": _REFERRER},
    "landing_first_view": {"referrer": _REFERRER},
    "upload_click": {"files": IntRange(0, 100)},
    "signup_click": {"team_size": OneOf(TEAM_SIZES), "marketing": Flag()},
    "demo_start": {"from": OneOf(("landing", "workspace_home"))},
    "report_download": {"kind": OneOf(("md", "html", "sarif"))},
}
_MAX_SHOWN_KEY: Final = 40

PropValue = StrictStr | StrictBool | StrictInt


#: The screens an event may name in ``path`` — exactly the paths the web app sends. A free-text
#: path would let any client store names or numbers in the events table (a deep link carries a
#: workspace id, a typo'd URL anything at all), and only these are ever counted.
EVENT_SCREEN_PATHS: Final[tuple[str, ...]] = ("/", "/app", "/app/report", "/privacy")


def _props_help() -> str:
    parts = []
    for name, rules in EVENT_PROPS.items():
        keys = ", ".join(rules)
        parts.append(f"{name}: {keys}")
    return "이벤트별 허용 항목(모두 선택, 그 밖의 항목은 422) — " + " · ".join(parts)


class EventRequest(BaseModel):
    """A demand-measurement event (수요 측정). ``path`` names the screen, one of
    :data:`EVENT_SCREEN_PATHS` (a query or fragment is dropped first); ``props`` may hold only the
    keys :data:`EVENT_PROPS` allows for the event, with allowed values."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"name": "demo_start", "path": "/", "props": {"from": "landing"}}],
        }
    )

    name: EventName
    path: str | None = Field(
        default=None,
        max_length=200,
        description="이벤트가 난 화면: " + " · ".join(EVENT_SCREEN_PATHS) + " 중 하나(쿼리·# 뒤는 버림)",
    )
    props: dict[str, PropValue] = Field(default_factory=dict, description=_props_help())

    @field_validator("path")
    @classmethod
    def _path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        bare = re.split(r"[?#]", value, maxsplit=1)[0].strip()
        if not bare.startswith("/"):
            raise ValueError("path는 '/'로 시작하는 화면 경로여야 합니다.")
        screen = bare.rstrip("/") or "/"
        if screen not in EVENT_SCREEN_PATHS:
            raise ValueError(f"path는 화면 경로 {', '.join(EVENT_SCREEN_PATHS)} 중 하나여야 합니다.")
        return screen

    @field_validator("props", mode="before")
    @classmethod
    def _props(cls, value: object, info: ValidationInfo) -> object:
        """Checked against the allow-list *before* the type check, so a value of the wrong type
        (``1.5`` for ``files``) gets the rule's own message rather than one line per member of
        the str/bool/int union."""
        name = info.data.get("name")
        if not isinstance(name, str) or not isinstance(value, dict):
            return value  # a refused name, or not an object: that error is the one to report
        rules = EVENT_PROPS.get(name, {})
        clean: dict[object, object] = {}
        for key in sorted(value, key=str):
            rule = rules.get(key) if isinstance(key, str) else None
            if rule is None:
                allowed = ", ".join(rules) or "없음"
                shown = str(key)[:_MAX_SHOWN_KEY]
                raise ValueError(
                    f"'{name}' 이벤트의 props에 쓸 수 없는 항목입니다: '{shown}'(허용: {allowed})."
                )
            problem = rule.problem(value[key])
            if problem is not None:
                raise ValueError(f"'{name}' 이벤트의 props.{key}는 {problem}")
            clean[key] = value[key]
        return clean

    def stored_props(self) -> dict[str, object]:
        """What the event store keeps: the props plus the screen path (when given)."""
        out: dict[str, object] = dict(self.props)
        if self.path is not None:
            out["path"] = self.path
        return out


class Accepted(BaseModel):
    model_config = ConfigDict(frozen=True, json_schema_extra={"examples": [{"accepted": True}]})

    accepted: Literal[True] = True


class SignupDeleted(BaseModel):
    """Outcome of an administrator's deletion of one sign-up (withdrawal of consent)."""

    model_config = ConfigDict(frozen=True, json_schema_extra={"examples": [{"deleted": True}]})

    deleted: bool = Field(description="지운 신청이 있었는지(그 주소로 된 신청이 없었으면 false)")


# --------------------------------------------------------------------------- session


class SessionInfo(BaseModel):
    """Who the server thinks the caller is, so the web app shows the real role (not a guess)."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "user": "anonymous",
                    "role": "viewer",
                    "anonymous": True,
                    "auth_mode": "public",
                    "can_switch_role": False,
                }
            ]
        },
    )

    user: str
    role: Literal["viewer", "editor", "lead", "admin"]
    anonymous: bool
    auth_mode: Literal["dev", "token", "public"]
    #: only dev mode takes the role from request headers; elsewhere the token decides it
    can_switch_role: bool


# --------------------------------------------------------------------------- admin


class RequestInfo(BaseModel):
    """What the server saw of the request that asked: the peer, the proxy headers that may carry
    the visitor's address, and the per-client key ``JETTAE_CLIENT_IP_SOURCE`` derived from them.

    For the operator to check, once deployed behind a proxy, which header holds the real address.
    Header values are shown as received (several lines joined with ``", "``, cut at 512 characters).
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "peer": "10.214.3.7",
                    "x_forwarded_for": "203.0.113.9, 10.214.3.7",
                    "cf_connecting_ip": "203.0.113.9",
                    "true_client_ip": None,
                    "x_real_ip": None,
                    "client_ip_source": "header:CF-Connecting-IP",
                    "resolved_client": "203.0.113.9",
                }
            ]
        },
    )

    peer: str | None = Field(description="전송 계층의 상대 주소(uvicorn이 FORWARDED_ALLOW_IPS로 고친 값)")
    x_forwarded_for: str | None
    cf_connecting_ip: str | None
    true_client_ip: str | None
    x_real_ip: str | None
    client_ip_source: str = Field(description="JETTAE_CLIENT_IP_SOURCE (peer · header:<헤더> · xff:<n>)")
    resolved_client: str = Field(
        description="횟수 제한에 쓰는 접속 주소 키(IPv4, IPv6는 /64 대역, 읽지 못하면 invalid·unknown)"
    )


# --------------------------------------------------------------------------- health


class HealthStatus(BaseModel):
    model_config = ConfigDict(
        frozen=True, json_schema_extra={"examples": [{"status": "ok", "version": "0.1.0"}]}
    )

    status: Literal["ok"] = "ok"
    version: str


class ReadinessStatus(BaseModel):
    model_config = ConfigDict(
        frozen=True, json_schema_extra={"examples": [{"status": "ready", "detail": None}]}
    )

    status: Literal["ready", "unavailable"]
    detail: str | None = None
