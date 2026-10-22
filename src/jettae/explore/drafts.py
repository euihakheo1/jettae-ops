"""Unverified risk hypotheses (가설 초안) and the helpers both proposers share.

A :class:`HypothesisDraft` is what the exhaustive sweep and the AI explorer *propose*: a service,
the perturbations that are supposed to break its response, and — for the AI only — premises the
graph does not contain. It carries no status: only :func:`jettae.explore.verify.verify_hypotheses`
(the deterministic replay) turns a draft into a :class:`~jettae.domain.RiskHypothesis`.

No LLM, no I/O (import-linter keeps this module on the judgement side of the fence).
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from jettae.domain import (
    HypothesisOrigin,
    IncidentType,
    NodeKind,
    OpsGraph,
    Perturbation,
    PerturbationKind,
    RiskHypothesis,
    SourceSpan,
)

__all__ = [
    "EVIDENCE_LIMIT",
    "KIND_ORDER",
    "TARGET_KIND",
    "HypothesisDraft",
    "canonical_perturbations",
    "clean_text",
    "collect_evidence",
    "draft_from_hypothesis",
    "join_ko",
    "perturbation_label",
    "set_key",
]

#: Which node kind each perturbation kind may target (the same table the rehearsal uses).
TARGET_KIND: Final[dict[PerturbationKind, NodeKind]] = {
    PerturbationKind.PERSON_UNAVAILABLE: NodeKind.PERSON,
    PerturbationKind.CHANNEL_DOWN: NodeKind.CHANNEL,
    PerturbationKind.SYSTEM_DOWN: NodeKind.SYSTEM,
    PerturbationKind.LOCATION_DOWN: NodeKind.LOCATION,
    PerturbationKind.CONCURRENT_INCIDENT: NodeKind.SERVICE,
}

#: Display and search order of perturbation kinds: people first, then what reaches them.
KIND_ORDER: Final[dict[PerturbationKind, int]] = {
    PerturbationKind.PERSON_UNAVAILABLE: 0,
    PerturbationKind.CHANNEL_DOWN: 1,
    PerturbationKind.SYSTEM_DOWN: 2,
    PerturbationKind.LOCATION_DOWN: 3,
    PerturbationKind.CONCURRENT_INCIDENT: 4,
}

#: Source quotes kept per hypothesis (the report shows where each involved fact was read).
EVIDENCE_LIMIT: Final = 6

_SUFFIX: Final[dict[PerturbationKind, str]] = {
    PerturbationKind.PERSON_UNAVAILABLE: "부재",
    PerturbationKind.CHANNEL_DOWN: "다운",
    PerturbationKind.SYSTEM_DOWN: "장애",
    PerturbationKind.LOCATION_DOWN: "센터 장애",
    PerturbationKind.CONCURRENT_INCIDENT: "동시 장애",
}

#: Unicode format characters (bidi overrides, zero-width marks) and lone surrogates are dropped;
#: Hangul fillers render as nothing and count as blanks (same policy as AI scenario text).
_DROPPED_CATEGORIES: Final = frozenset({"Cf", "Cs"})
_HANGUL_FILLERS: Final = frozenset("ᅟᅠㅤﾠ")


def clean_text(text: str, limit: int) -> str:
    """Model or runbook text made safe to show and store: no control/format characters, NFC,
    whitespace collapsed, capped after NFC so a Hangul syllable is never split."""
    kept = "".join(
        " " if ch in _HANGUL_FILLERS or unicodedata.category(ch) == "Cc" else ch
        for ch in text
        if unicodedata.category(ch) not in _DROPPED_CATEGORIES
    )
    flat = " ".join(unicodedata.normalize("NFC", kept).split())
    return flat[:limit].rstrip()


def _bound(value: float | None) -> float:
    return math.inf if value is None else value


def canonical_perturbations(perturbations: Iterable[Perturbation]) -> tuple[Perturbation, ...]:
    """Perturbations in one total order (kind, target, window), duplicates removed."""
    unique = {(p.kind, p.target_id, p.start_minute, _bound(p.end_minute)): p for p in perturbations}
    return tuple(
        unique[key] for key in sorted(unique, key=lambda k: (KIND_ORDER[k[0]], k[0].value, k[1], k[2], k[3]))
    )


def set_key(
    service_id: str, incident_type: IncidentType, perturbations: Iterable[Perturbation]
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """What a hypothesis claims, independent of who proposed it and how it is worded."""
    targets = tuple(sorted({(p.kind.value, p.target_id) for p in perturbations}))
    return (service_id, incident_type.value, targets)


def perturbation_label(graph: OpsGraph, perturbation: Perturbation) -> str:
    """'김서연 부재', '슬랙 다운', '상암 IDC 센터 장애', '주문 동시 장애'."""
    node = graph.nodes.get(perturbation.target_id)
    name = node.name if node is not None else perturbation.target_id
    return f"{name} {_SUFFIX[perturbation.kind]}"


def join_ko(parts: Iterable[str]) -> str:
    return " + ".join(parts)


@dataclass(frozen=True)
class HypothesisDraft:
    """A proposed risk, not yet judged. ``perturbations`` are kept in canonical order."""

    origin: HypothesisOrigin
    service_id: str
    title: str
    explanation: str
    perturbations: tuple[Perturbation, ...]
    incident_type: IncidentType = IncidentType.OUTAGE
    assumptions: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return RiskHypothesis.fingerprint_of(
            self.origin, self.service_id, self.incident_type, self.perturbations, self.assumptions
        )

    @property
    def id(self) -> str:
        return f"hyp-{self.fingerprint}"

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.target_id for p in self.perturbations))

    def resolvable(self, graph: OpsGraph) -> bool:
        """Do the service and every target still exist with the kind the perturbation needs?"""
        service = graph.nodes.get(self.service_id)
        if service is None or service.kind is not NodeKind.SERVICE:
            return False
        for p in self.perturbations:
            node = graph.nodes.get(p.target_id)
            if node is None or node.kind is not TARGET_KIND[p.kind]:
                return False
        return bool(self.perturbations)


def collect_evidence(
    graph: OpsGraph, target_ids: Iterable[str], path_ids: Iterable[str]
) -> tuple[SourceSpan, ...]:
    """Where the facts a hypothesis involves were read, at most :data:`EVIDENCE_LIMIT` spans.

    First the sentences of the replayed path that touch a target ("정민혁 과장 15분 미응답 시 최영훈
    차장에게 상향"); a target no such sentence covers then adds its own first mention, channels
    last. A target's every mention would mostly repeat its name ("전화", "박재현 팀장") from
    unrelated runbooks. Facts that came from the roster, the catalog or an approved patch have no
    runbook span and add nothing here.
    """
    targets = list(dict.fromkeys(target_ids))
    wanted = set(targets)
    spans: list[SourceSpan] = []
    covered: set[str] = set()
    for ref in sorted(set(path_ids)):
        edge = graph.edges.get(ref)
        if edge is not None and edge.spans and (edge.src in wanted or edge.dst in wanted):
            spans.extend(edge.spans)
            covered |= {edge.src, edge.dst} & wanted
    nodes = [graph.nodes[nid] for nid in targets if nid in graph.nodes and nid not in covered]
    for node in sorted(nodes, key=lambda n: (n.kind is NodeKind.CHANNEL, n.id)):
        spans.extend(node.spans[:1])
    return tuple(dict.fromkeys(spans))[:EVIDENCE_LIMIT]


def draft_from_hypothesis(hypothesis: RiskHypothesis) -> HypothesisDraft:
    """Turn a verified hypothesis back into a proposal, to judge it again on another graph."""
    return HypothesisDraft(
        origin=hypothesis.origin,
        service_id=hypothesis.service_id,
        title=hypothesis.title,
        explanation=hypothesis.explanation,
        perturbations=canonical_perturbations(hypothesis.perturbations),
        incident_type=hypothesis.incident_type,
        assumptions=hypothesis.assumptions,
    )
