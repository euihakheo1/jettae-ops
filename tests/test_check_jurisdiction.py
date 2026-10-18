"""Statutory duties bind by jurisdiction: Korean reports for Korean organisations only.

The default policy is Korean (every existing bundle, the demo and the evaluation corpus keep
their findings); a policy for another jurisdiction is not told it misses a Korean 24h/72h report,
while the organisation's own duties (no jurisdiction) bind it wherever it operates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jettae.check.context import CheckContext
from jettae.check.registry import run_checks
from jettae.domain import (
    DEFAULT_OBLIGATIONS,
    KOREA,
    CatalogService,
    Extraction,
    Extractor,
    IncidentType,
    ObligationRule,
    OpsGraph,
    Policy,
    SimConfig,
)
from jettae.domain.extraction import ExtractedEntity
from jettae.domain.graph import NodeKind
from jettae.extract import assemble
from jettae.ingest import load_bytes, load_policy
from jettae.rehearse import config_hash
from jettae.rehearse.simulator import _policy_payload

TEXT = "# Checkout\nThe checkout service handles payments.\n"


def _graph() -> OpsGraph:
    doc = load_bytes("checkout.md", TEXT.encode("utf-8"))
    extraction = Extraction(
        doc_id=doc.doc_id,
        extractor=Extractor.LLM,
        entities=(ExtractedEntity(kind=NodeKind.SERVICE, name="checkout service", quote="checkout service"),),
    )
    catalog = (CatalogService(name="checkout service", tier=1),)
    return assemble([extraction], [doc], catalog=catalog).graph


def _kr_rules(policy: Policy) -> set[str]:
    ctx = CheckContext(_graph(), policy=policy)
    return {f.rule_id for f in run_checks(ctx) if f.rule_id.startswith("JT-KR-")}


def test_the_default_policy_is_korean_and_applies_korean_law() -> None:
    assert Policy().jurisdictions == (KOREA,)
    assert all(rule.jurisdiction == KOREA for rule in DEFAULT_OBLIGATIONS)
    assert _kr_rules(Policy()) == {"JT-KR-001", "JT-KR-003"}


def test_korean_law_does_not_bind_another_jurisdiction() -> None:
    assert _kr_rules(Policy(jurisdictions=("US",))) == set()
    assert _kr_rules(Policy(jurisdictions=("US", "KR"))) == {"JT-KR-001", "JT-KR-003"}


def test_an_own_duty_without_jurisdiction_binds_everywhere() -> None:
    own = ObligationRule(
        key="customer-notice",
        incident_type=IncidentType.BREACH,
        recipient="KISA",
        deadline_minutes=60,
        legal_basis="고객 계약 제7조",
    )
    policy = Policy(jurisdictions=("US",), obligations=(*DEFAULT_OBLIGATIONS, own))
    assert policy.obligations_in_force() == (own,)
    assert _kr_rules(policy) == {"JT-KR-001"}  # the rule family checks the duty the org declared


@pytest.mark.parametrize(
    ("raw", "expected"), [("us", ("US",)), ([" kr ", "US", "kr"], ("KR", "US")), (("JP",), ("JP",))]
)
def test_jurisdiction_codes_are_normalised(raw: object, expected: tuple[str, ...]) -> None:
    assert Policy.model_validate({"jurisdictions": raw}).jurisdictions == expected
    assert (
        ObligationRule.model_validate(
            {**DEFAULT_OBLIGATIONS[0].model_dump(), "jurisdiction": " kr"}
        ).jurisdiction
        == KOREA
    )


def test_a_policy_file_may_name_one_jurisdiction(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("jurisdictions: US\ncritical_tiers: [1]\n", encoding="utf-8")
    policy = load_policy(path)
    assert policy.jurisdictions == ("US",)
    assert [r.key for r in policy.obligations] == [r.key for r in DEFAULT_OBLIGATIONS]  # defaults kept
    assert policy.obligations_in_force() == ()


def test_replay_hashes_of_korean_policies_are_unchanged_and_others_differ() -> None:
    payload = _policy_payload(Policy())
    assert "jurisdictions" not in payload
    assert all("jurisdiction" not in item for item in payload["obligations"])
    config = SimConfig()
    assert config_hash(config, Policy()) == config_hash(config, Policy(jurisdictions=("kr",)))
    assert config_hash(config, Policy()) != config_hash(config, Policy(jurisdictions=("US",)))
