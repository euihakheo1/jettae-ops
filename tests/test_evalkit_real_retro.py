"""Study 2 (postmortem retrospective) with fake models on the fictional fixture report."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jettae.evalkit.realdata.coding import CodedGap
from jettae.evalkit.realdata.common import PIPELINE_REVISION
from jettae.evalkit.realdata.diagnose import EvidenceRecord, FindingRecord
from jettae.evalkit.realdata.dryrun import CharTokenCounter
from jettae.evalkit.realdata.entry import run_retro
from jettae.evalkit.realdata.retro import (
    JURISDICTION_RULE,
    UNKNOWN_JURISDICTION,
    GapUnitRecord,
    IncidentResult,
    ReconstructedBundle,
    RetroConfig,
    load_reports,
    locate_quote,
    match_units,
    plan_retro_calls,
    pool_gaps,
    report_jurisdiction,
    run_retro_study,
    sentence_units,
    strip_units,
    write_bundle,
)
from jettae.ingest import load_bundle_dir
from support_realdata import fake_clients, reconstructed_bundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "retro-reports"


def _config(out: Path, **overrides: object) -> RetroConfig:
    return RetroConfig(reports_dir=FIXTURE, out_dir=out, generated_at="2026-10-17", **overrides)


# --------------------------------------------------------------------------- input format


def test_fixture_report_loads_with_front_matter() -> None:
    reports = load_reports(FIXTURE)
    assert [r.incident_id for r in reports] == ["fictional-parcel-db-outage"]  # README is not a report
    report = reports[0]
    assert report.title.startswith("Fictional postmortem")
    assert report.source_url is None
    assert report.text.startswith("# Postmortem: order database outage (fictional)")
    assert "---" not in report.text.splitlines()[0]
    assert report.numbered().splitlines()[0].startswith("L1: ")


def test_reports_folder_rules(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain report", encoding="utf-8")
    (tmp_path / "a.md").write_text(
        "---\ntitle: A\nsource_url: https://example.com/a\n---\nbody\n", encoding="utf-8"
    )
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    reports = load_reports(tmp_path)
    assert [(r.incident_id, r.title, r.source_url, r.text) for r in reports] == [
        ("a", "A", "https://example.com/a", "body"),
        ("b", "b", None, "plain report"),
    ]
    assert len(load_reports(tmp_path, limit=1)) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="보고서 파일"):
        load_reports(empty)


# --------------------------------------------------------------------------- reconstruction & leakage


def test_leaked_sentences_are_stripped_deterministically() -> None:
    bundle = ReconstructedBundle.model_validate(reconstructed_bundle())
    units = sentence_units(bundle)
    texts = {u.unit_id: u.text for u in units}
    leaked = {
        uid for uid, text in texts.items() if text.startswith(("During this incident", "Writes recovered"))
    }
    assert len(leaked) == 2  # one markdown line, two sentences
    stripped = strip_units(bundle, leaked)
    markdown = stripped.runbooks[0].markdown
    assert "During this incident" not in markdown
    assert "Writes recovered" not in markdown
    assert "owned by Dana Kim" in markdown
    assert len(sentence_units(stripped)) == len(units) - 2
    assert strip_units(bundle, set()).runbooks[0].markdown.strip() == bundle.runbooks[0].markdown.strip()


def test_written_bundle_loads_with_the_product_loader(tmp_path: Path) -> None:
    bundle = ReconstructedBundle.model_validate(reconstructed_bundle())
    root = write_bundle(bundle, tmp_path / "bundle")
    loaded = load_bundle_dir(root)
    assert [d.path for d in loaded.documents] == ["runbooks/order-database.md"]
    assert {e.name for e in loaded.roster} == {"Dana Kim", "CTO"}
    assert [s.name for s in loaded.catalog] == ["order database"]  # duplicate dropped
    assert loaded.catalog[0].tier == 1
    assert loaded.policy.terminal_roles == ("CTO",)
    # writing again replaces, never accumulates
    write_bundle(bundle.model_copy(update={"roster": [], "catalog": [], "policy": None}), root)
    again = load_bundle_dir(root)
    assert again.roster == ()
    assert again.catalog == ()


# --------------------------------------------------------------------------- gap pooling & matching


REPORT = "Line one.\nThe pager was down and nobody answered.\nRollback was not documented anywhere.\n"


def _gap(quote: str, family: str, element: str | None = None) -> CodedGap:
    return CodedGap.model_validate({"quote": quote, "description": "d", "family": family, "element": element})


def test_locate_quote_is_exact_then_whitespace_insensitive() -> None:
    assert locate_quote(REPORT, "nobody answered") == (REPORT.index("nobody"), REPORT.index("nobody") + 15)
    assert locate_quote(REPORT, "L2: The pager   was down") is not None
    assert locate_quote(REPORT, "not there") is None


def test_pooling_by_overlap_and_ungrounded_gaps() -> None:
    gaps_a = [_gap("The pager was down", "PATH", "fallback_channel"), _gap("invented", "ESC")]
    gaps_b = [
        _gap("was down and nobody answered", "ESC"),
        _gap("Rollback was not documented", "PATH", "rollback"),
    ]
    units, ungrounded_a, ungrounded_b = pool_gaps(REPORT, gaps_a, gaps_b)
    assert (ungrounded_a, ungrounded_b) == (1, 0)
    assert [u.unit_id for u in units] == ["U1", "U2"]
    first, second = units
    assert (first.family_a, first.family_b) == ("PATH", "ESC")  # overlapping quotes = one unit
    assert first.quote == "The pager was down and nobody answered"
    assert (second.family_a, second.source_a, second.family_b) == (None, None, "PATH")


def _finding(family: str, element: str | None, status: str = "broken") -> FindingRecord:
    return FindingRecord(
        key=f"inc::JT-{family}-1:{element}:{status}",
        finding_id="f",
        rule_id=f"JT-{family}-001",
        family=family,
        severity="high",
        status=status,
        title="t",
        message="m",
        service_id=None,
        service_name=None,
        element=element,
        doc_path=None,
        evidence=(EvidenceRecord(quote="q", line=1, source_line=1),),
    )


def _unit(fa: str | None, ea: str | None, fb: str | None, eb: str | None) -> GapUnitRecord:
    return GapUnitRecord.model_validate(
        {
            "unit_id": "U1",
            "start": 0,
            "end": 1,
            "quote": "q",
            "family_a": fa,
            "element_a": ea,
            "family_b": fb,
            "element_b": eb,
            "source_a": "own" if fa else None,
            "source_b": "own" if fb else None,
        }
    )


def test_matching_rule() -> None:
    findings = [_finding("PATH", "fallback_channel"), _finding("ESC", None, status="unknown")]
    [agreed] = match_units([_unit("PATH", "fallback_channel", "PATH", "fallback_channel")], findings)
    assert agreed.is_gap
    assert agreed.in_scope
    assert agreed.matched
    assert agreed.matched_broken
    [wrong_element] = match_units([_unit("PATH", "approver", "PATH", "approver")], findings)
    assert wrong_element.in_scope
    assert not wrong_element.matched
    [split] = match_units([_unit("PATH", "fallback_channel", "ESC", None)], findings)
    assert split.matched
    assert not split.matched_broken
    [out] = match_units([_unit("out_of_scope", None, "PATH", "rollback")], findings)
    assert out.is_gap
    assert not out.in_scope
    assert not out.matched
    [no_gap] = match_units([_unit("not_a_gap", None, "PATH", None)], findings)
    assert not no_gap.is_gap
    [missing] = match_units([_unit(None, None, "PATH", None)], findings)
    assert not missing.is_gap


# --------------------------------------------------------------------------- the study


def test_retro_study_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "retro"
    clients, main, _coder_a, coder_b = fake_clients()
    outcome = asyncio.run(run_retro_study(_config(out), clients, echo=lambda _: None))
    assert outcome.stop_reason is None
    [result] = outcome.results
    assert isinstance(result, IncidentResult)
    assert result.leakage.units > 0
    assert {f.category for f in result.leakage.flagged} == {"incident_event"}
    runbook = (out / "bundles" / result.incident_id / "runbooks" / "order-database.md").read_text(
        encoding="utf-8"
    )
    assert "During this incident" not in runbook  # stripped before the product saw it
    assert result.diagnosis is not None
    assert result.diagnosis.findings
    assert result.ungrounded_a == 1  # coder A's invented quote is not a unit
    assert result.n == 5
    assert result.k == 4
    assert 0 <= result.m <= result.k
    assert result.m == sum(1 for u in result.units if u.matched)
    assert all(u.family_a is not None and u.family_b is not None for u in result.units)  # cross-coded
    assert {u.source_a for u in result.units} == {"own", "cross"}
    assert result.agreement.n == 5
    # coder B never sees coder A's labels: only quotes travel in the cross-coding request
    cross = [c for c in coder_b.calls if c.task == "cross_code"]
    assert cross
    assert all("PATH" not in c.documents[-1].text for c in cross)
    assert [c.task for c in main.calls][:2] == ["reconstruct", "leakage"]

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (metrics["n"], metrics["k"], metrics["m"]) == (result.n, result.k, result.m)
    assert metrics["flagged_share"]["n"] == result.k
    report = (out / "report.md").read_text(encoding="utf-8")
    for phrase in ("pre-incident flaggable", "사후 확증 편향(hindsight bias)", "N = 5", "k = 4", "Cohen's κ"):
        assert phrase in report
    leakage = json.loads((out / "bundles" / result.incident_id / "leakage.json").read_text(encoding="utf-8"))
    assert leakage["flagged"][0]["text"].startswith("During this incident")

    # resumable: a second run sends nothing
    again, main_again, _a, _b = fake_clients()
    asyncio.run(run_retro_study(_config(out), again, echo=lambda _: None))
    assert main_again.calls == []


def test_retro_plan_and_dry_run(tmp_path: Path) -> None:
    reports = load_reports(FIXTURE)
    calls = plan_retro_calls(reports, bundle_language="source")
    stages = [(c.stage, c.role, c.exact) for c in calls]
    assert ("reconstruct", "main", True) in stages
    assert ("leakage", "main", False) in stages
    assert ("code_gaps", "coder_a", True) in stages
    assert ("code_gaps", "coder_b", True) in stages
    assert ("cross_code", "coder_b", False) in stages
    assert any(s == "extract" for s, _, _ in stages)
    assert not any(s == "backtranslate" for s, _, _ in stages)  # English report, source-language bundle
    korean = plan_retro_calls(reports, bundle_language="ko")
    assert any(c.stage == "backtranslate" for c in korean)
    reconstruct = next(c for c in calls if c.stage == "reconstruct")
    assert reconstruct.documents == (reports[0].document(),)
    result = asyncio.run(
        run_retro(_config(tmp_path / "dry"), dry_run=True, echo=lambda _: None, counter=CharTokenCounter(3.0))
    )
    assert result.dry_run
    assert "실증 ② 사후 보고서 회고" in result.table.render()
    assert (tmp_path / "dry" / "dry_run.md").is_file()


# --------------------------------------------------------------------------- jurisdiction per incident

KOREAN_BODY = (
    "# 장애 보고서\n\n주문 데이터베이스가 두 시간 동안 쓰기를 받지 못했다. 담당자는 대응 절차를 따랐다.\n"
)
ENGLISH_BODY = "# Postmortem\n\nThe order database stopped accepting writes for two hours.\n"


@pytest.mark.parametrize(
    ("meta", "body", "expected"),
    [
        ({"country": "us"}, ENGLISH_BODY, ("US", "front_matter")),
        ({"jurisdiction": "KR", "country": "US"}, ENGLISH_BODY, ("KR", "front_matter")),
        ({"country": "United States"}, ENGLISH_BODY, ("ZZ", "text")),  # not a code: not trusted
        ({"language": "ko"}, ENGLISH_BODY, ("KR", "language")),
        ({"language": "en"}, KOREAN_BODY, ("ZZ", "language")),
        ({}, KOREAN_BODY, ("KR", "text")),
        ({}, ENGLISH_BODY, ("ZZ", "text")),
    ],
)
def test_jurisdiction_comes_from_the_front_matter(
    meta: dict[str, object], body: str, expected: tuple[str, str]
) -> None:
    assert report_jurisdiction(meta, body) == expected


def test_loaded_reports_carry_their_jurisdiction(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\ntitle: A\nlanguage: ko\n---\n" + KOREAN_BODY, encoding="utf-8")
    (tmp_path / "b.md").write_text("---\ntitle: B\ncountry: JP\n---\n" + ENGLISH_BODY, encoding="utf-8")
    (tmp_path / "c.txt").write_text(ENGLISH_BODY, encoding="utf-8")
    reports = load_reports(tmp_path)
    assert [(r.incident_id, r.jurisdiction, r.jurisdiction_basis) for r in reports] == [
        ("a", "KR", "language"),
        ("b", "JP", "front_matter"),
        ("c", UNKNOWN_JURISDICTION, "text"),
    ]
    assert load_reports(FIXTURE)[0].jurisdiction == UNKNOWN_JURISDICTION  # English fictional report


def test_the_bundle_policy_carries_the_incident_jurisdiction(tmp_path: Path) -> None:
    bundle = ReconstructedBundle.model_validate(reconstructed_bundle())
    root = write_bundle(bundle, tmp_path / "bundle", jurisdiction="US")
    policy = load_bundle_dir(root).policy
    assert policy.jurisdictions == ("US",)
    assert policy.terminal_roles == ("CTO",)  # what the reconstruction stated is kept
    assert not policy.obligations_in_force()  # the Korean statutory reports do not bind it
    bare = write_bundle(bundle.model_copy(update={"policy": None}), tmp_path / "bare", jurisdiction="KR")
    assert load_bundle_dir(bare).policy.jurisdictions == ("KR",)
    assert load_bundle_dir(bare).policy.obligations_in_force()


def _kr_rules(result: IncidentResult) -> set[str]:
    assert result.diagnosis is not None
    return {f.rule_id for f in result.diagnosis.findings if f.rule_id.startswith("JT-KR-")}


def test_korean_rules_bind_only_korean_incidents(tmp_path: Path) -> None:
    out = tmp_path / "retro"
    clients, *_ = fake_clients()
    outcome = asyncio.run(run_retro_study(_config(out), clients, echo=lambda _: None))
    [result] = outcome.results
    assert (result.jurisdiction, result.jurisdiction_basis) == (UNKNOWN_JURISDICTION, "text")
    assert _kr_rules(result) == set()
    policy = (out / "bundles" / result.incident_id / "policy.yaml").read_text(encoding="utf-8")
    assert "jurisdictions:" in policy
    assert UNKNOWN_JURISDICTION in policy
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["jurisdictions"] == {UNKNOWN_JURISDICTION: 1}
    assert "관할(사고별" in (out / "report.md").read_text(encoding="utf-8")

    korean = tmp_path / "reports-kr"
    korean.mkdir()
    text = (FIXTURE / "fictional-parcel-db-outage.md").read_text(encoding="utf-8")
    (korean / "kr-incident.md").write_text(text.replace("---\n", "---\ncountry: KR\n", 1), encoding="utf-8")
    kr_clients, *_ = fake_clients()
    config = RetroConfig(reports_dir=korean, out_dir=tmp_path / "retro-kr", generated_at="2026-10-17")
    [kr_result] = asyncio.run(run_retro_study(config, kr_clients, echo=lambda _: None)).results
    assert (kr_result.jurisdiction, kr_result.jurisdiction_basis) == ("KR", "front_matter")
    assert _kr_rules(kr_result)  # the same bundle under Korean jurisdiction is held to the Korean duties


def test_a_retro_folder_made_before_per_incident_jurisdiction_cannot_be_resumed(tmp_path: Path) -> None:
    out = tmp_path / "retro"
    out.mkdir()
    stored = _config(out).settings()
    assert stored["jurisdiction"] == JURISDICTION_RULE
    assert stored["pipeline"] == PIPELINE_REVISION
    del stored["jurisdiction"]
    (out / "run.json").write_text(json.dumps(stored), encoding="utf-8")
    clients, *_ = fake_clients()
    with pytest.raises(ValueError, match="jurisdiction"):
        asyncio.run(run_retro_study(_config(out), clients, echo=lambda _: None))
