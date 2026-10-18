"""Study 1 records what the product extracted, and its rubric states the element definitions.

* ``results.jsonl`` (record-v2) keeps, per runbook, the extraction IR with every quote's grounded
  line, the assembler's grounding bookkeeping and the assembled graph — so a wrong finding can be
  traced to the model's proposal, to grounding, or to the rules without paying for a re-run.
* The coding rubric (premise-claim-v4) defines detection, paging, owner, contact channel and
  mitigation the way the product's five elements do; a folder made under an earlier rubric or an
  earlier extraction prompt cannot be resumed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from jettae.domain import (
    DelegatedElement,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedDelegation,
    ExtractedEntity,
    ExtractedRelation,
    ExtractedSharedRelation,
    Extraction,
    Extractor,
    NodeKind,
)
from jettae.evalkit.realdata.coding import FINDING_CODER_INSTRUCTIONS, RUBRIC_VERSION
from jettae.evalkit.realdata.common import PIPELINE_REVISION
from jettae.evalkit.realdata.diagnose import chunk_log, extraction_log, graph_log
from jettae.evalkit.realdata.runbooks import (
    RECORD_VERSION,
    RunbookResult,
    RunbookStudyConfig,
    extraction_prompt_digest,
    run_runbook_study,
)
from jettae.extract import assemble
from jettae.ingest import load_bytes
from support_realdata import fake_clients, write_runbook_folder

RUNBOOK = """\
# Ledger exports

The ledger service writes nightly exports.
It is owned by the ledger team.
Escalation contacts live in the vault entry "Ledger escalation".
"""


def _config(folder: Path, out: Path, **overrides: Any) -> RunbookStudyConfig:
    return RunbookStudyConfig(input_dir=folder, out_dir=out, generated_at="2026-10-18", **overrides)


def _extraction(doc_id: str) -> Extraction:
    return Extraction(
        doc_id=doc_id,
        extractor=Extractor.LLM,
        entities=(
            ExtractedEntity(kind=NodeKind.SERVICE, name="ledger service", quote="ledger service"),
            ExtractedEntity(
                kind=NodeKind.TEAM,
                name="ledger team",
                quote="ledger team",
                attrs=(ExtractedAttr(key="title", value="team", quote="not in the text"),),
            ),
        ),
        relations=(
            ExtractedRelation(
                kind=EdgeKind.OWNED_BY,
                src=EntityRef(kind=NodeKind.SERVICE, name="ledger service"),
                dst=EntityRef(kind=NodeKind.TEAM, name="ledger team"),
                quote="It is owned by the ledger team.",
            ),
        ),
        delegations=(
            ExtractedDelegation(
                element=DelegatedElement.ESCALATION,
                target='vault entry "Ledger escalation"',
                quote='Escalation contacts live in the vault entry "Ledger escalation".',
            ),
        ),
        shared=(
            ExtractedSharedRelation(
                kind=EdgeKind.MITIGATED_BY,
                dst=EntityRef(kind=NodeKind.ACTION, name="rerun the export"),
                quote="Rerun the export tomorrow.",  # never written: an ungrounded proposal
            ),
        ),
    )


# --------------------------------------------------------------------------- extraction log (8)


def test_the_chunk_log_puts_every_quote_on_its_line_or_says_why_it_has_none() -> None:
    doc = load_bytes("ledger/exports.md", RUNBOOK.encode("utf-8"))
    log = chunk_log(_extraction(doc.doc_id), {doc.doc_id: doc}, {doc.doc_id: RUNBOOK})
    assert log.extractor == "llm"
    service, team = log.entities
    assert (service.name, service.quote.line, service.quote.source_line, service.quote.grounding) == (
        "ledger service",
        3,
        3,
        "exact",
    )
    assert team.attrs[0].quote.grounding == "absent"
    assert team.attrs[0].quote.line is None
    [relation] = log.relations
    assert (relation.src, relation.dst, relation.quote.line) == (
        "service:ledger service",
        "team:ledger team",
        4,
    )
    [delegation] = log.delegations
    assert (delegation.element, delegation.quote.line) == ("escalation", 5)
    [shared] = log.shared
    assert (shared.kind, shared.dst, shared.services, shared.quote.grounding) == (
        "mitigated_by",
        "action:rerun the export",
        (),
        "absent",
    )


def test_the_extraction_log_keeps_every_llm_chunk_and_only_rule_chunks_that_say_something() -> None:
    doc = load_bytes("ledger/exports.md", RUNBOOK.encode("utf-8"))
    docs = {doc.doc_id: doc}
    empty_rule = Extraction(doc_id=doc.doc_id, extractor=Extractor.RULE)
    empty_llm = Extraction(doc_id=doc.doc_id, extractor=Extractor.LLM, chunk_start=10, chunk_end=20)
    logs = extraction_log([empty_rule, empty_llm, _extraction(doc.doc_id)], docs, {})
    assert [(c.extractor, c.start) for c in logs] == [("llm", 10), ("llm", 0)]


def test_the_graph_log_lists_nodes_edges_lines_and_attributes() -> None:
    doc = load_bytes("ledger/exports.md", RUNBOOK.encode("utf-8"))
    graph = assemble([_extraction(doc.doc_id)], [doc]).graph
    log = graph_log(graph)
    nodes = {n.id: n for n in log.nodes}
    assert nodes["service:ledger-service"].lines == (3,)
    assert nodes["service:ledger-service"].attrs == {
        "delegated_escalation": 'vault entry "Ledger escalation"'
    }
    [edge] = log.edges
    assert (edge.kind, edge.src, edge.dst, edge.lines) == (
        "owned_by",
        "service:ledger-service",
        "team:ledger-team",
        (4,),
    )


def test_the_study_records_the_extraction_grounding_and_graph_of_each_runbook(tmp_path: Path) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    out = tmp_path / "out"
    clients, *_ = fake_clients()
    asyncio.run(run_runbook_study(_config(folder, out, review_size=3), clients, echo=lambda _: None))
    first = RunbookResult.model_validate_json(
        (out / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    diagnosis = first.diagnosis
    llm_chunks = [c for c in diagnosis.extraction if c.extractor == "llm"]
    assert llm_chunks, "every LLM chunk is logged"
    names = {(e.name, e.quote.line) for c in llm_chunks for e in c.entities}
    assert ("payments API", 3) in names
    assert diagnosis.grounding is not None
    assert diagnosis.grounding.grounded > 0
    assert diagnosis.graph is not None
    assert len(diagnosis.graph.nodes) == diagnosis.nodes
    assert len(diagnosis.graph.edges) == diagnosis.edges
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["extraction_items"] > 0
    assert metrics["grounding_dropped"] >= 0
    assert f"추출 기록({RECORD_VERSION})" in (out / "report.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- rubric and settings (7)


def test_the_rubric_defines_the_five_elements_like_the_product() -> None:
    assert RUBRIC_VERSION == "premise-claim-v4"
    text = FINDING_CODER_INSTRUCTIONS
    for phrase in (
        "[요소의 정의",
        "호출(페이징)이 아니어도 감지 신호다",
        "받는 사람은 그 도구의 라우팅·일정 설정이 정하므로 외부 위임과 같이 본다",
        "공급사 지원 창구",
        "공지·소통만 맡는 역할이나 공급사(벤더)는 담당자가 아니다",
        "임시 우회 방법",
        "묶음·문서 전체 진술",
        "비밀번호 관리자·금고의 항목",
        "템플릿·예시",
        "판정은 문서 전체로 한다",
        "[정책이 정한 전제",
        "주어진 것으로 본다",
        "판정하는 것은 그 전제 아래에서 문서 내용뿐이다",
    ):
        assert phrase in text


def test_settings_pin_the_rubric_the_extraction_prompt_and_the_record(tmp_path: Path) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    settings = _config(folder, tmp_path / "out").settings()
    assert settings["rubric"] == RUBRIC_VERSION
    assert settings["record"] == RECORD_VERSION == "record-v2"
    assert settings["extraction_prompt"] == extraction_prompt_digest()
    assert len(settings["extraction_prompt"]) == 16
    assert settings["pipeline"] == PIPELINE_REVISION == "pipeline-v2"


@pytest.mark.parametrize(
    ("key", "old"),
    [
        ("rubric", "premise-claim-v3"),
        ("extraction_prompt", "0" * 16),
        ("record", None),
        ("pipeline", None),
        ("pipeline", "pipeline-v1"),
    ],
)
def test_a_folder_made_under_older_settings_cannot_be_resumed(
    tmp_path: Path, key: str, old: str | None
) -> None:
    folder = write_runbook_folder(tmp_path / "input")
    out = tmp_path / "out"
    out.mkdir()
    config = _config(folder, out)
    stored = config.settings()
    if old is None:
        del stored[key]
    else:
        stored[key] = old
    (out / "run.json").write_text(json.dumps(stored), encoding="utf-8")
    clients, *_ = fake_clients()
    with pytest.raises(ValueError, match=key):
        asyncio.run(run_runbook_study(config, clients, echo=lambda _: None))
