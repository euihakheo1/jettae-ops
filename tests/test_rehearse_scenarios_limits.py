"""Hand-written scenario files go through the guarded YAML parser (alias bombs, deep nesting).

``scenarios.yaml`` is an upload (POST /workspaces/{ws}/inputs) parsed with every diagnosis, so it
gets the same bounds as the structured inputs (:func:`jettae.ingest.load_yaml_documents`): an
alias bomb or a deeply nested file is a Korean ``ValueError`` raised while parsing, and an
ordinary file — anchors and aliases included — parses exactly as ``yaml.safe_load`` did.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
import yaml

from jettae.check.context import CheckContext
from jettae.extract import RuleExtractor, assemble
from jettae.ingest import load_bundle_dir
from jettae.ingest.normalize import normalize_text
from jettae.ingest.structured import MAX_YAML_DEPTH, MAX_YAML_NODES
from jettae.rehearse import scenarios_from_yaml

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"

#: ~10^9 list items for under 400 bytes once every alias is expanded ("billion laughs").
ALIAS_BOMB = (
    'a: &a ["폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄"]\n'
    "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]\n"
    "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]\n"
    "d: &d [*c, *c, *c, *c, *c, *c, *c, *c, *c, *c]\n"
    "e: &e [*d, *d, *d, *d, *d, *d, *d, *d, *d, *d]\n"
    "f: &f [*e, *e, *e, *e, *e, *e, *e, *e, *e, *e]\n"
    "g: &g [*f, *f, *f, *f, *f, *f, *f, *f, *f, *f]\n"
    "h: &h [*g, *g, *g, *g, *g, *g, *g, *g, *g, *g]\n"
    "i: &i [*h, *h, *h, *h, *h, *h, *h, *h, *h, *h]\n"
)


@pytest.fixture(scope="module")
def demo_ctx() -> CheckContext:
    bundle = load_bundle_dir(DEMO_DIR)
    extractor = RuleExtractor()
    extractions = [
        extractor.extract(d, roster=bundle.roster, catalog=bundle.catalog) for d in bundle.documents
    ]
    graph = assemble(
        extractions,
        bundle.documents,
        roster=bundle.roster,
        catalog=bundle.catalog,
        oncall=bundle.oncall,
        policy=bundle.policy,
    ).graph
    return CheckContext(graph, bundle.policy, bundle.roster, bundle.oncall, bundle.catalog)


@pytest.mark.parametrize("as_bytes", [False, True])
def test_an_alias_bomb_is_rejected_while_parsing(demo_ctx: CheckContext, as_bytes: bool) -> None:
    started = time.perf_counter()
    source: str | bytes = ALIAS_BOMB.encode("utf-8") if as_bytes else ALIAS_BOMB
    with pytest.raises(ValueError, match=r"^시나리오 파일의 YAML 항목이 너무 많습니다\(") as caught:
        scenarios_from_yaml(source, demo_ctx)
    assert f"최대 {MAX_YAML_NODES:,}개" in str(caught.value)
    assert re.search(r"\(\d+행 \d+열 근처\)\.$", str(caught.value))
    assert time.perf_counter() - started < 2.0


def test_a_bomb_inside_a_scenario_is_rejected_before_any_field_is_read(demo_ctx: CheckContext) -> None:
    """The shape of a real attack: ``title`` (an error message would repr it) points at 10^9 items."""
    levels = ['&a ["폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄", "폭탄"]']
    levels += [
        f"&{name} [{', '.join([f'*{below}'] * 10)}]" for below, name in zip("abcdefg", "bcdefgh", strict=True)
    ]
    text = (
        "- id: bomb\n"
        "  service: 결제\n"
        f"  description: [{', '.join(levels)}]\n"
        f"  title: [{', '.join(['*h'] * 10)}]\n"
    )
    assert len(text.encode("utf-8")) < 1_000
    started = time.perf_counter()
    with pytest.raises(ValueError, match=r"^시나리오 파일의 YAML 항목이 너무 많습니다"):
        scenarios_from_yaml(text, demo_ctx)
    assert time.perf_counter() - started < 2.0


def test_an_alias_bomb_file_is_rejected_too(demo_ctx: CheckContext, tmp_path: Path) -> None:
    path = tmp_path / "scenarios.yaml"
    path.write_text(ALIAS_BOMB, encoding="utf-8")
    with pytest.raises(ValueError, match="YAML 항목이 너무 많습니다"):
        scenarios_from_yaml(path, demo_ctx)
    with pytest.raises(ValueError, match="YAML 항목이 너무 많습니다"):
        scenarios_from_yaml(str(path), demo_ctx)


def test_a_self_referencing_alias_and_deep_nesting_are_rejected(demo_ctx: CheckContext) -> None:
    with pytest.raises(ValueError, match="시나리오 파일의 YAML 별칭"):
        scenarios_from_yaml("- &loop [*loop]\n", demo_ctx)
    deep = "[" * (MAX_YAML_DEPTH + 5) + "]" * (MAX_YAML_DEPTH + 5) + "\n"
    with pytest.raises(
        ValueError, match=rf"시나리오 파일의 YAML 중첩이 너무 깊습니다\(최대 {MAX_YAML_DEPTH}단계\)"
    ):
        scenarios_from_yaml(deep, demo_ctx)


def test_the_demo_file_parses_exactly_as_safe_load_did(demo_ctx: CheckContext) -> None:
    from jettae.rehearse import scenarios as scenarios_module

    text = (DEMO_DIR / "scenarios.yaml").read_text(encoding="utf-8")
    assert scenarios_module._load_scenario_yaml(text) == yaml.safe_load(normalize_text(text))
    assert len(scenarios_from_yaml(DEMO_DIR / "scenarios.yaml", demo_ctx)) == 4


def test_ordinary_anchors_and_aliases_still_work(demo_ctx: CheckContext) -> None:
    with_aliases = (
        "- id: a\n"
        "  service: 결제\n"
        "  perturbations: &slack-and-cto\n"
        "    - {kind: channel_down, target: slack}\n"
        "    - {kind: person_unavailable, target: 한서윤}\n"
        "- id: b\n"
        "  service: 주문\n"
        "  perturbations: *slack-and-cto\n"
    )
    spelled_out = (
        "- id: a\n"
        "  service: 결제\n"
        "  perturbations:\n"
        "    - {kind: channel_down, target: slack}\n"
        "    - {kind: person_unavailable, target: 한서윤}\n"
        "- id: b\n"
        "  service: 주문\n"
        "  perturbations:\n"
        "    - {kind: channel_down, target: slack}\n"
        "    - {kind: person_unavailable, target: 한서윤}\n"
    )
    got = scenarios_from_yaml(with_aliases, demo_ctx)
    assert [s.id for s in got] == ["human:a", "human:b"]
    assert got == scenarios_from_yaml(spelled_out, demo_ctx)


def test_json_empty_and_comment_only_files_behave_as_before(demo_ctx: CheckContext) -> None:
    as_json = json.dumps([{"id": "j", "service": "결제"}], ensure_ascii=False)
    [scenario] = scenarios_from_yaml(as_json, demo_ctx)
    assert scenario.id == "human:j"
    assert scenarios_from_yaml("", demo_ctx) == []
    assert scenarios_from_yaml("# 아직 시나리오가 없습니다\n", demo_ctx) == []
    assert scenarios_from_yaml(b"\xef\xbb\xbf", demo_ctx) == []


def test_several_yaml_documents_are_one_korean_error(demo_ctx: CheckContext) -> None:
    text = "- id: a\n  service: 결제\n---\n- id: b\n  service: 주문\n"
    with pytest.raises(
        ValueError, match=r"YAML 문서가 하나만 있어야 합니다\('---'로 나뉜 문서가 2개입니다\)"
    ):
        scenarios_from_yaml(text, demo_ctx)


def test_a_syntax_error_names_the_line(demo_ctx: CheckContext) -> None:
    with pytest.raises(
        ValueError, match=r"^시나리오 파일의 YAML/JSON 형식이 올바르지 않습니다 \(3행 1열 근처\)\.$"
    ):
        scenarios_from_yaml("- id: x\n  service: [결제\n", demo_ctx)
