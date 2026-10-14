"""Work bounds of the loaders: small crafted uploads that used to cost minutes of CPU or hundreds of
MB of memory (alias bombs, unclosed tags, zip bombs, pages sharing one content stream) are now
rejected quickly with a Korean ``ValueError``; ordinary files load exactly as before."""

from __future__ import annotations

import io
import random
import re
import time
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import docx
import pytest
import yaml
from pypdf import PdfReader

from jettae.ingest import (
    load_bytes,
    load_catalog,
    load_document,
    load_oncall,
    load_policy,
    load_roster,
    load_yaml_documents,
    loaders,
)
from jettae.ingest.loaders import (
    _IMPLICIT_END,
    MAX_HTML_DEPTH,
    MAX_HTML_TAGS,
    MAX_PDF_PAGES,
    _html_text,
    _HtmlToMarkdown,
    _pdf_text,
)
from jettae.ingest.structured import MAX_YAML_DEPTH, MAX_YAML_NODES

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dmc-shop"
MB = 1024 * 1024


def _elapsed(action: Callable[[], object]) -> float:
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def _rejected_quickly(action: Callable[[], object], match: str, seconds: float = 2.0) -> None:
    def expect() -> None:
        with pytest.raises(ValueError, match=match):
            action()

    assert _elapsed(expect) < seconds


# --------------------------------------------------------------------------- YAML


def _alias_bomb(levels: int) -> str:
    """Ten aliases of the previous level per level: ~10^levels nodes once expanded, a few
    hundred bytes on the wire (level 7 needs several GB to walk as a tree)."""
    lines = ["a0: &a0 [x, x, x, x, x, x, x, x, x, x]"]
    for level in range(1, levels + 1):
        refs = ", ".join([f"*a{level - 1}"] * 10)
        lines.append(f"a{level}: &a{level} [{refs}]")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("load", "wrapper"),
    [
        (load_policy, ""),
        (load_catalog, "services: *a9\n"),
        (load_roster, "roster: *a9\n"),
        (load_oncall, "oncall: *a9\n"),
    ],
)
def test_alias_bomb_is_rejected_while_parsing(load: Callable[[str], Any], wrapper: str) -> None:
    source = _alias_bomb(9) + wrapper
    assert len(source) < 1000
    _rejected_quickly(lambda: load(source), "별칭 &/\\*로 펼친 항목 포함 최대", seconds=1.0)


def test_alias_bomb_in_bytes_upload_is_rejected() -> None:
    _rejected_quickly(lambda: load_policy(_alias_bomb(7).encode()), "정책 파일의 YAML 항목이 너무 많습니다")


def test_alias_into_its_own_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"YAML 별칭\(&/\*\)이 자기 자신을 가리킵니다 \(1행"):
        load_policy("critical_tiers: &a [1, *a]\n")


def test_deep_nesting_is_rejected_instead_of_hitting_the_recursion_limit() -> None:
    deep = "x: " + "[" * (MAX_YAML_DEPTH + 5) + "]" * (MAX_YAML_DEPTH + 5) + "\n"
    with pytest.raises(ValueError, match=f"YAML 중첩이 너무 깊습니다\\(최대 {MAX_YAML_DEPTH}단계\\)"):
        load_policy(deep)
    for depth in (MAX_YAML_DEPTH + 1, 100_000):  # JSON: over the bound / past json's own recursion
        with pytest.raises(ValueError, match="카탈로그 파일의 JSON 중첩이 너무 깊습니다"):
            load_catalog("[" * depth + "]" * depth)


def test_deep_aliases_count_their_full_height() -> None:
    nested = "[" * 30 + "x" + "]" * 30
    source = f"a: &a {nested}\nb: [[[[[[[[[[[[[[[[[[[[[[[[*a]]]]]]]]]]]]]]]]]]]]]]]]\n"
    with pytest.raises(ValueError, match="YAML 중첩이 너무 깊습니다"):
        load_yaml_documents(source, "정책")


def test_node_bound_is_exact_and_counts_plain_nodes_too() -> None:
    def flow(items: int) -> str:
        return "- [" + ", ".join(["1"] * items) + "]\n"

    # root sequence + inner sequence + items
    assert load_yaml_documents(flow(MAX_YAML_NODES - 2), "명부") == [[[1] * (MAX_YAML_NODES - 2)]]
    with pytest.raises(ValueError, match="명부 파일의 YAML 항목이 너무 많습니다"):
        load_yaml_documents(flow(MAX_YAML_NODES - 1), "명부")


def test_ordinary_anchors_and_merge_keys_still_work() -> None:
    compose = """
x-defaults: &defaults
  labels:
    jettae.io/tier: "1"
    jettae.io/location: 상암IDC
services:
  api:
    <<: *defaults
    depends_on: [db]
  db: *defaults
"""
    services = {svc.name: svc for svc in load_catalog(compose)}
    assert services["api"].tier == services["db"].tier == 1
    assert services["api"].location == "상암IDC"
    assert services["api"].depends_on == ("db",)
    policy = load_policy("critical_tiers: &tiers [1, 3]\nregulated_services: [결제]\n")
    assert policy.critical_tiers == (1, 3)


@pytest.mark.parametrize("name", ["policy.yaml", "catalog.yaml", "scenarios.yaml"])
def test_example_yaml_parses_exactly_like_safe_load_all(name: str) -> None:
    text = (EXAMPLE / name).read_text(encoding="utf-8")
    expected = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert load_yaml_documents(text, name) == expected


def test_example_structured_files_still_load() -> None:
    assert load_policy(EXAMPLE / "policy.yaml").critical_tiers
    assert len(load_catalog(EXAMPLE / "catalog.yaml")) > 10
    assert load_roster(EXAMPLE / "roster.csv")
    assert load_oncall(EXAMPLE / "oncall.csv")


def test_yaml_syntax_errors_keep_their_message() -> None:
    with pytest.raises(
        ValueError, match=r"정책 파일의 YAML/JSON 형식이 올바르지 않습니다 \(\d+행 \d+열 근처\)"
    ):
        load_policy("critical_tiers: [1, 2\n")


# --------------------------------------------------------------------------- HTML


def test_unclosed_tags_past_the_depth_bound_are_rejected_quickly() -> None:
    # 20,000 unclosed <b> took ~9 s with the old per-tag stack scan (quadratic)
    expected = f"bomb.html (요소가 너무 깊게 중첩되어 있습니다(최대 {MAX_HTML_DEPTH}단계)"
    _rejected_quickly(
        lambda: load_bytes("bomb.html", b"<b>" * 20_000 + "본문".encode()),
        "HTML 문서를 읽을 수 없습니다: " + re.escape(expected),
        seconds=1.0,
    )


def test_depth_bound_is_exact() -> None:
    text = "결제 서비스 담당 김대리".encode()
    assert load_bytes("ok.html", b"<b>" * MAX_HTML_DEPTH + text).text == "결제 서비스 담당 김대리"
    with pytest.raises(ValueError, match="중첩"):
        load_bytes("deep.html", b"<b>" * (MAX_HTML_DEPTH + 1) + text)


def test_tag_count_is_bounded() -> None:
    text = "본문".encode()
    assert load_bytes("ok.html", text + b"<br>" * MAX_HTML_TAGS).text == "본문"
    with pytest.raises(ValueError, match=f"태그가 너무 많습니다\\(최대 {MAX_HTML_TAGS:,}개\\)"):
        load_bytes("many.html", text + b"<br>" * (MAX_HTML_TAGS + 1))


def test_many_tags_inside_a_deep_stack_convert_in_linear_time() -> None:
    data = b"<span>" * (MAX_HTML_DEPTH - 2) + b"<b>a</b><dt>b</dt>" * 20_000
    assert _elapsed(lambda: _html_text("x.html", data)) < 3.0


def _reference_implicit_close(stack: list[str], tag: str) -> str | None:
    """The original linear scan, kept as the specification of the O(1) lookup."""
    candidates, boundaries = _IMPLICIT_END.get(tag, ((), frozenset()))
    for open_tag in reversed(stack):
        if open_tag in candidates:
            return open_tag
        if open_tag in boundaries:
            return None
    return None


def test_implicit_close_lookup_matches_the_linear_scan() -> None:
    related = {
        name for candidates, boundaries in _IMPLICIT_END.values() for name in (*candidates, *boundaries)
    }
    names = sorted({"b", "span", "div", *_IMPLICIT_END, *related})
    rng = random.Random(20261014)
    for _ in range(2000):
        parser = _HtmlToMarkdown()
        for name in rng.choices(names, k=rng.randint(0, 25)):
            parser._push(name)
        for tag in names:
            assert parser._closes_implicitly(tag) == _reference_implicit_close(parser._stack, tag)


# --------------------------------------------------------------------------- DOCX


def _base_docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("결제 서비스 장애 시 김대리에게 연락")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _with_zero_parts(base: bytes, *sizes: int) -> bytes:
    """``base`` plus all-zero media parts of the given sizes, deflated (a ~1000:1 ratio)."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(base)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info.filename))
        chunk = b"\0" * MB
        for index, size in enumerate(sizes):
            with dst.open(f"word/media/zeros{index}.bin", "w", force_zip64=True) as part:
                for start in range(0, size, MB):
                    part.write(chunk[: min(MB, size - start)])
    return out.getvalue()


def test_docx_decompression_bomb_is_rejected_before_parsing() -> None:
    bomb = _with_zero_parts(_base_docx(), 100 * MB)
    assert len(bomb) < 300 * 1024  # ~100 MB inflated from a ~140 KB upload
    _rejected_quickly(lambda: load_bytes("bomb.docx", bomb), "DOCX 파일 안의 항목 하나가 너무 큽니다", 1.0)


def test_docx_total_and_ratio_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_docx()
    with pytest.raises(
        ValueError, match=r"DOCX 파일의 압축률이 비정상적으로 높습니다: r\.docx \(\d+배, 최대 100배\)"
    ):
        load_bytes("r.docx", _with_zero_parts(base, 8 * MB))  # ~190x
    monkeypatch.setattr(loaders, "MAX_DOCX_UNCOMPRESSED_BYTES", 3 * MB)
    with pytest.raises(ValueError, match=r"DOCX 파일의 압축을 풀면 너무 큽니다: t\.docx"):
        load_bytes("t.docx", _with_zero_parts(base, 2 * MB, 2 * MB))


def test_ordinary_docx_with_a_small_compressible_part_still_loads() -> None:
    data = _with_zero_parts(_base_docx(), 512 * 1024)  # under the 1 MB ratio floor
    assert load_bytes("ok.docx", data).text == "결제 서비스 장애 시 김대리에게 연락"


@pytest.mark.parametrize("data", [b"PK\x03\x04 not a zip", b"", b"plain text"])
def test_broken_docx_is_a_korean_value_error(data: bytes) -> None:
    with pytest.raises(ValueError, match="DOCX 파일을 읽을 수 없습니다"):
        load_bytes("broken.docx", data)


# --------------------------------------------------------------------------- PDF

_LINE = b"BT /F1 12 Tf 72 720 Td (Restart the payment service and call the on-call lead) Tj ET\n"


def _flate(data: bytes, extra: bytes = b"") -> bytes:
    packed = zlib.compress(data, 9)
    return (
        b"<< /Length %d /Filter /FlateDecode %s>>\nstream\n" % (len(packed), extra) + packed + b"\nendstream"
    )


def _shared_pdf(pages: int, content: bytes, form: bytes | None = None) -> bytes:
    """``pages`` pages that all draw the *same* compressed content stream (and optional form)."""
    first_page = 5 if form is None else 6
    kids = " ".join(f"{first_page + i} 0 R" for i in range(pages))
    resources = b"/Font << /F1 3 0 R >>" + (b" /XObject << /X 5 0 R >>" if form is not None else b"")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _flate(content),
    ]
    if form is not None:
        objects.append(
            _flate(
                form,
                b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> ",
            )
        )
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << %s >> /Contents 4 0 R >>"
    objects.extend(page % resources for _ in range(pages))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _plain_pypdf(data: bytes) -> str:
    """What the loader extracted before the bounds: every page, no visitors."""
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip("\n") for page in reader.pages)


def test_pages_sharing_one_stream_are_capped_by_page_count() -> None:
    data = _shared_pdf(MAX_PDF_PAGES + 50, _LINE * 13_000)  # ~1 MB of content behind every page
    assert len(data) < 100 * 1024
    _rejected_quickly(
        lambda: load_bytes("bomb.pdf", data),
        rf"PDF 쪽수가 너무 많습니다: bomb\.pdf \(250쪽, 최대 {MAX_PDF_PAGES}쪽\)",
        1.0,
    )


def test_extracted_characters_are_capped_mid_page(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _shared_pdf(100, _LINE * 1500)  # ~85,000 characters on each of 100 pages
    expected = r"문서 텍스트가 너무 깁니다: long\.pdf \(최대 20,000자 초과\)"
    _rejected_quickly(lambda: load_bytes("long.pdf", data, max_chars=20_000), expected)
    monkeypatch.setattr(loaders, "MAX_PDF_CHARS", 20_000)  # the built-in cap applies without max_chars
    _rejected_quickly(lambda: load_bytes("long.pdf", data), expected)


def test_form_xobject_drawn_over_and_over_is_capped() -> None:
    data = _shared_pdf(1, b"/X Do\n" * 5000, form=_LINE * 50)
    _rejected_quickly(lambda: load_bytes("forms.pdf", data, max_chars=100_000), "최대 100,000자 초과")


def test_operator_budget_covers_drawing_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "MAX_PDF_OPERATIONS", 10_000)
    with pytest.raises(
        ValueError, match=r"PDF 내용이 너무 복잡합니다: ops\.pdf \(그리기 명령 최대 10,000개\)"
    ):
        load_bytes("ops.pdf", _shared_pdf(1, b"q\nQ\n" * 6000 + _LINE))
    with pytest.raises(ValueError, match="PDF 내용이 너무 복잡합니다"):  # inside a form too
        load_bytes("ops.pdf", _shared_pdf(1, b"/X Do\n" * 10, form=b"q\nQ\n" * 600 + _LINE))


def test_huge_streams_are_rejected_before_pypdf_parses_them(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _shared_pdf(1, b"q\nQ\n" * 750_000)  # 3 MB decoded from ~3 KB
    _rejected_quickly(lambda: load_bytes("p.pdf", page), "PDF의 페이지 내용 하나가 너무 큽니다", 1.0)
    form = _shared_pdf(1, b"/X Do\n", form=b"q\nQ\n" * 750_000)
    _rejected_quickly(
        lambda: load_bytes("f.pdf", form), r"PDF의 양식 객체\(form XObject\) 하나가 너무 큽니다", 1.0
    )
    monkeypatch.setattr(loaders, "MAX_PDF_CONTENT_BYTES", 100_000)  # shared streams count per page
    with pytest.raises(ValueError, match="페이지 내용 합계 최대 100,000 바이트"):
        load_bytes("s.pdf", _shared_pdf(3, _LINE * 500))


@pytest.mark.parametrize(
    "data",
    [
        _shared_pdf(1, _LINE),
        _shared_pdf(5, _LINE * 40),
        _shared_pdf(2, b"/X Do\n" * 3, form=_LINE * 3),
    ],
)
def test_bounded_extraction_matches_plain_pypdf(data: bytes) -> None:
    assert _pdf_text("x.pdf", data) == _plain_pypdf(data)


# --------------------------------------------------------------------------- max_chars


def test_max_chars_is_checked_on_the_converted_text(tmp_path: Path) -> None:
    text = "가" * 1000
    assert load_bytes("a.md", text.encode(), max_chars=1000).text == text
    assert load_bytes("a.md", text.encode()).text == text  # default: no cap
    with pytest.raises(ValueError, match=r"문서 텍스트가 너무 깁니다: b\.md \(1,001자, 최대 1,000자\)"):
        load_bytes("b.md", (text + "나").encode(), max_chars=1000)
    with pytest.raises(ValueError, match=r"문서 텍스트가 너무 깁니다: c\.html \(2,002자, 최대 1,500자\)"):
        load_bytes("c.html", f"<p>{text}</p><p>{text}</p>".encode(), max_chars=1500)
    path = tmp_path / "d.txt"
    path.write_text(text, encoding="utf-8")
    assert load_document(path, max_chars=1000).text == text
    with pytest.raises(ValueError, match="문서 텍스트가 너무 깁니다"):
        load_document(path, max_chars=999)


def test_max_chars_applies_to_pdf_and_docx() -> None:
    with pytest.raises(ValueError, match=r"문서 텍스트가 너무 깁니다: x\.pdf"):
        load_bytes("x.pdf", _shared_pdf(2, _LINE * 3), max_chars=100)
    assert load_bytes("x.pdf", _shared_pdf(1, _LINE), max_chars=1000).text
    with pytest.raises(ValueError, match=r"문서 텍스트가 너무 깁니다: x\.docx"):
        load_bytes("x.docx", _base_docx(), max_chars=10)
