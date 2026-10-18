"""The two-question coding rubric: premise and claim answered apart, the label derived from both.

Coders disagreed when one judged "is the service real?" and the other "is the gap real?" under a
single label. Each coder now answers both, the label is derived (correct = both yes, incorrect =
either no, unclear otherwise), both answers are stored, and agreement is reported on the label and
on each question.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jettae.domain import Document
from jettae.evalkit.realdata.coding import (
    FINDING_CODER_INSTRUCTIONS,
    FINDINGS_DOC_ID,
    RUBRIC_VERSION,
    Answer,
    CoderVerdict,
    FindingForCoder,
    FindingLabel,
    FindingLabelBatch,
    code_findings,
    derive_label,
)
from jettae.evalkit.realdata.diagnose import BACKTRANSLATION_SKIPPED, DiagnosisRecord
from jettae.evalkit.realdata.runbooks import (
    EvidenceRecord,
    FindingLabels,
    FindingRecord,
    RunbookResult,
    RunbookStudyConfig,
    compute_metrics,
    render_report,
)
from jettae.llm.fake import FakeLLM
from jettae.llm.schema import to_strict_json_schema


@pytest.mark.parametrize(
    ("premise", "claim", "label"),
    [
        ("yes", "yes", "correct"),
        ("yes", "no", "incorrect"),
        ("no", "yes", "incorrect"),  # a true gap about a service the document does not cover
        ("no", "no", "incorrect"),
        ("yes", "unclear", "unclear"),
        ("unclear", "yes", "unclear"),
        ("unclear", "no", "incorrect"),
        ("unclear", "unclear", "unclear"),
        (None, "yes", None),
        ("yes", None, None),
    ],
)
def test_the_label_is_derived_from_both_answers(premise: Any, claim: Any, label: Any) -> None:
    assert derive_label(premise, claim) == label


def test_the_schema_asks_both_questions_and_no_label() -> None:
    props = to_strict_json_schema(FindingLabelBatch)["properties"]["labels"]["items"]["properties"]
    assert set(props) == {"finding_ref", "premise_valid", "claim_true", "reason"}
    assert set(props["premise_valid"]["enum"]) == set(props["claim_true"]["enum"]) == {"yes", "no", "unclear"}


def test_instructions_separate_premise_from_claim_and_define_unknown() -> None:
    text = FINDING_CODER_INSTRUCTIONS
    assert "premise_valid" in text
    assert "claim_true" in text
    assert "두 질문을 섞지 마라" in text
    # a delegated element: "missing" is wrong, "cannot be checked" is right
    assert "broken(없음)이라 한 주장은 no, unknown(확인 불가)이라 한 주장은 yes" in text


def _finding(ref: str) -> FindingForCoder:
    return FindingForCoder(
        ref=ref,
        rule_id="JT-PATH-010",
        status="broken",
        severity="critical",
        title="담당자 없음",
        message="m",
        service="checkout",
        element="owner",
        evidence=((1, "q"),),
    )


def test_both_answers_are_stored_with_the_derived_label() -> None:
    def coder(task: str, schema: Any, documents: Any, context: str) -> FindingLabelBatch:
        assert [d.doc_id for d in documents][-1] == FINDINGS_DOC_ID
        return FindingLabelBatch(
            labels=[
                FindingLabel(finding_ref="F1", premise_valid="no", claim_true="yes", reason="링크 제목"),
                FindingLabel(finding_ref="F2", premise_valid="yes", claim_true="yes", reason="정말 없음"),
            ]
        )

    doc = Document(doc_id="d", title="t", text="L")
    verdicts = asyncio.run(
        code_findings(FakeLLM(coder), doc, [_finding("F1"), _finding("F2"), _finding("F3")])
    )
    assert verdicts == [
        CoderVerdict(label="incorrect", premise_valid="no", claim_true="yes", reason="링크 제목"),
        CoderVerdict(label="correct", premise_valid="yes", claim_true="yes", reason="정말 없음"),
        CoderVerdict(label=None, error="omitted"),
    ]


def _verdict(premise: Answer, claim: Answer) -> CoderVerdict:
    return CoderVerdict(label=derive_label(premise, claim), premise_valid=premise, claim_true=claim)


def _result(answers: list[tuple[tuple[Answer, Answer], tuple[Answer, Answer]]]) -> RunbookResult:
    findings = [
        FindingRecord(
            key=f"r.md::JT-PATH-010:{i}",
            finding_id=f"JT-PATH-010:{i}",
            rule_id="JT-PATH-010",
            family="PATH",
            severity="critical",
            status="broken",
            title="t",
            message="m",
            service_id="service:s",
            service_name="s",
            element="owner",
            doc_path="r.md",
            evidence=(EvidenceRecord(quote="q", line=1, source_line=1),),
        )
        for i in range(len(answers))
    ]
    diagnosis = DiagnosisRecord(
        graph_hash="h",
        extractor="hybrid",
        nodes=1,
        edges=0,
        findings=tuple(findings),
        holds=(),
        llm_calls=0,
        llm_by_task={},
        llm_input_tokens=0,
        llm_output_tokens=0,
        backtranslation=BACKTRANSLATION_SKIPPED,
        backtranslation_skipped_calls=0,
        security_notices=0,
        notes=(),
    )
    labels = tuple(
        FindingLabels(key=f.key, coder_a=_verdict(*a), coder_b=_verdict(*b))
        for f, (a, b) in zip(findings, answers, strict=True)
    )
    return RunbookResult(
        doc_path="r.md",
        doc_sha256="s",
        title="t",
        korean=False,
        source_url="https://x/r.md",
        diagnosis=diagnosis,
        labels=labels,
        usd=0.0,
    )


def test_metrics_report_agreement_per_question_and_premise_errors(tmp_path: Any) -> None:
    #           coder A (premise, claim)   coder B (premise, claim)
    answers: list[tuple[tuple[Answer, Answer], tuple[Answer, Answer]]] = [
        (("no", "yes"), ("no", "yes")),  # both: invented service -> incorrect
        (("no", "yes"), ("yes", "yes")),  # premise disagreement
        (("yes", "yes"), ("yes", "yes")),  # both correct
        (("yes", "no"), ("yes", "no")),  # both: the element is stated -> incorrect
        (("yes", "yes"), ("yes", "unclear")),
    ]
    config = RunbookStudyConfig(input_dir=tmp_path, out_dir=tmp_path / "out", generated_at="2026-10-17")
    metrics = compute_metrics(
        [_result(answers)], config=config, total=1, failed=0, stop_reason=None, spend=None, commit="c"
    )
    q = metrics.questions
    assert q is not None
    assert (q.premise_a.yes, q.premise_a.no, q.premise_b.yes, q.premise_b.no) == (3, 2, 4, 1)
    assert (q.claim_b.yes, q.claim_b.no, q.claim_b.unclear) == (3, 1, 1)
    assert q.agreement_premise.n == q.agreement_claim.n == 5
    assert q.agreement_premise.observed == 0.8
    # premise error = no / (yes + no); agreed = both no / (both no + both yes)
    assert (q.premise_error.coder_a.hits, q.premise_error.coder_a.n) == (2, 5)
    assert (q.premise_error.agreed.hits, q.premise_error.agreed.n) == (1, 4)
    # precision over a valid premise = claim yes / (claim yes + claim no) where premise is yes
    assert (q.precision_valid_premise.coder_a.hits, q.precision_valid_premise.coder_a.n) == (2, 3)
    assert (q.precision_valid_premise.coder_b.hits, q.precision_valid_premise.coder_b.n) == (2, 3)
    assert (q.precision_valid_premise.agreed.hits, q.precision_valid_premise.agreed.n) == (1, 2)
    # the derived label keeps its own agreement
    assert metrics.labels_a.incorrect == 3
    assert metrics.agreement.n == 5
    assert metrics.rubric == RUBRIC_VERSION
    [family] = metrics.families
    assert family.agreement_premise == q.agreement_premise
    report = render_report(metrics)
    assert "판정 질문별 (전제·주장)" in report
    assert "전제 오류율" in report


def test_verdicts_coded_before_the_rubric_count_as_missing_answers(tmp_path: Any) -> None:
    result = _result([(("yes", "yes"), ("yes", "yes"))])
    legacy = CoderVerdict(label="correct")
    old = result.model_copy(
        update={"labels": tuple(item.model_copy(update={"coder_a": legacy}) for item in result.labels)}
    )
    config = RunbookStudyConfig(input_dir=tmp_path, out_dir=tmp_path / "out")
    metrics = compute_metrics(
        [old], config=config, total=1, failed=0, stop_reason=None, spend=None, commit="c"
    )
    assert metrics.questions is not None
    assert metrics.questions.premise_a.missing == 1
    assert metrics.labels_a.correct == 1
