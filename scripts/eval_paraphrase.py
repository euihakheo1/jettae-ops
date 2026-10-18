"""Other-family re-expression robustness test on the sealed set (봉인셋 재서술 강건성 시험).

Usage (repository root, product venv; ``-X utf8`` on Windows):
    python -X utf8 scripts/eval_paraphrase.py --dry-run
    python -X utf8 scripts/eval_paraphrase.py --paraphrase --yes --max-usd 15     # PAID (OpenAI)
    python -X utf8 scripts/eval_paraphrase.py --evaluate                          # free (condition B)
    python -X utf8 scripts/eval_paraphrase.py --evaluate --llm --yes --max-usd 45 # PAID (Claude D/DR)

Why this exists: the rule parser behind condition B was tuned on a dev split of the same generator
(``docs/eval/report.md`` §15), so its sealed-set numbers may partly measure familiarity with the
generator's templates. Here a model of another family (OpenAI ``gpt-6.1-sol``; the product's LLM
extractor is Claude) rewrites each sampled case's runbook in the SAME style, keeping every fact.
A deterministic fact-preservation gate — built from the case's gold spec, never from a model —
admits only rewrites that still state every fact and still carry the injected defect; the rest are
excluded and counted by reason. B then runs on the admitted rewrites and on the originals of the
same cases (paired).

What it does NOT do: the base runbooks stay the generator's 20 synthetic organisations (not
translated public runbooks or 공공누리 manuals), and the rewrite keeps the original style instead of
producing the three styles from one neutral text. See ``docs/eval/paraphrase-plan.md``.

Steps:
* ``--dry-run`` (free): the stratified sample, the gate self-tests (identity, undone defect,
  perturbations), B on the original sample, and cost estimates. Writes into ``--out``.
* ``--paraphrase`` (PAID, needs ``--yes`` and ``--max-usd``): one rewrite per case, at most
  ``MAX_ATTEMPTS`` tries, each gated immediately; resumable (``paraphrases.jsonl``).
* ``--evaluate`` (free): B on admitted rewrites vs originals. ``--llm`` adds D/DR (PAID, Claude),
  on both versions of the admitted cases, resumable (``llm_results.jsonl``).

No model is called without ``--yes``; API keys are loaded only inside the paid paths, handed to
the SDK constructor and never printed. Products under ``src/`` are not modified (frozen ccf3b4d).
"""

from __future__ import annotations

import argparse
import difflib
import functools
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import jettae.evalkit as kit
from jettae.domain import EdgeKind, IncidentType
from jettae.evalkit.conditions import CaseResult, Condition, case_inputs, run_sync
from jettae.evalkit.corpus import EvalCase
from jettae.evalkit.gold import (
    EXEC_TEAM,
    GIVEN_NAMES,
    NICKNAMES,
    SECURITY_TEAM,
    Fact,
    GoldOrg,
    GoldService,
)
from jettae.evalkit.harness import EvalResults
from jettae.evalkit.mutate import ALL_OPERATORS, Operator
from jettae.evalkit.realdata.pricing import (
    CLAUDE_THINKING_TOKENS,
    OPENAI_REASONING_TOKENS,
    PRICES,
    PRICES_AS_OF,
    ModelRef,
    call_usd,
)
from jettae.evalkit.realdata.providers import (
    BudgetedLLM,
    BudgetExhausted,
    OpenAIResponsesLLM,
    SpendTracker,
    expected_call_usd,
    gather_jobs,
)
from jettae.evalkit.render import POSITIONS, STYLES, render
from jettae.extract.llm_extractor import (
    EXTRACTION_INSTRUCTIONS,
    LlmGraphPayload,
    build_extraction_context,
    chunk_document,
)
from jettae.llm.base import LLMDisabledError, LLMError, LLMOutputError, LLMRefusalError, UntrustedDocument
from jettae.llm.prompting import build_system, render_user_content
from jettae.llm.schema import to_strict_json_schema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "data" / "eval" / "paraphrase"
SEED, ORGS = 20261012, 20
KST = timezone(timedelta(hours=9))

PARAPHRASE_MODEL = "gpt-6.1-sol"
PARAPHRASE_TASK = "paraphrase"
PROMPT_VERSION = "paraphrase-v1"
GATE_VERSION = "gate-v1"
#: Tries per case. A retry is decided by the gate alone (never by what B does with the text).
MAX_ATTEMPTS = 2
PARAPHRASE_MAX_TOKENS = 16_000
#: A rewrite that still contains more than this share of the original's sentences/rows verbatim
#: is not a re-expression and is rejected (pre-registered, see the plan).
MAX_COPIED_SHARE = 0.5

D_MODEL = "claude-sonnet-5-5"  # the extraction model of the 2×2 pilot and real-data study ①
D_EFFORT = "medium"
LLM_ARMS = (Condition.D_LLM_GRAPH, Condition.DR_HYBRID)

#: Planning assumptions of the estimates (conservative: Korean text ≈ 1 token per character).
CHARS_PER_TOKEN_CONSERVATIVE = 1.0
CHARS_PER_TOKEN_HARNESS = 2.5  # the evalkit harness's own heuristic, shown only as a lower reference
REWRITE_LENGTH_RATIO = 1.15  # rewrite length vs original (assumption; also covers JSON escaping)

CASE_DOC_ID = "runbook.md"  # neutral: the case id names the injected operator and must not leak


# =========================================================================== corpus and sample


@dataclass(frozen=True)
class Corpus:
    cases: tuple[EvalCase, ...]
    by_id: Mapping[str, EvalCase]
    #: case id -> the gold spec the case was rendered from (the mutated org for mutated cases)
    specs: Mapping[str, GoldOrg]
    #: org id -> the clean spec
    clean_orgs: Mapping[str, GoldOrg]


def build_corpus() -> Corpus:
    orgs = kit.generate_gold_orgs(ORGS, SEED)
    cases = tuple(kit.build_cases(orgs, seed=SEED))
    mutated: dict[tuple[str, str | None], GoldOrg] = {}
    for org in orgs:
        mutated[(org.org_id, None)] = org
        for op in ALL_OPERATORS:
            mutated[(org.org_id, op.value)] = kit.apply(org, op, kit.target_service_for(org, op)).org
    specs = {c.case_id: mutated[(c.org_id, c.operator.value if c.operator else None)] for c in cases}
    return Corpus(cases, {c.case_id: c for c in cases}, specs, {o.org_id: o for o in orgs})


def stratified_sample(corpus: Corpus) -> list[dict[str, Any]]:
    """180 cases: every (org × style) cell holds one clean control and two mutated cases at the
    same position (so each mutated case has its clean twin in the sample).

    Operator of slot k in cell (org i, style s) = ALL_OPERATORS[(2i + k + 4s) mod 10]: each of the
    10 operators appears 4× per style (12× in all, 2 front + 2 back per style), each org gets 6
    distinct operators, position = POSITIONS[(i + s) mod 2] (10 front / 10 back per style).
    """
    orgs = sorted(corpus.clean_orgs)
    rows: list[dict[str, Any]] = []
    for si, style in enumerate(STYLES):
        for oi, org_id in enumerate(orgs):
            position = POSITIONS[(oi + si) % 2]
            ops: list[Operator | None] = [None, *(ALL_OPERATORS[(2 * oi + k + 4 * si) % 10] for k in (0, 1))]
            for op in ops:
                cid = kit.case_id(org_id, op, style, position)
                case = corpus.by_id[cid]
                rows.append(
                    {
                        "case_id": cid,
                        "org_id": org_id,
                        "operator": op.value if op else None,
                        "held_out": case.held_out,
                        "style": style,
                        "position": position,
                        "layout": case.layout,
                        "target_service": case.target_service,
                        "text_chars": len(case.text),
                        "text_sha256": hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
                    }
                )
    return rows


_SEAL_ADDED_LINES = ("jurisdictions:\n- KR\n", "  jurisdiction: KR\n")


def legacy_case_hash(case: EvalCase) -> str:
    """The sealed hash of a case (the freeze added two default policy lines; see sealed_set_recheck)."""
    body = case.model_dump(mode="json")
    for line in _SEAL_ADDED_LINES:
        body["policy_yaml"] = body["policy_yaml"].replace(line, "")
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =========================================================================== text analysis

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+(.*\S)\s*$")
_LABELED = re.compile(r"^\**([^:：|]{1,24}?)\**\s*[:：]\s*(.*)$")
_TIER = re.compile(r"\s*[(（]\s*(?:tier|티어|등급)\s*\d+\s*[)）]\s*", re.IGNORECASE)
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_SENTENCE = re.compile(r"(?<=[.!?。])\s+")
_POLITE_END = re.compile(r"(요|니다|죠)[.!?~]*\)?\s*$")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*)\n\s*```\s*$", re.DOTALL)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


@dataclass(frozen=True)
class Unit:
    """One statement: a table cell (with its column header and row label), a list item, or a
    prose sentence — assigned to the service whose section, row or item label it sits in."""

    service: str | None
    kind: str  # "table" | "item" | "prose"
    label: str
    content: str
    line: int

    @property
    def text(self) -> str:
        return f"{self.label} {self.content}".strip()


@dataclass(frozen=True)
class Parsed:
    headings: tuple[str, ...]
    units: tuple[Unit, ...]

    def of(self, service: str) -> list[Unit]:
        return [u for u in self.units if u.service == service]

    def region(self, service: str) -> str:
        return "\n".join(u.text for u in self.of(service))


def _service_label(label: str, services: Sequence[str]) -> str | None:
    core = _TIER.sub("", label.strip(" *`")).strip()
    core = re.sub(r"\s*서비스$", "", core)
    norm = _norm(core)
    for s in services:
        if _norm(s) == norm:
            return s
    return None


def _first_named(text: str, services_longest_first: Sequence[str]) -> str | None:
    best: tuple[int, str] | None = None
    for s in services_longest_first:
        i = text.find(s)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, s)
    return best[1] if best else None


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(text: str, services: Sequence[str]) -> Parsed:
    """Split a runbook into statements assigned to services (works on originals and rewrites)."""
    longest = sorted(services, key=len, reverse=True)
    headings: list[str] = []
    units: list[Unit] = []
    section: str | None = None
    header: list[str] | None = None
    for no, raw in enumerate(text.split("\n")):
        line = raw.strip()
        heading = _HEADING.match(raw)
        if heading:
            headings.append(line)
            section = _service_label(heading.group(2), services)
            header = None
            continue
        if line.startswith("|"):
            cells = _cells(line)
            if all(_SEPARATOR_CELL.match(c.replace(" ", "")) for c in cells if c):
                continue
            if header is None:
                header = cells
                continue
            row_service = section or _service_label(cells[0], services)
            if len(cells) == 1:
                units.append(Unit(row_service, "table", "", cells[0], no))
            for i, cell in enumerate(cells[1:], start=1):
                head = header[i] if i < len(header) else ""
                svc = row_service or _service_label(head, services)
                units.append(Unit(svc, "table", f"{head} {cells[0]}".strip(), cell, no))
            continue
        header = None
        if not line:
            continue
        item = _LIST_ITEM.match(raw)
        if item:
            body = item.group(1)
            labeled = _LABELED.match(body)
            label, content = (labeled.group(1).strip(), labeled.group(2).strip()) if labeled else ("", body)
            svc = section or _service_label(label, services) or _first_named(label or body, longest)
            units.append(Unit(svc, "item", label, content, no))
            continue
        units.extend(
            Unit(section, "prose", "", sentence.strip(), no)
            for sentence in _SENTENCE.split(line)
            if sentence.strip()
        )
    return Parsed(tuple(headings), tuple(units))


# ---- vocabulary --------------------------------------------------------------------------------

#: canonical medium value -> words that name it (case-folded search)
MEDIA_WORDS: Mapping[str, tuple[str, ...]] = {
    "slack": ("슬랙", "slack"),
    "kakaotalk": ("카카오톡", "카톡", "kakao"),
    "teams": ("팀즈", "teams"),
    "phone": ("전화", "휴대폰", "핸드폰", "phone"),
    "sms": ("문자", "sms"),
    "email": ("이메일", "메일", "email", "e-mail"),
    "pager": ("pagerduty", "페이저듀티", "페이저", "pager"),
    "internal_messenger": ("사내 메신저", "메신저", "messenger"),
}
OWN = ("담당", "맡", "책임자", "오너", "owner", "주관", "전담")
REPORT = ("신고", "kisa", "개인정보보호위원회", "개인정보 보호위원회", "인터넷진흥원", "보고")
ACTION = ("롤백", "페일오버", "되돌", "원복", "재시작", "복구", "rollback", "failover", "전환")
#: words that make an "X가 맡아요" sentence about carrying out an action, not about ownership
ACTION_CONTEXT = (*ACTION, "실행", "작업", "진행", "돌리", "돌려", "조치")
APPROVE = (
    "승인", "결재", "허가", "허락", "컨펌", "판단", "결정", "확인받", "승낙", "정하", "정합니다",
    "정해요", "정해 주", "sign-off", "approv",
)  # fmt: skip
ALARM = ("알람", "알림", "경보", "감지", "모니터링", "호출", "임계", "alert", "alarm", "페이징")
#: detection words for "the deleted detection came back" (호출 alone can be an escalation page)
DETECT = ("알람", "알림", "경보", "감지", "모니터링", "임계", "alert", "alarm")
ESC = (
    "미응답", "무응답", "응답이 없", "응답 없", "답이 없", "답 없", "못 받", "안 받", "조용",
    "상향", "에스컬레이션", "escalat", "올려", "올리", "넘겨", "넘기",
)  # fmt: skip
BREACH = ("침해", "kisa", "인터넷진흥원", "해킹")
PRIVACY = ("개인정보 유출", "개인정보유출", "개인정보보호위원회", "개인정보 보호위원회", "개인정보위")
#: role words that name an approver without a name (deleted-approver check)
BARE_APPROVER_TITLES = (
    "팀장", "cto", "ciso", "대표", "리드", "lead", "매니저", "본부장", "실장", "이사", "책임자",
    "관리자", "상급자", "상사",
)  # fmt: skip
_EMPTY_CELL = frozenset({"", "-", "—", "–", "없음", "n/a", "na", "x"})

PERSON_TITLES = (
    "사원", "주임", "대리", "과장", "차장", "팀장", "부장", "실장", "이사", "매니저", "CTO", "CISO",
    "CEO", "대표",
)  # fmt: skip
_TITLE_ALT = "|".join([*PERSON_TITLES, "님", "씨", "선배", "책임"])
COMPACT_TITLES = ("주임", "대리", "과장", "차장", "팀장", "부장", "실장")
#: common Korean surnames for the new-person detector (wider than the generator's pool, so a
#: rewrite inventing "홍길동 대리" is caught); frequent non-name syllables are left out on purpose
KO_SURNAMES = frozenset((
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황",
    "안", "송", "류", "유", "전", "고", "문", "양", "손", "배", "백", "허", "남", "심", "노", "곽",
    "성", "우", "구", "민", "진", "엄", "채", "원", "천", "방", "공", "현", "함", "변", "염", "여",
    "추", "소", "석", "선", "설", "마", "길", "연", "표", "명", "반", "왕", "금", "옥", "육", "인",
    "맹", "탁", "국", "홍", "경", "봉",
))  # fmt: skip
#: last syllables that end particles/verbs rather than names ("이어서 팀장님", "이번엔 대리")
_NOT_NAME_ENDINGS = frozenset("서은는이가을를도만엔에의로과와고면데요며한할된될줄운인던다게지께랑나야")
_GIVEN = frozenset(GIVEN_NAMES)
_NICKS = frozenset(NICKNAMES)
#: "XX님" words that are titles or generic roles, not a named person ("팀장님", "고객님", "기사님")
_HONORIFIC_WORDS = frozenset((
    "팀장", "대리", "과장", "차장", "부장", "실장", "이사", "사원", "주임", "대표", "선배", "고객",
    "담당", "당직", "기사", "회원", "작가", "셀러", "사장", "선생", "회장", "원장",
))  # fmt: skip

_NUMBER = re.compile(r"(?<![0-9A-Za-z.])(\d+(?:\.\d+)?)\s*(만|천)?\s*(분|시간|%|퍼센트|초|건|ms|일|회|배)")
_EXTRA_TOKENS = re.compile(r"(?<![0-9A-Za-z])(p\d{2,3}|\d+xx)(?![0-9A-Za-z])", re.IGNORECASE)
_HOURS = re.compile(r"(?<![0-9A-Za-z.])(\d+)\s*시간")


def has(text: str, words: Iterable[str]) -> bool:
    folded = text.casefold()
    return any(w.casefold() in folded for w in words)


def media(text: str) -> frozenset[str]:
    folded = text.casefold()
    return frozenset(m for m, words in MEDIA_WORDS.items() if any(w in folded for w in words))


def numbers(text: str) -> frozenset[str]:
    plain = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    out = {f"{n}{mag or ''}{'%' if unit == '퍼센트' else unit}" for n, mag, unit in _NUMBER.findall(plain)}
    out |= {t.lower() for t in _EXTRA_TOKENS.findall(plain)}
    return frozenset(out)


def hours(text: str) -> frozenset[int]:
    return frozenset(int(h) for h in _HOURS.findall(text))


@dataclass(frozen=True)
class People:
    """Roster reference forms of one org: full name, aliases (given name + title/님, nickname),
    and the 성+직급 form where it is unambiguous — the forms the roster resolver accepts."""

    patterns: tuple[tuple[str, re.Pattern[str]], ...]
    names: frozenset[str]
    latin_aliases: frozenset[str]
    surname_titles: Mapping[tuple[str, str], int]
    teams: frozenset[str]
    #: one alternation of every form (longest first) and matched form -> person
    combined: re.Pattern[str]
    form_owner: Mapping[str, str]
    hangul_aliases: frozenset[str]

    def _matches(self, text: str) -> Iterable[tuple[str, re.Match[str]]]:
        for m in self.combined.finditer(text):
            owner = self.form_owner.get(m.group(0))
            if owner is not None:
                yield owner, m

    def refs(self, text: str) -> frozenset[str]:
        return frozenset(name for name, _ in self._matches(text))

    def positions(self, text: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, m in self._matches(text):
            out.setdefault(name, m.start())
        return out

    def strip_refs(self, text: str, names: Iterable[str]) -> str:
        chosen = set(names)
        return self.combined.sub(
            lambda m: " " if self.form_owner.get(m.group(0)) in chosen else m.group(0), text
        )


_PEOPLE_CACHE: dict[str, People] = {}


def people_of(spec: GoldOrg) -> People:
    cached = _PEOPLE_CACHE.get(spec.org_id)
    if cached is not None:
        return cached
    counts = Counter((p.surname, p.title) for p in spec.people)
    forms: list[tuple[str, str, str]] = []  # (person, form text, regex source)
    latin: set[str] = set()
    for p in spec.people:
        forms.append((p.name, p.name, rf"(?<![가-힣]){re.escape(p.name)}"))
        for alias in p.aliases:
            if re.fullmatch(r"[A-Za-z]+", alias):
                latin.add(alias)
                forms.append((p.name, alias, rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])"))
            else:
                forms.append((p.name, alias, rf"(?<![가-힣]){re.escape(alias)}(?=\s?(?:{_TITLE_ALT}))"))
        if p.title in COMPACT_TITLES and counts[(p.surname, p.title)] == 1:
            compact = p.surname + p.title
            forms.append((p.name, compact, rf"(?<![가-힣]){re.escape(compact)}"))
    forms.sort(key=lambda f: len(f[1]), reverse=True)
    owner = {text: person for person, text, _ in forms}
    if len(owner) != len(forms):
        raise ValueError(f"{spec.org_id}: 같은 호칭이 두 사람을 가리킵니다")
    teams = {p.team for p in spec.people} | {s.team for s in spec.services} | {EXEC_TEAM, SECURITY_TEAM}
    out = People(
        tuple((person, re.compile(src)) for person, _, src in forms),
        frozenset(p.name for p in spec.people),
        frozenset(latin),
        counts,
        frozenset(teams),
        re.compile("|".join(f"(?:{src})" for _, _, src in forms)),
        owner,
        frozenset(a for p in spec.people for a in p.aliases if not re.fullmatch(r"[A-Za-z]+", a)),
    )
    _PEOPLE_CACHE[spec.org_id] = out
    return out


def new_people(text: str, spec: GoldOrg) -> list[str]:
    """Person-like references that are not on the roster (명부 외 인물). Heuristic, conservative."""
    ppl = people_of(spec)
    found: list[str] = []
    for m in re.finditer(r"(?<![A-Za-z])([A-Z][a-z]{1,11})(?![A-Za-z])", text):
        word = m.group(1)
        if word in ppl.latin_aliases:
            continue
        if word in _NICKS or re.match(r"\s?(?:님|씨)", text[m.end() : m.end() + 3]):
            found.append(word)
    for i in range(len(text) - 2):
        if text[i] in KO_SURNAMES and (i == 0 or not ("가" <= text[i - 1] <= "힣")):
            full = text[i : i + 3]
            if full[1:] in _GIVEN and full not in ppl.names:
                found.append(full)
    titled = re.compile(r"(?<![가-힣])([가-힣]{3})\s(?:사원|주임|대리|과장|차장|팀장|부장)")
    for m in titled.finditer(text):
        token = m.group(1)
        if (
            token[0] in KO_SURNAMES
            and token not in ppl.names
            and token[-1] not in _NOT_NAME_ENDINGS
            and not token.endswith(("팀", "자", "부", "실"))
        ):
            found.append(token)
    compact = re.compile(rf"(?<![가-힣])([가-힣])({'|'.join(COMPACT_TITLES)})")
    for m in compact.finditer(text):
        surname, title = m.group(1), m.group(2)
        if surname in KO_SURNAMES and ppl.surname_titles.get((surname, title), 0) != 1:
            found.append(surname + title)
    for m in re.finditer(r"(?<![가-힣])([가-힣]{2})님(?!팀)", text):
        token = m.group(1)
        if token not in _HONORIFIC_WORDS and token not in ppl.hangul_aliases:
            found.append(token + "님")
    return sorted(set(found))


def new_teams(text: str, spec: GoldOrg) -> list[str]:
    known = people_of(spec).teams
    tokens = {m.group(1) for m in re.finditer(r"(?<![가-힣A-Za-z])([가-힣A-Za-z]{1,12}팀)(?!장|원|즈)", text)}
    return sorted(t for t in tokens if t not in known)


# =========================================================================== the gate


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    copied_share: float | None = None
    similarity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": GATE_VERSION,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "copied_share": self.copied_share,
            "similarity": self.similarity,
        }


def reason_code(reason: str) -> str:
    return reason.split(":", 1)[0]


class _Ctx:
    """Keyword view of one service's statements. Service, system and person names are blanked
    out first, so a service called "알림", a system called "notify-api" or a person called
    "정하준" cannot satisfy (or trip) a keyword."""

    def __init__(self, spec: GoldOrg, units: Sequence[Unit]) -> None:
        self.spec = spec
        self.people = people_of(spec)
        self._names = sorted(
            [s.name for s in spec.services] + [x for s in spec.services for x in s.systems],
            key=len,
            reverse=True,
        )
        self._everyone = [p.name for p in spec.people]
        self.items: list[tuple[Unit, str, str, str]] = []
        for u in units:
            self.items.append((u, self._blank(u.text), self._blank(u.label), self._blank(u.content)))

    def _blank(self, text: str) -> str:
        text = self.people.strip_refs(text, self._everyone)
        for n in self._names:
            text = text.replace(n, " ")
        return _TIER.sub(" ", text)

    def refs(self, u: Unit) -> frozenset[str]:
        return self.people.refs(u.text)

    @staticmethod
    def is_approval(k: str) -> bool:
        return has(k, APPROVE) and (has(k, ACTION) or not has(k, ALARM))

    @staticmethod
    def is_action(label: str, content: str) -> bool:
        return has(content, ACTION) or (has(label, ACTION) and content.strip().casefold() not in _EMPTY_CELL)

    def owner_statements(self) -> list[Unit]:
        return [
            u
            for u, k, _, _ in self.items
            if has(k, OWN)
            and not has(k, REPORT)
            and not has(k, ACTION_CONTEXT)
            and not self.is_approval(k)
            and self.refs(u)
        ]


def _presence_problems(s: GoldService, ctx: _Ctx) -> list[str]:  # noqa: PLR0912 - one check per fact
    out: list[str] = []
    name = s.name
    region = "\n".join(u.text for u, *_ in ctx.items)
    nregion = _norm(region)
    out.extend(
        f"system_missing:{name}:{sys_name}" for sys_name in s.systems if _norm(sys_name) not in nregion
    )
    if _norm(s.location) not in nregion:
        out.append(f"location_missing:{name}")
    if s.owner is not None and not any(s.owner in ctx.refs(u) for u in ctx.owner_statements()):
        out.append(f"owner_missing:{name}")
    if s.trigger is not None:
        alarm = [u for u, k, _, _ in ctx.items if has(k, ALARM)]
        if not alarm:
            out.append(f"detection_missing:{name}")
        elif s.first_responder not in frozenset().union(*(ctx.refs(u) for u in alarm)):
            out.append(f"notify_target_missing:{name}")
    if s.action is not None:
        if not any(ctx.is_action(lab, con) for _, _, lab, con in ctx.items):
            out.append(f"action_missing:{name}")
        approval = [u for u, k, _, _ in ctx.items if ctx.is_approval(k)]
        approved = frozenset().union(*(ctx.refs(u) for u in approval)) if approval else frozenset()
        out.extend(f"approver_missing:{name}:{a}" for a in s.action.approvers if a not in approved)
        if s.action.requires_approval and not approval:
            out.append(f"approval_requirement_missing:{name}")
        if s.action.executor is not None and s.action.executor not in ctx.people.refs(region):
            out.append(f"executor_missing:{name}")
    out.extend(
        f"recipient_missing:{name}:{ob.recipient}"
        for ob in s.obligations
        if _norm(ob.recipient) not in nregion
    )
    breach_h = {o.deadline_hours for o in s.obligations if o.incident_type is IncidentType.BREACH}
    privacy_h = {o.deadline_hours for o in s.obligations if o.incident_type is IncidentType.PRIVACY_LEAK}
    for u, k, _, _ in ctx.items:
        stated = hours(u.text)
        b, p = has(k, BREACH), has(k, PRIVACY)
        if stated and (b or p):
            allowed = (breach_h if b else set()) | (privacy_h if p else set())
            if not stated <= allowed:
                out.append(f"deadline_mismatch:{name}:{sorted(stated)}")
    targets = [step.target for step in s.escalation]
    if len(targets) >= 2 and len(set(targets)) == len(targets) and s.first_responder not in targets:
        esc_text = "\n".join(u.text for u, k, _, _ in ctx.items if has(k, ESC) and not ctx.is_approval(k))
        pos = ctx.people.positions(esc_text)
        order = [pos.get(t) for t in targets]
        if any(p is None for p in order) or order != sorted(order):  # type: ignore[type-var]
            out.append(f"escalation_order_changed:{name}")
    return out


def _defect_problems(case: EvalCase, spec: GoldOrg, ctx: _Ctx) -> list[str]:  # noqa: PLR0912 - one per operator
    """The injected defect must still be in effect in the target service (nothing undone)."""
    op = case.operator
    t = case.target_service
    if op is None or t is None:
        return []
    s = spec.service(t)
    cto = spec.roles.get("CTO", ())
    out: list[str] = []
    if op is Operator.DELETE_OWNER:
        if ctx.owner_statements():
            out.append(f"defect_undone_owner:{t}")
    elif op is Operator.ESCALATION_CYCLE:
        for u, k, _, _ in ctx.items:
            if has(k, ESC) and not ctx.is_approval(k) and ctx.refs(u) & set(cto):
                out.append(f"defect_undone_cycle:{t}")
                break
    elif op is Operator.DELETE_APPROVER:
        executor = {s.action.executor} if s.action and s.action.executor else set()
        for u, k, _, _ in ctx.items:
            if not ctx.is_approval(k):
                continue
            if ctx.refs(u) - executor:
                out.append(f"defect_undone_approver:{t}")
                break
            if has(ctx.people.strip_refs(k, executor), BARE_APPROVER_TITLES):
                out.append(f"defect_undone_approver_role:{t}")
                break
    elif op is Operator.DELETE_FALLBACK_CHANNEL:
        if len(media("\n".join(u.text for u, *_ in ctx.items))) > 1:
            out.append(f"defect_undone_fallback:{t}")
    elif op is Operator.DELETE_ROLLBACK:
        for _u, _k, lab, con in ctx.items:
            if ctx.is_action(lab, con) or has(con, ("승인",)):
                out.append(f"defect_undone_action:{t}")
                break
    elif op is Operator.DELETE_ESCALATION:
        hop_people = set(cto) | {
            p
            for p in spec.roles.get("팀장", ())
            if p in ctx.people.refs("\n".join(x.text for x, *_ in ctx.items))
        }
        for u, k, _, _ in ctx.items:
            if ctx.is_approval(k):
                continue
            refs = ctx.refs(u)
            if (has(k, ESC) and refs - {s.first_responder}) or refs & hop_people:
                out.append(f"defect_undone_escalation:{t}")
                break
    elif op is Operator.DELETE_DETECTION:
        if any(has(k, DETECT) for _, k, _, _ in ctx.items):
            out.append(f"defect_undone_detection:{t}")
    elif op is Operator.DELETE_REPORTING_STEP:  # noqa: SIM102 - one branch per operator
        if any(has(k, BREACH) for _, k, _, _ in ctx.items):
            out.append(f"defect_undone_breach_step:{t}")
    # DEPARTED_CONTACT and DEADLINE_CONTRADICTION: the roster carries the departure, and the
    # 48-hour deadline is enforced by the number and deadline checks above.
    return out


def _style_problems(style: str, parsed: Parsed, services: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in services:
        kinds = Counter(u.kind for u in parsed.of(s))
        if style == "table" and not kinds["table"]:
            out.append(f"style_not_table:{s}")
        elif style == "explicit" and (kinds["table"] or not kinds["item"]):
            out.append(f"style_not_list:{s}")
        elif style == "colloquial":
            if kinds["table"] or kinds["item"] > kinds["prose"]:
                out.append(f"style_not_prose:{s}")
            elif not any(_POLITE_END.search(u.content) for u in parsed.of(s) if u.kind == "prose"):
                out.append(f"style_not_polite:{s}")
    return out


def _segments(text: str) -> list[str]:
    """Sentences, list items and table rows of the original that are long enough to count."""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _HEADING.match(raw):
            continue
        if line.startswith("|"):
            if not all(_SEPARATOR_CELL.match(c.replace(" ", "")) for c in _cells(line) if c):
                out.append(line)
            continue
        item = _LIST_ITEM.match(raw)
        out += [item.group(1)] if item else [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return [s for s in out if len(s) >= 12]


def copied_share(original: str, candidate: str) -> float:
    segments = _segments(original)
    if not segments:
        return 0.0
    flat = re.sub(r"[ \t]+", " ", candidate)
    return round(sum(re.sub(r"[ \t]+", " ", s) in flat for s in segments) / len(segments), 4)


def clean_rewrite(text: str) -> str:
    """The only normalisation applied to a rewrite: line endings and one wrapping code fence."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    fenced = _FENCE.match(text)
    return (fenced.group(1) if fenced else text).strip("\n") + "\n"


def gate(candidate: str, case: EvalCase, spec: GoldOrg, *, require_rewrite: bool = True) -> GateResult:  # noqa: PLR0912 - one check per fact kind
    """Deterministic fact-preservation gate of one rewrite against its case (see the plan).

    ``require_rewrite=False`` skips only the "is it really a re-expression" check, for the
    self-tests that feed the original text itself.
    """
    if not candidate.strip():
        return GateResult(False, ("empty",))
    services = [s.name for s in spec.services]
    orig = _parsed_original(case, tuple(services))
    cand = parse(candidate, services)
    reasons: list[str] = []
    if cand.headings != orig.headings:
        reasons.append("headings_changed")
    reasons += _style_problems(case.style, cand, services)
    ppl = people_of(spec)
    for s in spec.services:
        units = cand.of(s.name)
        if not units:
            reasons.append(f"service_missing:{s.name}")
            continue
        o_region, c_region = orig.region(s.name), cand.region(s.name)
        if media(o_region) != media(c_region):
            reasons.append(f"media_changed:{s.name}:{sorted(media(o_region))}->{sorted(media(c_region))}")
        o_num, c_num = numbers(o_region), numbers(c_region)
        if o_num != c_num:
            reasons.append(f"numbers_changed:{s.name}:-{sorted(o_num - c_num)}+{sorted(c_num - o_num)}")
        o_ppl, c_ppl = ppl.refs(o_region), ppl.refs(c_region)
        if o_ppl != c_ppl:
            reasons.append(f"people_changed:{s.name}:-{sorted(o_ppl - c_ppl)}+{sorted(c_ppl - o_ppl)}")
        ctx = _Ctx(spec, units)
        reasons += _presence_problems(s, ctx)
        if s.name == case.target_service:
            reasons += _defect_problems(case, spec, ctx)
    if added := sorted(ppl.refs(candidate) - ppl.refs(case.text)):
        reasons.append(f"roster_people_added:{added}")
    invented = sorted(set(new_people(candidate, spec)) - set(new_people(case.text, spec)))
    if invented:
        reasons.append(f"new_person:{invented}")
    teams = sorted(set(new_teams(candidate, spec)) - set(new_teams(case.text, spec)))
    if teams:
        reasons.append(f"new_team:{teams}")
    share: float | None = None
    similarity: float | None = None
    if require_rewrite:
        share = copied_share(case.text, candidate)
        similarity = round(difflib.SequenceMatcher(None, case.text, candidate).ratio(), 4)
        if share > MAX_COPIED_SHARE:
            reasons.append(f"not_rewritten:{share}")
    return GateResult(not reasons, tuple(dict.fromkeys(reasons)), share, similarity)


_ORIGINAL_PARSES: dict[tuple[str, str], Parsed] = {}


def _parsed_original(case: EvalCase, services: tuple[str, ...]) -> Parsed:
    key = (case.case_id, hashlib.sha256(case.text.encode("utf-8")).hexdigest())
    parsed = _ORIGINAL_PARSES.get(key)
    if parsed is None:
        parsed = _ORIGINAL_PARSES[key] = parse(case.text, services)
    return parsed


# =========================================================================== gate self-tests


def _undone_text(case: EvalCase, corpus: Corpus) -> str:
    """The case rendered from the CLEAN org with the same order: the defect "undone" in place."""
    clean = corpus.clean_orgs[case.org_id]
    return render(
        clean, style=case.style, position=case.position, focus_service=case.target_service, seed=SEED
    ).text


def _focus(case: EvalCase, spec: GoldOrg) -> GoldService:
    return spec.service(case.target_service or case.focus_service)


def _perturb_name(case: EvalCase, spec: GoldOrg) -> str | None:
    """Replace the focus service's first referenced roster person with an off-roster person."""
    parsed = parse(case.text, [s.name for s in spec.services])
    region = parsed.region(_focus(case, spec).name)
    ppl = people_of(spec)
    pos = ppl.positions(region)
    if not pos:
        return None
    victim = min(pos, key=pos.__getitem__)
    for name, pattern in ppl.patterns:
        if name == victim:
            m = pattern.search(case.text)
            if m:
                fake = "홍길동" if m.group(0) == victim else "길동"
                return case.text[: m.start()] + fake + case.text[m.end() :]
    return None


def _perturb_number(case: EvalCase, spec: GoldOrg) -> str | None:
    head = case.text.find(f"{_focus(case, spec).name}")
    for m in _NUMBER.finditer(case.text, max(0, head)):
        n = m.group(1)
        if "." in n:
            continue
        return case.text[: m.start(1)] + str(int(n) + 1) + case.text[m.end(1) :]
    return None


def _perturb_channel(case: EvalCase, spec: GoldOrg) -> str | None:
    s = _focus(case, spec)
    parsed = parse(case.text, [x.name for x in spec.services])
    region_media = media(parsed.region(s.name))
    extra = next((w for m, ws in MEDIA_WORDS.items() if m not in region_media for w in ws[:1]), None)
    units = parsed.of(s.name)
    if extra is None or not units:
        return None
    lines = case.text.split("\n")
    no = units[0].line
    line = lines[no]
    if line.rstrip().endswith("|"):
        lines[no] = line.rstrip()[:-1].rstrip() + f" ({extra} 병행) |"
    else:
        lines[no] = line.rstrip() + f" 급하면 {extra}로도 연락하세요."
    return "\n".join(lines)


def _perturb_drop(case: EvalCase, spec: GoldOrg) -> str | None:
    """Drop the focus service's detection statement (or, if it has none, its systems statement)."""
    s = _focus(case, spec)
    parsed = parse(case.text, [x.name for x in spec.services])
    units = parsed.of(s.name)
    pick = next((u for u in units if has(u.text, DETECT)), None) or next(
        (u for u in units if any(x in u.text for x in s.systems)), None
    )
    if pick is None:
        return None
    lines = case.text.split("\n")
    if pick.kind == "prose":
        lines[pick.line] = lines[pick.line].replace(pick.content, "").replace("  ", " ")
    else:
        del lines[pick.line]
    return "\n".join(lines)


PERTURBATIONS = {
    "off_roster_person": _perturb_name,
    "number_changed": _perturb_number,
    "channel_added": _perturb_channel,
    "statement_dropped": _perturb_drop,
}


def gate_selftest(corpus: Corpus, sample_ids: Sequence[str]) -> dict[str, Any]:
    """V1 identity on all 1,320 originals, V2 the defect undone, V3 perturbations of the sample."""
    started = time.perf_counter()
    v1_fail: list[dict[str, Any]] = []
    for case in corpus.cases:
        g = gate(case.text, case, corpus.specs[case.case_id], require_rewrite=False)
        if not g.passed:
            v1_fail.append({"case_id": case.case_id, "reasons": list(g.reasons)})
    v2_by_op: dict[str, Counter[str]] = {}
    v2_cases: dict[str, int] = Counter()
    v2_rejected: dict[str, int] = Counter()
    v2_specific: dict[str, int] = Counter()
    for case in corpus.cases:
        if case.operator is None or case.operator is Operator.DEPARTED_CONTACT:
            continue
        op = case.operator.value
        g = gate(_undone_text(case, corpus), case, corpus.specs[case.case_id], require_rewrite=False)
        v2_cases[op] += 1
        if not g.passed:
            v2_rejected[op] += 1
            codes = {reason_code(r) for r in g.reasons}
            v2_by_op.setdefault(op, Counter()).update(codes)
            if any(c.startswith("defect_undone") or c in ("numbers_changed", "media_changed") for c in codes):
                v2_specific[op] += 1
    v3: dict[str, dict[str, Any]] = {}
    for kind, fn in PERTURBATIONS.items():
        tried = rejected = 0
        reasons: Counter[str] = Counter()
        for cid in sample_ids:
            case = corpus.by_id[cid]
            spec = corpus.specs[cid]
            text = fn(case, spec)
            if text is None or text == case.text:
                continue
            tried += 1
            g = gate(text, case, spec, require_rewrite=False)
            if not g.passed:
                rejected += 1
                reasons.update({reason_code(r) for r in g.reasons})
        v3[kind] = {"tried": tried, "rejected": rejected, "reason_codes": dict(reasons.most_common())}
    return {
        "gate_version": GATE_VERSION,
        "v1_identity": {
            "what": "원문 그대로를 관문에 넣으면 모두 통과해야 한다(관문이 원문을 잘못 거르지 않는지)",
            "cases": len(corpus.cases),
            "passed": len(corpus.cases) - len(v1_fail),
            "failures": v1_fail[:50],
        },
        "v2_defect_undone": {
            "what": "같은 순서로 결함 없는 조직을 렌더한 글(=재서술이 결함을 되돌린 경우)은 걸러야 한다. "
            "담당자 퇴사는 본문이 같아 제외",
            "per_operator": {
                op: {
                    "cases": v2_cases[op],
                    "rejected": v2_rejected[op],
                    "rejected_by_defect_or_fact_check": v2_specific[op],
                    "reason_codes": dict(v2_by_op.get(op, Counter()).most_common()),
                }
                for op in sorted(v2_cases)
            },
            "cases": sum(v2_cases.values()),
            "rejected": sum(v2_rejected.values()),
        },
        "v3_perturbations": {
            "what": "표본 원문에 사실 하나를 바꾼 글(명부 밖 인물, 숫자 변경, 채널 추가, 감지 문장 삭제)은 "
            "걸러야 한다",
            **v3,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


# =========================================================================== the paraphrase request


class ParaphrasePayload(BaseModel):
    """Structured output of one rewrite: the whole rewritten runbook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(description="다시 쓴 문서 전체(마크다운)")


STYLE_RULES_KO: Mapping[str, tuple[str, str]] = {
    "table": (
        "표",
        "마크다운 표(| 로 구분한 행)로 쓴다. 열 이름, 열 순서, 행 순서, 셀 안의 문장은 바꿔도 되지만 표 "
        "형식은 "
        "유지한다. 원문에 표 밖 목록이 있으면 같은 성격의 목록으로 두고, 서비스 이름으로 시작하는 항목은 "
        "서비스 "
        "이름으로 시작하게 둔다. 한 셀이나 한 행에 서로 다른 서비스의 내용을 섞지 않는다.",
    ),
    "explicit": (
        "명시형 목록",
        "'- 항목 이름: 내용' 꼴의 목록으로 쓴다. 항목 이름, 항목 순서, 내용의 문장 표현은 바꿔도 되지만 한 "
        "줄에 "
        "한 가지 내용을 적는 목록 형식을 유지하고, 표나 긴 문단으로 바꾸지 않는다.",
    ),
    "colloquial": (
        "구어체 메모",
        "동료에게 말로 설명하듯 해요체 구어 메모로 쓴다. 표나 목록 없이 문단으로 쓰고, 말투·연결어·문장 "
        "순서를 "
        "바꾼다.",
    ),
}


def paraphrase_instructions(style: str) -> str:
    name, rules = STYLE_RULES_KO[style]
    return "\n".join(
        [
            "너는 한국어 운영 문서를 고쳐 쓰는 편집자다. untrusted_document로 주어진 장애 대응 문서를, 같은 "
            "사실을 "
            "담은 채 다른 사람이 새로 쓴 것처럼 다시 써서 text 필드에 문서 전체를 넣어라. 이 작업의 결과는 "
            "인용이 "
            "아니라 새로 쓴 문서다(안전 규칙 3은 text에 적용하지 않는다). 대신 아래 규칙을 지켜라.",
            "",
            f"문체: {name} — {rules}",
            "",
            "반드시 지킬 것:",
            "1. 사실을 빼거나 더하거나 바꾸지 마라. 사람, 직급과 호칭, 연락 수단, 숫자와 단위, "
            "서비스·시스템·위치 "
            "이름, 감지 조건, 상향 순서와 대기 시간, 조치(롤백·페일오버 등)와 승인·실행 담당, 신고 "
            "기관·시한·신고 "
            "담당이 모두 사실이다.",
            "2. 원문에 없는 사람, 팀, 연락 수단, 시간, 단계, 역할, 설명을 새로 쓰지 마라. 원문에 어떤 항목이 "
            "없으면 "
            "없는 채로 둬라. '없음', '미정', '확인 필요' 같은 말도 덧붙이지 마라.",
            "3. 사람은 원문에 적힌 호칭을 글자 그대로 써라(예: '김민수 대리', '민수 대리님', 'Kate님', "
            "'김대리님'). "
            "다른 사람의 이름이나 새 호칭을 만들지 마라. 조사와 '님'은 문장에 맞게 붙이거나 뗄 수 있다.",
            "4. 서비스·시스템·위치·연락 수단 이름과 숫자는 원문 표기 그대로 써라. 숫자를 말로 바꾸거나 "
            "단위를 바꾸지 "
            "마라(예: '48시간'을 '이틀'로, '슬랙'을 'Slack'으로, '1만 건'을 '10,000건'으로 바꾸지 않는다).",
            "5. '#' 또는 '##'로 시작하는 제목 줄은 글자 하나 바꾸지 말고 원래 순서대로 둬라. 각 제목 아래 "
            "내용은 그 "
            "제목 아래에만 쓰고, 절을 합치거나 나누거나 옮기지 마라.",
            "6. 그 범위 안에서는 표현을 적극적으로 바꿔라: 문장 구조, 어순, 연결어, 동사·명사 표현, 항목이나 "
            "문장의 "
            "순서를 바꾸고, 원문 문장을 그대로 옮겨 적지 마라.",
            "7. 머리말, 맺음말, 편집 메모, 코드 블록을 덧붙이지 마라.",
        ]
    )


def paraphrase_document(case: EvalCase) -> UntrustedDocument:
    return UntrustedDocument(doc_id=CASE_DOC_ID, title="", text=case.text)


# =========================================================================== cost estimates


def _openai_call_parts(case: EvalCase) -> tuple[int, int]:
    """(input characters, expected visible output characters) of one rewrite request."""
    system = build_system(paraphrase_instructions(case.style))
    user = render_user_content((paraphrase_document(case),))
    schema = json.dumps(OpenAIResponsesLLM.text_format(ParaphrasePayload), ensure_ascii=False)
    return len(system) + len(user) + len(schema), math.ceil(len(case.text) * REWRITE_LENGTH_RATIO)


def openai_call_usd(model: str, input_chars: int, output_chars: int, chars_per_token: float) -> float:
    reasoning = OPENAI_REASONING_TOKENS.get(model, 3_000)
    out_tokens = min(PARAPHRASE_MAX_TOKENS, math.ceil(output_chars / chars_per_token) + reasoning)
    return call_usd(model, input_tokens=math.ceil(input_chars / chars_per_token), output_tokens=out_tokens)


def _paraphrase_estimator(model: str):  # BudgetedLLM's Estimator signature
    def estimate(
        task: str, documents: tuple[UntrustedDocument, ...], extra_chars: int, max_tokens: int
    ) -> float:
        doc_chars = sum(len(d.text) for d in documents)
        system_and_schema = extra_chars + 4_000  # preamble, schema and document wrapper (upper bound)
        out_chars = math.ceil(doc_chars * REWRITE_LENGTH_RATIO)
        return openai_call_usd(model, system_and_schema + doc_chars, out_chars, CHARS_PER_TOKEN_CONSERVATIVE)

    return estimate


def _d_calls(case: EvalCase, length_ratio: float = 1.0) -> list[tuple[int, int]]:
    """(input characters, chunk characters) of every extraction call D/DR would send for ``case``."""
    inputs = case_inputs(case)
    fixed = len(build_system(EXTRACTION_INSTRUCTIONS)) + len(
        json.dumps(to_strict_json_schema(LlmGraphPayload), ensure_ascii=False)
    )
    context = len(build_extraction_context(inputs.roster, inputs.catalog))
    out: list[tuple[int, int]] = []
    for chunk in chunk_document(inputs.document):
        chars = math.ceil(len(chunk.text) * length_ratio)
        out.append((fixed + context + chars + 600, chars))  # +600: document wrapper and reminder
    return out


def d_case_usd(
    case: EvalCase,
    model: str,
    effort: str,
    length_ratio: float = 1.0,
    *,
    input_chars_per_token: float | None = None,
) -> tuple[int, float]:
    """Expected D/DR extraction cost of one case — by default exactly the budget gate's reservation
    (input 1 character = 1 token); ``input_chars_per_token`` prices the input more optimistically
    (output and thinking unchanged), only as a lower reference."""
    ref = ModelRef("anthropic", model)
    calls = _d_calls(case, length_ratio)
    if input_chars_per_token is None:
        usd = sum(
            expected_call_usd(
                ref, effort, "extract", input_chars=i, doc_chars=c, korean=True, max_tokens=16_000
            )  # type: ignore[arg-type]
            for i, c in calls
        )
    else:
        thinking = CLAUDE_THINKING_TOKENS[effort]  # type: ignore[index]
        usd = sum(
            call_usd(
                model,
                input_tokens=math.ceil(i / input_chars_per_token),
                output_tokens=min(16_000, math.ceil(c * 2.0) + thinking),
            )
            for i, c in calls
        )
    return len(calls), usd


#: The 20% 2×2 run (same corpus, same D model and effort). Read only: its measured D tokens
#: calibrate the D/DR estimate below. Missing file → no calibration, nothing else changes.
RUN1_RESULTS = ROOT.parent / "data" / "eval" / "llm-sample20" / "results.jsonl"


def d_run1_calibration(corpus: Corpus) -> dict[str, Any] | None:
    """Measured D cost of the 20% run's D rows without error vs this script's estimate for the
    same cases. The harness counts neither prompt-cache tokens nor the tokens of rejected answers,
    so the measured side is approximate (see eval_llm_resume.py)."""
    if not RUN1_RESULTS.is_file():
        return None
    text = RUN1_RESULTS.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    d_arm = Condition.D_LLM_GRAPH.value
    ok = [r for r in rows if r["condition"] == d_arm and r["error"] is None and r["case_id"] in corpus.by_id]
    if not ok:
        return None
    tokens_in = sum(r["input_tokens"] for r in ok)
    tokens_out = sum(r["output_tokens"] for r in ok)
    measured = sum(
        call_usd(D_MODEL, input_tokens=r["input_tokens"], output_tokens=r["output_tokens"]) for r in ok
    )
    estimated = sum(d_case_usd(corpus.by_id[r["case_id"]], D_MODEL, D_EFFORT)[1] for r in ok)
    return {
        "source": RUN1_RESULTS.as_posix(),
        "d_rows_without_error": len(ok),
        "measured_input_tokens_per_case": round(tokens_in / len(ok)),
        "measured_output_tokens_per_case": round(tokens_out / len(ok)),
        "measured_usd": round(measured, 2),
        "measured_usd_per_case": round(measured / len(ok), 4),
        "estimated_usd_same_cases": round(estimated, 2),
        "ratio_measured_to_estimate": round(measured / estimated, 4),
    }


def estimates(corpus: Corpus, sample_ids: Sequence[str]) -> dict[str, Any]:
    cases = [corpus.by_id[i] for i in sample_ids]
    parts = [_openai_call_parts(c) for c in cases]
    sol = sum(openai_call_usd(PARAPHRASE_MODEL, i, o, CHARS_PER_TOKEN_CONSERVATIVE) for i, o in parts)
    sol_low = sum(openai_call_usd(PARAPHRASE_MODEL, i, o, CHARS_PER_TOKEN_HARNESS) for i, o in parts)
    luna = sum(openai_call_usd("gpt-6-luna", i, o, CHARS_PER_TOKEN_CONSERVATIVE) for i, o in parts)
    d_orig = [d_case_usd(c, D_MODEL, D_EFFORT) for c in cases]
    d_para = [d_case_usd(c, D_MODEL, D_EFFORT, REWRITE_LENGTH_RATIO) for c in cases]
    d_para_low = sum(
        d_case_usd(c, D_MODEL, D_EFFORT, REWRITE_LENGTH_RATIO, input_chars_per_token=CHARS_PER_TOKEN_HARNESS)[
            1
        ]
        for c in cases
    )
    d_orig_low = sum(
        d_case_usd(c, D_MODEL, D_EFFORT, input_chars_per_token=CHARS_PER_TOKEN_HARNESS)[1] for c in cases
    )
    n = len(cases)
    calibration = d_run1_calibration(corpus)
    calibrated: dict[str, Any] = {}
    if calibration is not None:
        ratio = calibration["ratio_measured_to_estimate"]
        calibrated = {
            "run1_calibration": calibration,
            "usd_paraphrased_all_cases_run1_calibrated": round(sum(u for _, u in d_para) * ratio, 2),
            "usd_originals_all_cases_run1_calibrated": round(sum(u for _, u in d_orig) * ratio, 2),
            "usd_both_versions_paired_run1_calibrated": round(
                (sum(u for _, u in d_para) + sum(u for _, u in d_orig)) * ratio, 2
            ),
        }
    return {
        "prices_as_of": PRICES_AS_OF,
        "prices_per_mtok": {m: [p.input_per_mtok, p.output_per_mtok] for m, p in PRICES.items()},
        "assumptions": [
            f"입력·출력 토큰 = 글자 수 / {CHARS_PER_TOKEN_CONSERVATIVE:g} (한국어 1글자 ≈ 1토큰, 보수적). "
            f"참고값은 하네스 경험값 글자 {CHARS_PER_TOKEN_HARNESS:g}개 = 1토큰",
            f"재서술 출력 = 원문 글자 × {REWRITE_LENGTH_RATIO:g} + OpenAI 추론 토큰 "
            f"{OPENAI_REASONING_TOKENS.get(PARAPHRASE_MODEL, 3000):,}(가정, 출력으로 과금)",
            f"D 추출 출력 = 청크 글자 × 2 + Claude 사고 토큰 {CLAUDE_THINKING_TOKENS[D_EFFORT]:,}"
            f"(effort {D_EFFORT}); 재서술본 길이는 원문 × {REWRITE_LENGTH_RATIO:g}로 가정",
            "프롬프트 캐시 할인, 스키마 재요청(repair) 1회분은 반영하지 않음 — 캐시는 비용을 줄이고 repair는 "
            "드묾",
            "D와 DR은 사례마다 LLM 추출 1회를 공유하므로 D 비용 = D+DR 비용",
        ],
        "paraphrase": {
            "model": PARAPHRASE_MODEL,
            "cases": n,
            "calls_one_attempt": n,
            "usd_one_attempt": round(sol, 2),
            "usd_worst_case_all_retried": round(sol * MAX_ATTEMPTS, 2),
            "usd_if_30pct_retried": round(sol * 1.3, 2),
            "usd_one_attempt_at_2_5_chars_per_token": round(sol_low, 2),
            "usd_one_attempt_gpt_6_luna_alternative": round(luna, 2),
            "per_case_usd_mean": round(sol / n, 4),
        },
        "llm_d_dr": {
            "model": D_MODEL,
            "effort": D_EFFORT,
            "calls_per_version": sum(k for k, _ in d_orig),
            "usd_paraphrased_all_cases": round(sum(u for _, u in d_para), 2),
            "usd_originals_all_cases": round(sum(u for _, u in d_orig), 2),
            "usd_both_versions_paired": round(sum(u for _, u in d_para) + sum(u for _, u in d_orig), 2),
            "per_case_usd_mean_original": round(sum(u for _, u in d_orig) / n, 4),
            "usd_both_versions_input_at_2_5_chars_per_token": round(d_para_low + d_orig_low, 2),
            "usd_paraphrased_input_at_2_5_chars_per_token": round(d_para_low, 2),
            **calibrated,
            "note": "재서술 관문을 통과한 사례만 실행하므로 실제는 통과 비율만큼 줄어든다. "
            "run1_calibrated = 20% 2×2 실행에서 잰 D 토큰(캐시·거절된 답 토큰 미포함)으로 "
            "이 추정식을 보정한 값",
        },
    }


# =========================================================================== files


def _now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    tmp.replace(path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line.strip()]


def load_sample(out: Path, corpus: Corpus) -> list[str]:
    path = out / "sample.json"
    if not path.is_file():
        raise SystemExit(f"{path}가 없습니다. 먼저 --dry-run으로 표본을 만드세요.")
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [r["case_id"] for r in data["cases"]]
    for row in data["cases"]:
        case = corpus.by_id[row["case_id"]]
        if hashlib.sha256(case.text.encode("utf-8")).hexdigest() != row["text_sha256"]:
            raise SystemExit(f"표본 사례의 본문이 sample.json과 다릅니다: {row['case_id']} (코퍼스가 바뀜)")
    return ids


@dataclass
class Ledger:
    """Cumulative spend of one output folder per step (``spend.json``), so ``--max-usd`` is a cap
    over every run of that step, not per invocation."""

    path: Path
    step: str
    run_id: str = field(default_factory=lambda: datetime.now(KST).strftime("%Y%m%dT%H%M%S"))

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"runs": []}

    def prior_usd(self) -> float:
        return sum(
            r["spent_usd"]
            for r in self._load()["runs"]
            if r["step"] == self.step and r["run_id"] != self.run_id
        )

    def save(self, tracker: SpendTracker, model: str) -> None:
        data = self._load()
        report = tracker.report()
        line = {
            "run_id": self.run_id,
            "step": self.step,
            "model": model,
            "updated": _now(),
            "spent_usd": round(report.spent_usd + tracker.reserved_usd, 6),
            "by_model": [x.model_dump() for x in report.by_model],
        }
        data["runs"] = [r for r in data["runs"] if r["run_id"] != self.run_id or r["step"] != self.step] + [
            line
        ]
        data["total_usd_by_step"] = {
            step: round(sum(r["spent_usd"] for r in data["runs"] if r["step"] == step), 6)
            for step in sorted({r["step"] for r in data["runs"]})
        }
        write_json(self.path, data)


# =========================================================================== paid step 1: rewrites


class RequestFailureError(RuntimeError):
    """A request got no answer from the model (credit, key, rate limit, network). It is not a try:
    the run stops, nothing is recorded for that case, and the next run sends it again. (A wrong,
    truncated or refused answer is the model's answer and counts as a try.)"""


def _attempts_by_case(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        if r.get("prompt_version") == PROMPT_VERSION:
            out.setdefault(r["case_id"], []).append(r)
    return out


def pending_rewrites(out: Path, sample_ids: Sequence[str]) -> list[tuple[str, int]]:
    """(case id, attempts already made) of every case that still needs a try."""
    done = _attempts_by_case(read_jsonl(out / "paraphrases.jsonl"))
    todo: list[tuple[str, int]] = []
    for cid in sample_ids:
        attempts = done.get(cid, [])
        if any(a["gate"]["passed"] for a in attempts) or len(attempts) >= MAX_ATTEMPTS:
            continue
        todo.append((cid, len(attempts)))
    return todo


async def run_paraphrases(
    corpus: Corpus,
    sample_ids: Sequence[str],
    out: Path,
    *,
    model: str,
    max_usd: float,
    concurrency: int,
    sdk: Any | None = None,
) -> dict[str, Any]:
    """PAID: rewrite every pending case (OpenAI Responses API via the study adapter), gate each try."""
    path = out / "paraphrases.jsonl"
    ledger = Ledger(out / "spend.json", "paraphrase")
    tracker = SpendTracker(max_usd=max_usd, prior_usd=ledger.prior_usd())
    tracker.on_change = lambda: ledger.save(tracker, model)
    owned = sdk is None
    if sdk is None:
        from jettae.evalkit.realdata.keys import OPENAI_KEY, load_api_key
        from jettae.evalkit.realdata.providers import new_openai_sdk

        key = load_api_key(OPENAI_KEY)
        if key is None:
            raise SystemExit(
                "OPENAI_API_KEY가 없습니다(환경 변수 또는 .env). 유료 재서술을 시작할 수 없습니다."
            )
        sdk = new_openai_sdk(key)
        del key
    llm = BudgetedLLM(
        OpenAIResponsesLLM(model, client=sdk, tracker=tracker), tracker, _paraphrase_estimator(model)
    )
    todo = pending_rewrites(out, sample_ids)
    counts: Counter[str] = Counter()

    async def one(cid: str, made: int) -> None:
        case, spec = corpus.by_id[cid], corpus.specs[cid]
        for attempt in range(made + 1, MAX_ATTEMPTS + 1):
            record: dict[str, Any] = {
                "case_id": cid,
                "attempt": attempt,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "source_sha256": hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
                "at": _now(),
            }
            try:
                result = await llm.structured(
                    task=PARAPHRASE_TASK,
                    instructions=paraphrase_instructions(case.style),
                    schema=ParaphrasePayload,
                    documents=(paraphrase_document(case),),
                    max_tokens=PARAPHRASE_MAX_TOKENS,
                )
            except (BudgetExhausted, LLMDisabledError):
                raise
            except (LLMOutputError, LLMRefusalError) as exc:  # the model's answer: a try that failed
                record.update(text=None, error=f"{type(exc).__name__}: {exc}")
                record["gate"] = {"version": GATE_VERSION, "passed": False, "reasons": ["llm_error"]}
            except LLMError as exc:  # no answer (credit, key, rate limit, network): not a try
                raise RequestFailureError(f"{type(exc).__name__}: {exc}") from exc
            else:
                text = clean_rewrite(result.data.text)
                usage = result.usage
                record.update(
                    text=text,
                    error=None,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    usd=round(
                        call_usd(model, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens), 6
                    ),
                    gate=gate(text, case, spec).as_dict(),
                )
            append_jsonl(path, record)
            counts["attempts"] += 1
            if record["gate"]["passed"]:
                counts["passed"] += 1
                return
        counts["failed_all_attempts"] += 1

    stopped: str | None = None
    try:
        await gather_jobs([functools.partial(one, cid, made) for cid, made in todo], concurrency=concurrency)
    except (BudgetExhausted, RequestFailureError, LLMDisabledError) as exc:
        stopped = f"{type(exc).__name__}: {exc}"
    finally:
        ledger.save(tracker, model)
        if owned:
            await sdk.close()
    return {
        "pending_at_start": len(todo),
        **counts,
        "stopped": stopped,
        "spent_usd": tracker.report().spent_usd,
    }


# =========================================================================== evaluation


def chosen_rewrites(
    out: Path, corpus: Corpus, sample_ids: Sequence[str], *, require_rewrite: bool = True
) -> dict[str, Any]:
    """Per case: the first try that passes the gate (re-checked now), plus every try's reasons."""
    attempts = _attempts_by_case(read_jsonl(out / "paraphrases.jsonl"))
    kept: dict[str, str] = {}
    first_try_passed = 0
    tried = 0
    reasons_final: Counter[str] = Counter()
    reasons_any: Counter[str] = Counter()
    by_operator: dict[str, Counter[str]] = {}
    recheck_mismatch: list[str] = []
    per_case: dict[str, dict[str, Any]] = {}
    for cid in sample_ids:
        tries = sorted(attempts.get(cid, []), key=lambda a: a["attempt"])
        if not tries:
            continue
        tried += 1
        case, spec = corpus.by_id[cid], corpus.specs[cid]
        op = case.operator.value if case.operator else "clean"
        stats = by_operator.setdefault(op, Counter())
        stats["tried"] += 1
        info: dict[str, Any] = {"attempts": len(tries), "kept_attempt": None, "reasons": []}
        for a in tries:
            text = a.get("text")
            g = (
                gate(text, case, spec, require_rewrite=require_rewrite)
                if text
                else GateResult(False, ("llm_error",))
            )
            if bool(a["gate"]["passed"]) != g.passed:
                recheck_mismatch.append(f"{cid}#{a['attempt']}")
            codes = sorted({reason_code(r) for r in g.reasons})
            reasons_any.update(codes)
            info["reasons"].append(list(g.reasons))
            if g.passed:
                kept[cid] = text
                info["kept_attempt"] = a["attempt"]
                info["copied_share"], info["similarity"] = g.copied_share, g.similarity
                first_try_passed += a["attempt"] == 1
                stats["kept"] += 1
                break
        else:
            reasons_final.update({reason_code(r) for r in info["reasons"][-1]})
            stats["excluded"] += 1
            for code in {reason_code(r) for r in info["reasons"][-1]}:
                stats[f"reason:{code}"] += 1
        per_case[cid] = info
    return {
        "kept": kept,
        "tried": tried,
        "first_try_passed": first_try_passed,
        "excluded_reason_codes_last_try": dict(reasons_final.most_common()),
        "reason_codes_any_try": dict(reasons_any.most_common()),
        "by_operator": {k: dict(v) for k, v in sorted(by_operator.items())},
        "gate_recheck_mismatch": recheck_mismatch,
        "per_case": per_case,
    }


def _hit_table(
    cases: Sequence[EvalCase], a: EvalResults, b: EvalResults, condition: Condition
) -> dict[str, Any]:
    ra = {r.case_id: r for r in a.of(condition)}
    rb = {r.case_id: r for r in b.of(condition)}
    cells: Counter[str] = Counter()
    lost: list[str] = []
    gained: list[str] = []
    for c in cases:
        if c.is_clean:
            continue
        ha, hb = kit.is_hit(c, ra[c.case_id]), kit.is_hit(c, rb[c.case_id])
        cells[f"{'hit' if ha else 'miss'}->{'hit' if hb else 'miss'}"] += 1
        if ha and not hb:
            lost.append(c.case_id)
        if hb and not ha:
            gained.append(c.case_id)
    disc = len(lost) + len(gained)
    k = min(len(lost), len(gained))
    p = 1.0 if disc == 0 else min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2**disc)
    return {"cells": dict(cells), "lost": lost, "gained": gained, "mcnemar_exact_p": round(p, 6)}


def _subset(results: EvalResults, ids: set[str]) -> EvalResults:
    return EvalResults(
        cases=tuple(c for c in results.cases if c.case_id in ids),
        conditions=results.conditions,
        results=tuple(r for r in results.results if r.case_id in ids),
        workers=results.workers,
    )


def _brief(metrics: kit.EvalMetrics, condition: Condition) -> dict[str, Any]:
    s = metrics.conditions[condition.value]
    by_style = metrics.by_style[condition.value]
    by_op = metrics.by_operator[condition.value]
    fp = metrics.clean_fp[condition.value]
    field_scores = metrics.field_f1.get(condition.value)
    return {
        "n_cases": s.n_cases,
        "errors": s.errors,
        "detection_active": s.detection_active.model_dump(),
        "detection_held_out": s.detection_held_out.model_dump(),
        "detection_paired_active": s.detection_paired_active.model_dump(),
        "detection_paired_held_out": s.detection_paired_held_out.model_dump(),
        "clean_fp": s.clean_fp.model_dump(),
        "clean_fp_by_style": {k: v.model_dump() for k, v in fp.by_style.items()},
        "clean_fp_rules": {k: v.model_dump() for k, v in fp.per_rule.items()},
        "by_style": {
            k: {"active": v.active.model_dump(), "held_out": v.held_out.model_dump()}
            for k, v in by_style.items()
        },
        "by_operator": {k: {"hits": v.hits, "n": v.n, "rate": v.rate} for k, v in by_op.items()},
        "field": s.field.model_dump() if s.field else None,
        "field_by_style": {k: v.model_dump() for k, v in field_scores.by_style.items()}
        if field_scores
        else None,
        "hallucination": s.hallucination.model_dump() if s.hallucination else None,
        "ungrounded": s.ungrounded,
    }


def _paraphrased_cases(corpus: Corpus, kept: Mapping[str, str], sample_ids: Sequence[str]) -> list[EvalCase]:
    return [corpus.by_id[cid].model_copy(update={"text": kept[cid]}) for cid in sample_ids if cid in kept]


def _quarantined(cases: Sequence[EvalCase]) -> int:
    return sum(bool(case_inputs(c).document.quarantined) for c in cases)


def evaluate_b(
    corpus: Corpus, sample_ids: Sequence[str], kept: Mapping[str, str], workers: int
) -> dict[str, Any]:
    originals = [corpus.by_id[cid] for cid in sample_ids]
    rewrites = _paraphrased_cases(corpus, kept, sample_ids)
    b = (Condition.B_RULE_GRAPH,)
    res_orig = kit.run_eval(originals, b, workers=workers)
    res_para = kit.run_eval(rewrites, b, workers=workers) if rewrites else None
    kept_ids = set(kept)
    stamp = _now()
    orig_all = kit.compute_metrics(res_orig, generated_at=stamp)
    out: dict[str, Any] = {"B_original_full_sample": _brief(orig_all, b[0])}
    if res_para is not None:
        orig_kept = _subset(res_orig, kept_ids)
        out["B_original_kept"] = _brief(kit.compute_metrics(orig_kept, generated_at=stamp), b[0])
        out["B_paraphrased_kept"] = _brief(kit.compute_metrics(res_para, generated_at=stamp), b[0])
        out["paired_detection_all_mutated"] = _hit_table(res_para.cases, orig_kept, res_para, b[0])
        for style in STYLES:
            subset = [c for c in res_para.cases if c.style == style]
            out[f"paired_detection_{style}"] = _hit_table(subset, orig_kept, res_para, b[0])
        excluded = [c for c in originals if c.case_id not in kept_ids and not c.is_clean]
        ro = {r.case_id: r for r in res_orig.of(b[0])}
        out["B_original_on_excluded_mutated"] = {
            "n": len(excluded),
            "hits": sum(kit.is_hit(c, ro[c.case_id]) for c in excluded),
        }
        out["quarantined_documents"] = {
            "original_kept": _quarantined(orig_kept.cases),
            "paraphrased": _quarantined(rewrites),
        }
    return out


# ---- paid step 2: D / DR ------------------------------------------------------------------------


def _result_record(variant: str, r: CaseResult) -> dict[str, Any]:
    return {
        "variant": variant,
        "case_id": r.case_id,
        "condition": r.condition.value,
        "fired": sorted([list(k) for k in r.fired], key=str),
        "severe": sorted([list(k) for k in r.severe], key=str),
        "facts": None
        if r.facts is None
        else sorted([[f.kind.value, f.src, f.dst, [list(a) for a in f.attrs]] for f in r.facts]),
        "graph_hash": r.graph_hash,
        "ungrounded": r.ungrounded,
        "llm_calls": r.llm_calls,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "seconds": round(r.seconds, 3),
        "error": r.error,
        "model": D_MODEL,
        "effort": D_EFFORT,
    }


def _result_from(record: Mapping[str, Any]) -> CaseResult:
    facts = record["facts"]
    return CaseResult(
        case_id=record["case_id"],
        condition=Condition(record["condition"]),
        fired=frozenset((k[0], k[1]) for k in record["fired"]),
        severe=frozenset((k[0], k[1]) for k in record["severe"]),
        facts=None
        if facts is None
        else frozenset(
            Fact(EdgeKind(k), s, d, tuple((a[0], a[1]) for a in attrs)) for k, s, d, attrs in facts
        ),
        graph_hash=record["graph_hash"],
        ungrounded=record["ungrounded"],
        llm_calls=record["llm_calls"],
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        seconds=record["seconds"],
        error=record["error"],
    )


_STOP_ERRORS = ("BudgetExhausted", "LLMDisabledError")
#: How the harness records a request that got no answer (credit, key, rate limit, network):
#: "<exception type>: <message>" with the base LLMError type. A refused or truncated chunk is the
#: model's answer and is recorded differently ("LLM 응답이 거부된 청크 ..."), so it is kept.
_REQUEST_FAILURE_PREFIXES = ("LLMError: ",)


async def run_llm_arms(
    jobs_in: Sequence[tuple[str, EvalCase]],
    out: Path,
    *,
    max_usd: float,
    concurrency: int,
    sdk: Any | None = None,
) -> dict[str, Any]:
    """PAID: D and DR (one shared Claude extraction per case) on the given (variant, case) pairs."""
    from jettae.evalkit.realdata.clients import budgeted
    from jettae.evalkit.realdata.providers import claude_llm

    path = out / "llm_results.jsonl"
    ledger = Ledger(out / "spend.json", "llm")
    tracker = SpendTracker(max_usd=max_usd, prior_usd=ledger.prior_usd())
    tracker.on_change = lambda: ledger.save(tracker, D_MODEL)
    owned = sdk is None
    if sdk is None:
        from jettae.evalkit.realdata.keys import ANTHROPIC_KEY, load_api_key
        from jettae.evalkit.realdata.providers import new_anthropic_sdk

        key = load_api_key(ANTHROPIC_KEY)
        if key is None:
            raise SystemExit("ANTHROPIC_API_KEY가 없습니다(환경 변수 또는 .env). D/DR을 시작할 수 없습니다.")
        sdk = new_anthropic_sdk(key)
        del key
    llm = budgeted(
        claude_llm(D_MODEL, D_EFFORT, sdk, tracker), ModelRef("anthropic", D_MODEL), D_EFFORT, tracker
    )  # type: ignore[arg-type]
    done = {(r["variant"], r["case_id"]) for r in read_jsonl(path) if r.get("model") == D_MODEL}
    todo = [(v, c) for v, c in jobs_in if (v, c.case_id) not in done]

    async def one(variant: str, case: EvalCase) -> None:
        results = await kit.arun_llm_graphs(case, llm, list(LLM_ARMS))
        for r in results:
            if r.error and r.error.startswith(_STOP_ERRORS):
                raise (
                    BudgetExhausted(r.error)
                    if r.error.startswith("BudgetExhausted")
                    else LLMDisabledError(r.error)
                )
            if r.error and r.error.startswith(_REQUEST_FAILURE_PREFIXES):
                # no answer from the model: not recorded, so the next run sends this case again
                raise RequestFailureError(r.error)
        for r in results:
            append_jsonl(path, _result_record(variant, r))

    stopped: str | None = None
    try:
        await gather_jobs([functools.partial(one, v, c) for v, c in todo], concurrency=concurrency)
    except (BudgetExhausted, RequestFailureError, LLMDisabledError) as exc:
        stopped = f"{type(exc).__name__}: {exc}"
    finally:
        ledger.save(tracker, D_MODEL)
        if owned:
            await sdk.close()
    return {"pending_at_start": len(todo), "stopped": stopped, "spent_usd": tracker.report().spent_usd}


def evaluate_llm(out: Path, originals: Sequence[EvalCase], rewrites: Sequence[EvalCase]) -> dict[str, Any]:
    records = [r for r in read_jsonl(out / "llm_results.jsonl") if r.get("model") == D_MODEL]
    out_metrics: dict[str, Any] = {}
    stamp = _now()
    by_variant: dict[str, EvalResults] = {}
    for variant, cases in (("original", originals), ("paraphrase", rewrites)):
        ids = {c.case_id for c in cases}
        results = [_result_from(r) for r in records if r["variant"] == variant and r["case_id"] in ids]
        have = {r.case_id for r in results}
        chosen = tuple(c for c in cases if c.case_id in have)
        if not chosen:
            continue
        rank = {c: i for i, c in enumerate(LLM_ARMS)}
        pos = {c.case_id: i for i, c in enumerate(chosen)}
        results.sort(key=lambda r: (pos[r.case_id], rank[r.condition]))
        res = EvalResults(cases=chosen, conditions=LLM_ARMS, results=tuple(results))
        by_variant[variant] = res
        metrics = kit.compute_metrics(res, generated_at=stamp)
        for arm in LLM_ARMS:
            out_metrics[f"{arm.value}_{variant}"] = _brief(metrics, arm)
    if len(by_variant) == 2:
        both = {c.case_id for c in by_variant["original"].cases} & {
            c.case_id for c in by_variant["paraphrase"].cases
        }
        a, b = _subset(by_variant["original"], both), _subset(by_variant["paraphrase"], both)
        for arm in LLM_ARMS:
            out_metrics[f"paired_{arm.value}"] = _hit_table(b.cases, a, b, arm)
    return out_metrics


# =========================================================================== reporting


def _pct(rate: Mapping[str, Any] | None) -> str:
    if not rate or rate.get("n", 0) == 0 or rate.get("rate") is None:
        return "—"
    lo, hi = rate["ci95"]
    return f"{rate['rate'] * 100:.1f}% ({rate['hits']}/{rate['n']}, 95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"


#: Pre-registered (docs/eval/paraphrase-plan.md, 해석 규칙 4): below this final gate pass rate the
#: test is reported as insufficient and its numbers are reference values only.
MIN_GATE_PASS_RATE = 0.5


def _gate_lines(summary: Mapping[str, Any], g: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """What the gate left out, stated next to the numbers: the pre-registered pass-rate rule, the
    cases not tried yet, the pass rate per operator, and B on the excluded originals."""
    tried, kept, sample = g["tried"], len(g["kept"]), summary["sample_size"]
    lines: list[str] = []
    if tried:
        rate = kept / tried
        verdict = (
            "사전 등록 기준(50% 이상) 충족"
            if rate >= MIN_GATE_PASS_RATE
            else "**사전 등록 기준 50% 미만 → '재서술 시험 불충분'. 아래 수치는 참고값으로만 쓴다.**"
        )
        lines.append(f"- 최종 관문 통과율 {kept}/{tried} = {rate * 100:.1f}%: {verdict}")
    if tried < sample:
        lines.append(
            f"- 재서술을 아직 시도하지 않은 사례 {sample - tried}건(표본 {sample}건). 중단된 실행이면 "
            "--paraphrase를 다시 실행한다."
        )
    cells = []
    for op, st in g.get("by_operator", {}).items():
        undone = sum(v for k, v in st.items() if k.startswith("reason:defect_undone"))
        cells.append(
            f"{op} {st.get('kept', 0)}/{st.get('tried', 0)}" + (f"(결함 되돌림 {undone})" if undone else "")
        )
    if cells:
        lines.append("- 연산자별 관문 통과(통과/시도): " + ", ".join(cells))
    any_try = g.get("reason_codes_any_try", {})
    lines.append(
        f"- 시도 중 명부 밖 인물을 만든 것 {any_try.get('new_person', 0)}건, 원문에 없던 명부 인물을 넣은 것 "
        f"{any_try.get('roster_people_added', 0)}건(시도 기준)"
    )
    excluded = b.get("B_original_on_excluded_mutated")
    if excluded and excluded["n"]:
        lines.append(
            f"- 관문에서 빠진 결함 사례 {excluded['n']}건의 원문 B 탐지 {excluded['hits']}/{excluded['n']} "
            "(통과 사례 원문의 탐지율과 견줘 제외가 B에 유리했는지 본다)"
        )
    return lines


def render_evaluation_md(summary: Mapping[str, Any]) -> str:
    g = summary["gate"]
    b = summary["b"]
    lines = [
        "# 재서술 강건성 시험 결과 (자동 생성)",
        "",
        f"생성 {summary['generated_at']} · 재서술 모델 `{summary['paraphrase_model']}` · 프롬프트 "
        f"`{PROMPT_VERSION}` · "
        f"관문 `{GATE_VERSION}` · 표본 {summary['sample_size']}건",
        "",
        "## 사실 보존 관문",
        "",
        f"- 재서술을 시도한 사례 {g['tried']}건 중 관문 통과 {len(g['kept'])}건 "
        f"(첫 시도 통과 {g['first_try_passed']}건, 최대 {MAX_ATTEMPTS}회 시도), 제외 "
        f"{g['tried'] - len(g['kept'])}건",
        f"- 제외 사유(마지막 시도 기준, 사례 수): {g['excluded_reason_codes_last_try'] or '없음'}",
        f"- 관문 재검사 불일치(실행 당시 판정과 지금 판정이 다른 시도): {len(g['gate_recheck_mismatch'])}건",
        *_gate_lines(summary, g, b),
        "",
        "## B(규칙 파서 + 그래프) — 같은 사례의 원문 대 재서술본",
        "",
        "| 지표 | 원문(표본 전체) | 원문(통과 사례만) | 재서술본(통과 사례) |",
        "|---|---|---|---|",
    ]
    full, kept, para = b.get("B_original_full_sample"), b.get("B_original_kept"), b.get("B_paraphrased_kept")

    def row(label: str, key: str) -> str:
        return (
            f"| {label} | {_pct(full and full[key])} | {_pct(kept and kept[key])} "
            f"| {_pct(para and para[key])} |"
        )

    lines += [
        row("탐지율(활성 8종)", "detection_active"),
        row("탐지율(봉인 2종)", "detection_held_out"),
        row("대조군 보정 탐지율(활성)", "detection_paired_active"),
        row("정상 대조군 오탐률", "clean_fp"),
    ]
    if para:
        lines += ["", "문체별 활성 탐지율(통과 사례): 원문 → 재서술본", ""]
        for style in STYLES:
            o = kept["by_style"].get(style, {}).get("active") if kept else None
            p = para["by_style"].get(style, {}).get("active")
            lines.append(f"- {style}: {_pct(o)} → {_pct(p)}")
        pair = b["paired_detection_all_mutated"]
        lines += [
            "",
            f"짝 비교(결함 사례): {pair['cells']}, McNemar 정확 검정 p = {pair['mcnemar_exact_p']}",
            f"필드 F1: 원문 {kept['field'] and kept['field']['f1']} → 재서술본 "
            f"{para['field'] and para['field']['f1']}; "
            f"누락 필드 환각률: {_pct(kept['hallucination'])} → {_pct(para['hallucination'])}; "
            f"근거 없는 요소: {para['ungrounded']}",
        ]
    if summary.get("llm"):
        lines += [
            "",
            "## D/DR (Claude 추출 + 그래프)",
            "",
            "```json",
            json.dumps(summary["llm"], ensure_ascii=False, indent=1)[:6000],
            "```",
        ]
    lines += [
        "",
        "## 읽는 법",
        "",
        "- 재서술본 수치는 관문을 통과한 재서술에 대한 값이다. 관문이 거른 재서술은 분모에 없다.",
        "- 기준 런북은 생성기의 합성 조직 20곳이고, 공개 런북 번안·공공누리 매뉴얼이 아니다.",
        "- 사전 등록 문서: docs/eval/paraphrase-plan.md",
    ]
    return "\n".join(lines) + "\n"


# =========================================================================== CLI


def _workers(requested: int | None) -> int:
    return requested if requested is not None else max(1, min(4, kit.default_workers()))


def cmd_dry_run(args: argparse.Namespace, corpus: Corpus) -> int:
    out: Path = args.out
    rows = stratified_sample(corpus)
    sealed = json.loads((ROOT / "docs/eval/manifest.json").read_text(encoding="utf-8"))
    sealed_hashes = set(sealed["case_hashes"])
    in_seal = sum(legacy_case_hash(corpus.by_id[r["case_id"]]) in sealed_hashes for r in rows)
    counts = {
        "by_style": dict(Counter(r["style"] for r in rows)),
        "by_operator": dict(Counter(r["operator"] or "clean" for r in rows)),
        "by_position": dict(Counter(r["position"] for r in rows)),
        "by_layout": dict(Counter(r["layout"] for r in rows)),
        "orgs": len({r["org_id"] for r in rows}),
        "active": sum(1 for r in rows if r["operator"] and not r["held_out"]),
        "held_out": sum(1 for r in rows if r["held_out"]),
        "clean": sum(1 for r in rows if r["operator"] is None),
    }
    write_json(
        out / "sample.json",
        {
            "generated_at": _now(),
            "corpus": {
                "generator": kit.GENERATOR_VERSION,
                "seed": SEED,
                "orgs": ORGS,
                "cases": len(corpus.cases),
            },
            "design": stratified_sample.__doc__.strip(),  # type: ignore[union-attr]
            "sealed_manifest_root": sealed["merkle_root"],
            "sample_cases_in_seal": f"{in_seal}/{len(rows)}",
            "counts": counts,
            "cases": rows,
        },
    )
    print(f"표본 {len(rows)}건 → {out / 'sample.json'} (봉인 해시와 일치 {in_seal}/{len(rows)})")
    print(f"  {counts}")
    ids = [r["case_id"] for r in rows]
    selftest = gate_selftest(corpus, ids)
    write_json(out / "gate_selftest.json", selftest)
    v1, v2 = selftest["v1_identity"], selftest["v2_defect_undone"]
    print(f"관문 자체 검증 → {out / 'gate_selftest.json'}")
    print(f"  V1 원문 그대로 통과 {v1['passed']}/{v1['cases']}")
    print(f"  V2 결함을 되돌린 글 거름 {v2['rejected']}/{v2['cases']}")
    for kind in PERTURBATIONS:
        v = selftest["v3_perturbations"][kind]
        print(f"  V3 {kind}: 거름 {v['rejected']}/{v['tried']}")
    est = estimates(corpus, ids)
    write_json(out / "estimate.json", {"generated_at": _now(), **est})
    p, d = est["paraphrase"], est["llm_d_dr"]
    print(f"비용 추정(목록가 {PRICES_AS_OF}, 호출 없음) → {out / 'estimate.json'}")
    print(
        f"  재서술 {p['model']}: 1회 시도 ${p['usd_one_attempt']} · 30% 재시도 ${p['usd_if_30pct_retried']} "
        f"· "
        f"전부 재시도 ${p['usd_worst_case_all_retried']} (참고: 글자 2.5개=1토큰이면 "
        f"${p['usd_one_attempt_at_2_5_chars_per_token']}, "
        f"gpt-6-luna면 ${p['usd_one_attempt_gpt_6_luna_alternative']})"
    )
    print(
        f"  D/DR {d['model']} effort {d['effort']}: 재서술본 ${d['usd_paraphrased_all_cases']} · 원문 "
        f"${d['usd_originals_all_cases']} · 짝 비교(둘 다) ${d['usd_both_versions_paired']} (180건 전부 통과 "
        f"가정; "
        f"입력을 글자 2.5개=1토큰으로 보면 짝 비교 ${d['usd_both_versions_input_at_2_5_chars_per_token']})"
    )
    if "run1_calibration" in d:
        cal = d["run1_calibration"]
        print(
            f"  D/DR 20% 실행 실측 보정(D {cal['d_rows_without_error']}건, "
            f"건당 ${cal['measured_usd_per_case']}, 실측/추정 ×{cal['ratio_measured_to_estimate']}): "
            f"짝 비교 ${d['usd_both_versions_paired_run1_calibrated']}"
        )
    if not args.skip_baseline:
        workers = _workers(args.workers)
        started = time.perf_counter()
        res = kit.run_eval([corpus.by_id[i] for i in ids], (Condition.B_RULE_GRAPH,), workers=workers)
        brief = _brief(kit.compute_metrics(res, generated_at=_now()), Condition.B_RULE_GRAPH)
        write_json(
            out / "baseline_b.json", {"generated_at": _now(), "workers": workers, "B_original_sample": brief}
        )
        print(f"B(원문 표본, 무료) → {out / 'baseline_b.json'} ({time.perf_counter() - started:.0f}초)")
        print(
            f"  활성 {_pct(brief['detection_active'])} · 봉인 {_pct(brief['detection_held_out'])} · "
            f"오탐 {_pct(brief['clean_fp'])}"
        )
    return 0


def _refuse(what: str, estimate_line: str) -> int:
    print(estimate_line)
    print(
        f"{what}은(는) 유료 호출입니다. 위 추정치를 확인한 뒤 --yes와 --max-usd를 붙여 다시 실행하세요. "
        f"(호출하지 않았습니다)"
    )
    return 2


def cmd_paraphrase(args: argparse.Namespace, corpus: Corpus) -> int:
    ids = load_sample(args.out, corpus)
    todo = pending_rewrites(args.out, ids)
    parts = [_openai_call_parts(corpus.by_id[cid]) for cid, _ in todo]
    usd = sum(openai_call_usd(args.model, i, o, CHARS_PER_TOKEN_CONSERVATIVE) for i, o in parts)
    line = (
        f"남은 재서술 {len(todo)}건, 모델 {args.model}, 1회 시도 추정 ${usd:.2f} "
        f"(최대 {MAX_ATTEMPTS}회 시도면 최대 ${usd * MAX_ATTEMPTS:.2f})"
    )
    if not args.yes or args.max_usd is None:
        return _refuse("재서술(OpenAI)", line)
    print(line)
    summary = run_sync(
        run_paraphrases(
            corpus, ids, args.out, model=args.model, max_usd=args.max_usd, concurrency=args.concurrency
        )
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["stopped"] is None else 3


def cmd_evaluate(args: argparse.Namespace, corpus: Corpus) -> int:
    ids = load_sample(args.out, corpus)
    if not read_jsonl(args.out / "paraphrases.jsonl"):
        print(f"{args.out / 'paraphrases.jsonl'}에 재서술 결과가 없습니다. 먼저 --paraphrase를 실행하세요.")
        return 1
    gate_summary = chosen_rewrites(args.out, corpus, ids)
    kept: dict[str, str] = gate_summary["kept"]
    rewrites = _paraphrased_cases(corpus, kept, ids)
    originals_kept = [corpus.by_id[cid] for cid in ids if cid in kept]
    llm_summary: dict[str, Any] | None = None
    if args.llm:
        jobs = [("original", c) for c in originals_kept] + [("paraphrase", c) for c in rewrites]
        done = {(r["variant"], r["case_id"]) for r in read_jsonl(args.out / "llm_results.jsonl")}
        todo = [(v, c) for v, c in jobs if (v, c.case_id) not in done]
        usd = sum(d_case_usd(c, D_MODEL, D_EFFORT)[1] for _, c in todo)
        line = (
            f"남은 D/DR 실행 {len(todo)}건(원문·재서술본), 모델 {D_MODEL} effort {D_EFFORT}, 추정 ${usd:.2f}"
        )
        if not args.yes or args.max_usd is None:
            return _refuse("D/DR(Claude)", line)
        print(line)
        run = run_sync(run_llm_arms(jobs, args.out, max_usd=args.max_usd, concurrency=args.concurrency))
        print(json.dumps(run, ensure_ascii=False))
        if run["stopped"] is not None:
            print(
                "INCOMPLETE: D/DR 실행이 멈췄습니다. 원인을 고친 뒤 같은 명령을 다시 실행하면 "
                "남은 것만 보냅니다."
            )
        llm_summary = {"run": run, **evaluate_llm(args.out, originals_kept, rewrites)}
    b = evaluate_b(corpus, ids, kept, _workers(args.workers))
    summary = {
        "generated_at": _now(),
        "paraphrase_model": PARAPHRASE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "gate_version": GATE_VERSION,
        "sample_size": len(ids),
        "gate": {k: v for k, v in gate_summary.items() if k != "kept"} | {"kept": sorted(kept)},
        "b": b,
        "llm": llm_summary,
    }
    write_json(args.out / "evaluation.json", summary)
    (args.out / "evaluation.md").write_bytes(render_evaluation_md(summary).encode("utf-8"))
    print(f"결과 → {args.out / 'evaluation.json'}, {args.out / 'evaluation.md'}")
    return 3 if llm_summary is not None and llm_summary["run"]["stopped"] is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="무료: 표본·관문 자체 검증·B 원문 기준선·비용 추정"
    )
    mode.add_argument(
        "--paraphrase", action="store_true", help="유료(OpenAI): 재서술 + 관문 (--yes, --max-usd 필요)"
    )
    mode.add_argument("--evaluate", action="store_true", help="무료: 통과한 재서술에서 B 평가 (--llm은 유료)")
    ap.add_argument("--llm", action="store_true", help="--evaluate와 함께: D/DR(Claude) 유료 실행")
    ap.add_argument("--yes", action="store_true", help="유료 호출 승인")
    ap.add_argument("--max-usd", type=float, default=None, help="이 단계의 누적 지출 상한(결과 폴더 기준)")
    ap.add_argument("--model", default=PARAPHRASE_MODEL, help="재서술 모델 (OpenAI)")
    ap.add_argument("--concurrency", type=int, default=4, help="동시 요청 수")
    ap.add_argument("--workers", type=int, default=None, help="B 실행 프로세스 수 (기본 최대 4)")
    ap.add_argument("--skip-baseline", action="store_true", help="--dry-run에서 B 원문 기준선 생략")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"결과 폴더 (기본 {DEFAULT_OUT})")
    args = ap.parse_args(argv)
    if args.llm and not args.evaluate:
        ap.error("--llm은 --evaluate와 함께만 씁니다")
    if args.model not in PRICES or not args.model.startswith("gpt-"):
        ap.error(f"재서술 모델은 가격표에 있는 OpenAI 모델이어야 합니다: {args.model}")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    corpus = build_corpus()
    if args.dry_run:
        return cmd_dry_run(args, corpus)
    if args.paraphrase:
        return cmd_paraphrase(args, corpus)
    return cmd_evaluate(args, corpus)


if __name__ == "__main__":
    raise SystemExit(main())
