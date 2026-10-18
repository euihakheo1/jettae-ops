"""Deterministic Korean rule parser: runbook text -> :class:`Extraction` IR with zero LLM calls.

This module is two things at once:

* the **LLM-OFF path** (P4) — when generative AI is disabled or fails, the pipeline still gets
  an operations graph from the same assembler funnel;
* the **"rule parser" arm of the 2×2 ablation** — so it must be strong on explicit runbooks
  (key/value lines, markdown tables, numbered escalation steps) and *honest* on colloquial
  ones: it only emits what a pattern can point at, never what it merely suspects (P2, P3).

How it reads a document
-----------------------
1. Quarantined ranges (prompt-injection text, masked PII) are blanked out before anything is
   scanned, so no mention or pattern can start, end or bridge inside them (P5).
2. The document is walked line by line keeping a *scope stack*: markdown headings, bracket
   headers ("[결제 서비스]"), a list line led by a service name ("- 결제: … 알람 → 김대리
   호출"), a trigger sentence that names its service ("결제 장애가 계속되면") and table rows
   all set the current service; sub-sections inherit it. Tables are read by their header: one
   row per service, an ordered contact ladder (순서 | 담당자 | 대기), a directory (이름 | 역할 |
   연락처) or a field sheet of the section (항목 | 내용).
3. Each sentence (or table cell) is turned into a *template* where every recognised mention
   becomes one placeholder character (person/role/team, channel, action, service). The relation
   patterns are ordinary regular expressions over that template, which keeps them readable
   ("P에게 C로 연락") and makes overlapping vocabularies impossible to confuse.
4. Scope state carries the facts that span sentences: the declared owner (else the first
   contact of free prose, decided when the scope closes; a call written in a labelled field
   such as "감지: … → 김대리 호출" is never taken for the owner), who was contacted last (the
   escalation source), the primary channel (the fallback source), triggers (paired as NOTIFIES
   with the contacts their own sentence names, else who receives the scope's pages, else the
   first contact) and obligations (for 신고 담당).
5. Negation is read where it changes the fact: "연락하지 마세요" is no contact, "김대리가
   아니라" no owner, "승인 불필요" no approver — and "승인 없이 진행 금지" a requirement.

Every quote is a verbatim slice of ``Document.text`` chosen so that its *first* occurrence in
the document is the very place it was read from whenever possible (the assembler grounds
quotes by search); a quote whose only occurrences are quarantined is never emitted.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

from jettae.domain import (
    EDGE_SIGNATURES,
    AttrKey,
    CatalogService,
    Document,
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    Extractor,
    IncidentType,
    NodeKind,
    RosterEntry,
)
from jettae.domain.graph import AttrValue
from jettae.extract.korean import (
    DURATION_PATTERN,
    NON_NAME_WORDS,
    NON_SERVICE_WORDS,
    NOUN_PATTERN,
    Lexicon,
    Mention,
    channel_reading,
    find_delay,
    find_durations,
    find_mentions,
    incident_type_of,
    incident_word_span,
    is_service_noun,
    names_channel,
    parse_minutes,
    recipients_in,
    service_from_cell,
    service_in_label,
)

Span = tuple[int, int]

# --------------------------------------------------------------------------- template alphabet

#: Placeholder characters (Unicode private use area: never present in normalized runbooks).
P = "\ue000"  # person / role / team
C = "\ue001"  # channel
A = "\ue002"  # action
S = "\ue003"  # catalog service / system

_PLACEHOLDER: dict[NodeKind, str] = {
    NodeKind.PERSON: P,
    NodeKind.ROLE: P,
    NodeKind.TEAM: P,
    NodeKind.CHANNEL: C,
    NodeKind.ACTION: A,
    NodeKind.SERVICE: S,
    NodeKind.SYSTEM: S,
}
_PEOPLE = frozenset({NodeKind.PERSON, NodeKind.ROLE, NodeKind.TEAM})
#: Entities named after their section's service ("결제 롤백"): their evidence is section-specific.
_SCOPED_KINDS = frozenset({NodeKind.ACTION, NodeKind.TRIGGER, NodeKind.OBLIGATION})
_APPROVER_KINDS = frozenset({NodeKind.PERSON, NodeKind.ROLE})
#: Markdown emphasis characters dropped from templates ("**담당자**: 김대리").
_TEMPLATE_SKIP = frozenset("*`")

# --------------------------------------------------------------------------- pattern pieces

_DUR = DURATION_PATTERN
_SEP = r"\s*(?:,|/|·|및|와|과|또는|혹은|이나|나|&)\s*"
#: Upper bound on the extra items of a listed mention ("김대리, 박과장 및 최팀장"). Real lists
#: are short; an unbounded ``(?:SEP P)*`` lets a hostile document of thousands of comma-separated
#: names drive every list pattern quadratic (each start position rescans the whole line).
_MORE = "{0,11}"
P_LIST = rf"{P}(?:{_SEP}{P}){_MORE}"
C_LIST = rf"{C}(?:{_SEP}{C}){_MORE}"
_TO = r"(?:에게로|에게|한테|께로|께)"
#: Recipient marker of a contact, which may carry a topic/additive particle ("박팀장에게도 공유").
_TO_X = rf"{_TO}(?:는|도|만)?"
#: Adverbs between a recipient and the channel/verb ("박팀장한테는 평소엔 슬랙으로 연락").
_ADV = (
    r"(?:(?:가장\s*먼저|먼저|우선|곧바로|바로|즉시|일단|빠르게|신속히"
    r"|평소(?:에는|엔|에)?|평상시(?:에는|엔|에)?|보통(?:은)?|주로)\s*)"
)
_CONTACT_VERB = r"(?:연락|전화|호출|알림|알리|알려|공유|콜|문의|메시지|메세지|DM|멘션|페이징|전파|통보|전달)"
_GIVE_VERB = r"(?:주|줘|보내|드리|드려|남기|남겨|날리|날려|때리|때려|걸)"

_TIME_MOD = (
    r"(?:이내에?|내에?|안에|동안|이상|넘게|넘도록|지나도|지나서도|넘어도"
    r"|경과\s*(?:후|시)에?도?|후에도|뒤에도|후|뒤)"
)
_NO_ANSWER = (
    r"(?:미응답|무응답|응답\s*(?:이\s*)?(?:없(?:으면|을\s*(?:경우|시|때)|는\s*경우)|안\s*(?:오면|되면))"
    r"|답(?:변)?\s*(?:이\s*)?(?:없(?:으면|을\s*(?:경우|시|때)|는\s*경우)|안\s*(?:오면|올\s*(?:때|경우)))"
    r"|반응\s*(?:이\s*)?없(?:으면|을\s*(?:경우|시|때)|는\s*경우)"
    r"|응답하지\s*않(?:으면|을\s*(?:경우|시|때)|는\s*경우)"
    r"|연락\s*(?:이\s*)?(?:안\s*(?:되면|될\s*(?:경우|시|때)|닿으면)|두절\s*시|불가\s*시)"
    r"|안\s*받(?:으시|으)면|받지\s*않(?:으시|으)면|못\s*받(?:으시|으)면|확인\s*(?:이\s*)?안\s*되면|확인되지\s*않으면"
    r"|해결\s*(?:이\s*)?안\s*되면|해결되지\s*않으면|안\s*되면|안\s*될\s*(?:경우|때|시)"
    r"|조용(?:하면|할\s*(?:경우|때))|부재\s*시|부재중이면|부재면|불가\s*시)"
)
_TIME_ONLY = r"(?:경과\s*(?:시|후)|지나면|넘으면|넘기면|초과\s*시)"
#: After a no-answer noun: "무응답 시", "미응답이면", "무응답일 경우".
_NO_ANSWER_TAIL = r"(?:시에?|때|이면|이라면|일\s*(?:때|경우|시))"
_CONDITION = (
    rf"(?:(?:(?P<m1>{_DUR})\s*{_TIME_MOD}?\s*)?{_NO_ANSWER}\s*{_NO_ANSWER_TAIL}?"
    rf"|(?P<m2>{_DUR})\s*{_TIME_ONLY})"
)
#: A further no-answer condition right after a name: "박팀장, 최실장도 안 되면" is a chain step.
_CONDITION_AHEAD = rf"\s*(?:이|가|도|께서|은|는)?\s*(?:{_DUR}\s*\S*\s*)?{_NO_ANSWER}"
_ESCALATE_VERB = (
    r"(?:상향|에스컬레이션|[Ee]scalat[a-z]*|보고|연락|호출|올려|올리|알림|알려|공유|이관|전달|넘기|넘겨|요청)"
)

_APPROVE = r"(?:승인|결재|허가|컨펌|[Cc]onfirm)"
_APPROVE_TAIL = r"(?:받아|받고|받은\s*(?:후|뒤|다음)|받은|받|후|뒤|하에|필요|필수|있어야|얻|득)"
_EXEC = r"(?:실행|수행|진행|처리|조치|작업)(?!\s*(?:상황|현황|사항|여부|내역))"
_DOWN = (
    r"(?:안\s*되면|안\s*될\s*(?:때|경우|시)|장애\s*(?:발생\s*)?시|장애가?\s*나면|다운\s*시|다운되면|불통\s*시"
    r"|불통이면|먹통(?:이면|\s*시)|불가\s*시|접속\s*(?:불가|안\s*되면)\s*시?|안\s*터지면|끊기면|두절\s*시|마비\s*시)"
)
#: Lazy noun: inside larger patterns it stops before particles ("결제도 영향" -> "결제").
_LAZY_NOUN = NOUN_PATTERN + "?"
_NOUN_PHRASE = rf"{_LAZY_NOUN}(?:\s{_LAZY_NOUN}){{0,2}}"
_WORD_START = r"(?<![가-힣A-Za-z0-9])"

# --------------------------------------------------------------------------- relation patterns

ESCALATE_RE = re.compile(
    rf"(?:(?P<src>{P})\s*(?:이|가|께서|도|은|는)?\s*)?{_CONDITION}\s*[,，]?\s*"
    rf"(?:(?:그\s*)?다음(?:은|에는|엔|으로|으론)?|이후(?:에는|엔)?)?\s*"
    rf"(?:→|->|=>|:)?\s*(?P<dst>{P}(?:{_SEP}{P}(?!{_CONDITION_AHEAD})){_MORE})\s*{_TO}?\s*(?:으로|로)?\s*"
    rf"(?:(?P<via>{C_LIST})\s*(?:으로|로)?\s*)?(?P<verb>{_ESCALATE_VERB})?"
)
CONTACT_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*{_TO_X}\s*{_ADV}?(?:(?P<via>{C_LIST})\s*(?:으로|로|을|를)?\s*{_ADV}?"
    rf"(?:{_CONTACT_VERB}|{_GIVE_VERB})?|{_CONTACT_VERB})"
)
CONTACT_VIA_FIRST_RE = re.compile(
    rf"(?P<via>{C_LIST})\s*(?:으로|로|을\s*통해|를\s*통해)\s*(?P<who>{P_LIST})\s*{_TO_X}\s*{_ADV}?"
    rf"(?:{_CONTACT_VERB}|{_GIVE_VERB})?"
)
#: "김대리 호출" is as common as "김대리를 호출" in terse runbooks.
CALL_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:(?:을|를)\s*)?{_ADV}?(?:(?P<via>{C_LIST})\s*(?:으로|로)\s*)?(?:호출|페이징|소집)"
)
#: The channel of a call written in parentheses after it: "김대리 호출(슬랙)", "박과장에게 연락 (전화)".
CALL_PAREN_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:(?:을|를|{_TO})\s*)?{_ADV}?(?:호출|페이징|연락|알림|전파|공유)\s*"
    rf"[(（]\s*(?P<via>{C_LIST})\s*[)）]"
)
#: The person receives the call: "김대리가 슬랙으로 호출을 받는다", "박과장이 전화로 연락받음".
CONTACT_PASSIVE_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|은|는|도)\s*{_ADV}?(?P<via>{C_LIST})\s*(?:으로|로)\s*"
    r"(?:호출|연락|알림|전화|페이징|메시지|메세지)\s*(?:을|를|이|가)?\s*받"
)
#: "김대리가 아니라 박과장", "김대리 아님": a name right before a denial is not being declared.
_NOT_THIS = r"(?!\s*(?:이|가)?\s*아(?:니|님))"
OWNER_DECL_RE = re.compile(
    rf"(?:(?:1차|주|메인|정)\s*)?(?<![2-9]차)(?<![2-9]차 )(?<!최종)(?<!최종 )(?<!마지막 )"
    rf"(?:담당|책임|오너|[Oo]wner)\s*(?:자|팀|부서)?\s*(?:\([^)\n]*\)\s*)?"
    rf"(?:(?:은|는)\s*[:：=]?|[:：=])\s*(?P<who>{P_LIST})(?!\s*{_TO}){_NOT_THIS}"
)
FIRST_RESPONDER_RE = re.compile(
    rf"1차\s*(?:담당|대응|연락)\s*(?:자|팀|처|대상)?\s*(?:은|는|[:：=])?\s*(?P<who>{P_LIST})(?!\s*{_TO}){_NOT_THIS}"
)
#: Numbered contact order: "1차: 김대리 / 2차: 박팀장 / 최종: CTO" (1차 = owner, later = escalation).
ORDINAL_RE = re.compile(
    rf"(?:(?P<ord>[1-9])\s*차|최종|마지막)\s*(?:담당|대응|연락|책임)?\s*(?:자|처|대상)?\s*[:：=]\s*"
    rf"(?P<who>{P_LIST})(?!\s*{_TO}){_NOT_THIS}"
)
OWNER_SUBJECT_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:1차\s*)?(?:담당(?!자)|대응\s*담당|책임지|맡)"
    r"(?!\S*\s*(?:않|아니|안\s*(?:함|한다|해)))"  # "김대리가 담당하지 않는다" denies ownership
)
OWNER_NEGATIVE_RE = re.compile(
    r"(?:담당|책임|오너)\s*(?:자|팀)?\s*(?:은|는)?\s*[:：=]?\s*"
    r"(?:없음|없다|없습니다|미정|미지정|공석|[-—–]|TBD|tbd|N/?A|n/?a|\?|확인\s*필요|추후\s*지정)"
)
#: The clause before "X가 맡는다" makes a task its topic ("신고는 X가 맡아요", "롤백은 X가 담당"):
#: X takes on that task, which says nothing about who owns the service.
_TASK_TOPIC_RE = re.compile(
    rf"(?:{A}|신고|통지|보고|공지|승인|결재|판단|결정|작업|실행|수행|조치|복구|처리)\s*(?:업무|작업|판단|결정)?\s*"
    rf"(?:은|는|도)\s*[^{P}{C}{A}{S}]{{0,20}}$"
)
# Approval patterns name the approval word ``ap``: what follows it decides the polarity
# ("승인 후" requires approval, "승인 불필요" waives it, "승인 없이 진행 금지" requires it again).
APPROVE_ACTION_RE = re.compile(
    rf"(?P<act>{A}(?:{_SEP}{A}){_MORE})\s*(?:은|는|을|를|시에는|시|할\s*때|하기\s*전에?|전에?)?\s*(?:반드시\s*|꼭\s*)?"
    rf"(?P<who>{P_LIST})\s*(?:의|에게|께)?\s*(?:사전\s*|최종\s*)?(?P<ap>{_APPROVE})"
)
APPROVE_SUBJECT_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:의|에게|께|한테|으로부터|로부터)?\s*(?:사전\s*|최종\s*)?(?P<ap>{_APPROVE})\s*"
    rf"(?:을|를|이|가)?\s*{_APPROVE_TAIL}"
)
APPROVE_DECL_RE = re.compile(
    rf"(?P<ap>{_APPROVE})\s*(?:(?:자|권자|권한자|라인|담당\s*자?)\s*(?:은|는)?\s*[:：=]?|[:：=])\s*"
    rf"(?P<who>{P_LIST}){_NOT_THIS}"
)
#: Who decides whether to take an action authorises it: "롤백 판단은 박팀장이 한다",
#: "페일오버 여부는 CTO가 결정".
_DECIDE = r"(?:판단|결정|의사\s*결정)"
APPROVE_JUDGE_RE = re.compile(
    rf"(?P<act>{A}(?:{_SEP}{A}){_MORE})\s*(?:여부\s*)?(?:의\s*)?(?P<ap>{_DECIDE})\s*(?:은|는)\s*"
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:최종\s*)?(?:하|해|합|내리|내려|맡)"
)
APPROVE_JUDGE_SUBJECT_RE = re.compile(
    rf"(?P<act>{A}(?:{_SEP}{A}){_MORE})\s*(?:여부|진행\s*여부)\s*(?:은|는)\s*"
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:최종\s*)?(?P<ap>{_DECIDE})"
)
#: Right after an approval word: the approval is waived ("승인 불필요", "승인을 받지 않아도").
_NOT_REQUIRED = (
    r"\s*(?:을|를|이|가|은|는|도)?\s*"
    r"(?:없이|없어도|불필요|필요\s*(?:없|하지\s*않|치\s*않)|생략|받지\s*않|안\s*받|면제)"
)
NOT_REQUIRED_RE = re.compile(_NOT_REQUIRED)
NO_APPROVAL_RE = re.compile(rf"(?:승인|결재)(?={_NOT_REQUIRED})")
#: A prohibition later in the clause turns a waiver back into a requirement:
#: "승인 없이 진행 금지" = 승인이 필요하다.
PROHIBITION_RE = re.compile(
    r"금지|불가|안\s*(?:됨|된다|됩니다|돼)|(?:하|되)지\s*(?:않|말|마)|하면\s*안|해서는\s*안"
    r"|말\s*것|수\s*없|못\s*(?:함|한다|합니다|해)"
)
#: Directly before an action: "승인 없이 롤백 금지" prohibits *unapproved* rollback only.
_UNAPPROVED_BEFORE_RE = re.compile(
    r"(?:승인|결재)\s*(?:을|를)?\s*(?:없이|안\s*받고|받지\s*않고)\s*(?:는|도)?\s*$"
)
#: Spoken "run it": "김대리가 돌리면 된다" (the action itself is named earlier in the sentence).
_RUN = r"(?:돌리|돌려|돌립)"
EXECUTE_SUBJECT_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:직접\s*)?"
    rf"(?:(?P<act>{A})(?!\s*(?:을|를)?\s*{_APPROVE})\s*(?:을|를)?\s*(?:{_EXEC})?|{_EXEC}|{_RUN})"
)
EXECUTE_ACTION_RE = re.compile(
    rf"(?P<act>{A})\s*(?:작업)?\s*(?:은|는)\s*(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:직접\s*)?"
    rf"(?:{_EXEC}|담당|맡|하)"
)
EXECUTE_DECL_RE = re.compile(
    rf"(?:실행|수행|작업)\s*(?:자|담당\s*자?|주체)?\s*(?:은|는)?\s*[:：=]\s*(?P<who>{P_LIST}){_NOT_THIS}"
)
#: The work itself as the topic: "실제 작업은 김대리가 해요", "실행은 박과장이 맡는다".
EXECUTE_TOPIC_RE = re.compile(
    rf"(?:실제\s*)?(?:작업|실행|수행|처리)\s*(?:은|는)\s*(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:직접\s*)?"
    r"(?:해|하|합|진행|맡|담당|수행|실행|처리)"
)
#: "롤백 담당: 김대리" names who performs the rollback, not who owns the service.
EXECUTE_ACTION_DECL_RE = re.compile(
    rf"(?P<act>{A})\s*(?:담당|책임)\s*(?:자)?\s*(?:은|는)?\s*[:：=]\s*(?P<who>{P_LIST}){_NOT_THIS}"
)
PAREN_CONTACT_RE = re.compile(rf"(?P<who>{P})\s*[(（](?P<inner>[^)）\n]*)[)）]")
PERSON_COLON_RE = re.compile(rf"\s*(?P<who>{P})\s*[:：]\s*(?P<via>{C_LIST})")
CONTACT_DECL_RE = re.compile(
    rf"(?:(?P<who>{P})\s*(?:의)?\s*)?(?:연락\s*(?:처|수단|방법|채널)|연락(?=\s*(?:은|는|[:：=]))"
    rf"|주\s*(?:채널|연락\s*수단)|채널)\s*(?:은|는)?\s*[:：=]?\s*(?P<via>{C_LIST})"
)
#: After a channel: an optional note and arrow before its outage ("사내 메신저(1순위) → 안 되면").
_CHANNEL_NOTE = r"\s*(?:[(（][^)）\n]{0,20}[)）]\s*)?(?:(?:→|->|=>)\s*)?"
FALLBACK_RE = re.compile(
    rf"(?P<a>{C}){_CHANNEL_NOTE}(?:이|가|은|는|도)?\s*{_DOWN}\s*(?:에는|엔|에)?\s*[,，]?\s*(?:→|->|=>)?\s*"
    rf"(?:{P_LIST}\s*{_TO}?\s*)?(?P<b>{C}(?:{_SEP}{C}(?!\s*(?:이|가|은|는|도)?\s*{_DOWN})){_MORE})"
)
FALLBACK_DECL_RE = re.compile(
    rf"(?:(?:대체|백업|보조|예비|비상)\s*(?:채널|연락\s*(?:수단|채널|처|망)?|수단)\s*(?:은|는)?\s*[:：=]?"
    rf"|(?:대체|백업|예비)\s*[:：=])\s*(?P<b>{C_LIST})"
)
#: The contact through the first channel got no answer (not "해결이 안 되면": that is no channel).
_UNREACHED = (
    r"(?:답(?:변)?\s*(?:이\s*)?(?:없(?:으면|을\s*(?:경우|때))|안\s*(?:오면|올\s*(?:때|경우)))"
    r"|응답\s*(?:이\s*)?(?:없(?:으면|을\s*(?:경우|때))|안\s*오면)|(?:무응답|미응답)(?:이면|\s*시)"
    r"|(?:안|못)\s*받(?:으시|으)면|연락\s*(?:이\s*)?(?:안\s*(?:되면|닿으면)|두절\s*시)|안\s*되면|안\s*될\s*(?:때|경우))"
)
#: The same person tried on a second channel once the first gets no answer: "슬랙으로 연락하시고,
#: 안 되면 전화로", "카톡 먼저 주시고, 답이 안 오면 문자로 다시 연락". The first channel must be
#: used to reach someone (a contact verb right after it), no person may stand between the two
#: channels, and none may follow the second in its clause — that would be an escalation.
FALLBACK_SEQ_RE = re.compile(
    rf"(?P<a>{C})\s*(?:으로|로|을|를)?\s*{_ADV}?(?:{_CONTACT_VERB}|{_GIVE_VERB})[가-힣]{{0,4}}"
    rf"(?:\s*(?:주|줘)[가-힣]{{0,3}})?\s*[,，]?\s*{_UNREACHED}\s*[,，]?\s*(?:다시\s*|그때는?\s*|그럼\s*)?"
    rf"(?P<b>{C})\s*(?:으로|로)(?![^,，.\n{P}]*{P})"
)
#: A channel with a rank note: "슬랙(1순위), 문자(대체)", "전화 (백업)".
RANKED_CHANNEL_RE = re.compile(rf"{C}\s*[(（]\s*(?P<note>[^)）\n]{{1,12}}?)\s*[)）]")
_PRIMARY_NOTE_RE = re.compile(
    r"(?:1|일)\s*순위|1\s*차|주\s*(?:채널|연락\s*수단)?|기본|메인|우선|평소|[Pp]rimary|[Mm]ain"
)
_SECONDARY_NOTE_RE = re.compile(
    r"[2-9]\s*순위|[2-9]\s*차|대체|백업|보조|예비|비상|[Ss]econdary|[Bb]ackup|[Ff]allback"
)
#: A channel outage as a condition ("슬랙이 안 되면", "카톡 불통 시").
CHANNEL_DOWN_RE = re.compile(rf"{C}{_CHANNEL_NOTE}(?:이|가|은|는|도)?\s*{_DOWN}")
#: A prohibition right after a contact verb: "연락하지 마세요", "전화 금지", "연락하면 안 됨".
PROHIBITED_AFTER_RE = re.compile(
    r"\s*(?:은|는|을|를|도)?\s*"
    r"(?:(?:하|드리|주|걸|보내|남기)?(?:시)?지\s*(?:마|말|않)|(?:하|해)(?:서는|선|면)\s*안\s*(?:됨|되|돼|된)"
    r"|안\s*(?:함|한다|합니다|해도)|금지|불가|자제|말\s*것|필요\s*없|불필요)"
)
NEGATED_ACTION_RE = re.compile(
    rf"{A}\s*(?:은|는|이|가|을|를)?\s*(?:절대\s*)?"
    rf"(?:불가|금지|하지\s*마|하지\s*말|하면\s*안|안\s*됨|불가능|하지\s*않|미지원|지원\s*안)"
)
ROLE_COLON_RE = re.compile(rf"(?P<role>{P})\s*[:：=]\s*(?P<who>{P})")
ROLE_PAREN_RE = re.compile(rf"(?P<who>{P})\s*[(（]\s*(?P<role>{P})\s*[)）]")
REPORTER_SUBJECT_RE = re.compile(
    rf"(?P<who>{P_LIST})\s*(?:이|가|께서|에서)\s*(?:[^\s{P}]+\s*){{0,3}}?(?:신고|통지)"
)
REPORTER_OBJECT_RE = re.compile(
    rf"(?:신고|통지)\s*(?:업무)?\s*(?:는|은)\s*(?P<who>{P_LIST})\s*(?:이|가|께서|에서)"
)
REPORTER_DECL_RE = re.compile(rf"(?:신고|통지)\s*(?:담당|책임)\s*(?:자)?\s*[:：=]\s*(?P<who>{P_LIST})")
#: "KISA 신고(담당: 김CISO)": the person in charge of the report, noted right after it.
REPORTER_PAREN_RE = re.compile(
    rf"(?:신고|통지)\s*[(（]\s*(?:담당|책임|주관)\s*(?:자)?\s*[:：=]?\s*(?P<who>{P_LIST})\s*[)）,，]"
)
#: "신고는 24시간 안에 해야 하는데, 김CISO 담당이에요": the reporter named as "X 담당" at the end.
REPORTER_ROLE_RE = re.compile(
    rf"(?:신고|통지)[^{P}\n]{{0,60}}?[,，]\s*(?P<who>{P_LIST})\s*(?:이|가)?\s*(?:담당|책임|소관)"
    r"(?=\s*(?:이에요|이예요|예요|입니다|이다|이야|이고|이며|임|이에|[.。!]|$))"
)
#: Words that refer to "the mitigation steps" without naming one (approval/executor context).
_GENERIC_ACTION_RE = re.compile(r"조치|복구\s*작업|대응\s*작업")
_HOURS_RE = re.compile(r"(?<![\d.])(?P<h>\d+)\s*시간\s*(?:이내|내에|내|안에|까지)")
_REPORT_VERB_RE = re.compile(r"신고|통지|보고")

# --------------------------------------------------------------------------- trigger patterns

_SIGNAL = (
    r"(?:장애\s*알람|알람|경보|알림|모니터링|고객\s*문의|문의|VOC|민원|[Aa]lert|ALERT|[Aa]larm|오류율|에러율|에러\s*로그|에러|오류|응답\s*지연"
    r"|응답\s*시간|지연\s*시간|지연|레이턴시|[Ll]atency|타임아웃|[Tt]imeout|5xx|5XX|50[234]|4xx|결제\s*실패율|실패율"
    r"|CPU|cpu|메모리|디스크|트래픽|헬스\s*체크|헬스체크|[Hh]ealth\s*[Cc]heck|큐\s*적체|적체|TPS)"
)
_SIGNAL_END = (
    r"(?:초과|이상|발생|지속|감지|울리면|울릴\s*(?:때|경우)|뜨면|떨어지면"
    r"|미만|급증|폭증|증가|도달|수신|오면|올\s*때)"
)
ALERT_RE = re.compile(
    rf"{_WORD_START}(?<!에게 )(?<!한테 )(?<!께 )(?:(?P<svc>{S})\s*(?:서비스\s*)?)?"
    rf"(?P<t>(?:[가-힣A-Za-z0-9%._\-]+\s+){{0,3}}?{_SIGNAL}[^{P}{C}{A}{S}\n]{{0,30}}?{_SIGNAL_END}"
    rf"(?:\s*(?:이|가)?\s*{_DUR}\s*(?:이상\s*)?(?:지속|계속))?)"
)
#: Optional service subject of a trigger: "결제 장애가", "결제 쪽 문제", "결제 서비스에서 장애".
_SUBJECT = (
    rf"{_WORD_START}(?:(?P<svc>{S})|(?P<noun>{_LAZY_NOUN}))?\s*(?:서비스\s*|시스템\s*)?"
    rf"(?:쪽에서\s*|쪽\s*|측\s*|에서\s*)?"
)
PERSIST_RE = re.compile(
    rf"{_SUBJECT}(?P<kind>장애|문제|이슈|에러|오류|지연)\s*(?:가|이)?\s*(?:계속|지속)\s*"
    rf"(?:되면|될\s*(?:경우|때|시)|되는\s*경우|되거나|시)"
)
OCCUR_RE = re.compile(
    rf"{_SUBJECT}(?P<kind>장애|문제|이슈)\s*(?:가|이)?\s*"
    rf"(?:생기면|생길\s*(?:경우|때|시)|발생하면|발생\s*시|발생할\s*(?:경우|때|시)|발생한\s*경우|나면|날\s*(?:경우|때)"
    rf"|터지면|일어나면|시(?:에는|에)?(?![가-힣])|때(?![가-힣]))"
)
DETECT_DECL_RE = re.compile(
    rf"(?:감지|탐지|모니터링)\s*(?:방법|수단|조건|기준|채널|도구)?\s*(?:은|는)?\s*[:：=]\s*(?P<v>[^{P}{A}\n]{{2,60}})"
)
#: An alarm that goes off under a stated condition, whatever the metric is called:
#: "주문 실패가 5분 넘게 이어지면 알람이 울린다", "알람은 지연 상태가 감지되면 바로 떠요".
#: The condition (``t``) must end in a conditional verb and the alarm must fire right after it.
_ALARM = r"(?:알람|알림|경보|얼럿|[Aa]lert|[Aa]larm)"
#: Without a verb saying it goes off, "알림" is as likely a customer notice ("지연 시 알림 발송").
_ALARM_STRICT = r"(?:알람|경보|얼럿|[Aa]lert|[Aa]larm)"
#: The alarm goes off: 울리다/뜨다/오다/나다/발생하다 in any ending ("울린다", "떠요", "옵니다").
_ALARM_FIRES = r"(?:울[리려립린림]|뜨|떠|뜹|뜬|와|오|옵|온|발생|나|납|난|날)"
_COND_END = (
    r"(?:이어지면|지속되면|계속되면|감지되면|발생하면|확인되면|넘으면|넘어가면|넘기면|초과하면|떨어지면"
    r"|올라가면|늘어나면|줄어들면|쌓이면|밀리면|튀면|되면|이면|하면)"
)
_MONITOR_PREFIX = (
    r"(?:(?:모니터링|대시보드|관제|[Gg]rafana|그라파나|[Dd]atadog|데이터독)\s*(?:에서|상|에)?\s*)?"
)
#: One character of a condition: no mention but a service, no clause break or field label colon
#: (a decimal point or digit grouping inside a number is fine: "p95 1.5초", "1,000건").
_COND_CHAR = rf"(?:[^{P}{C}{A}\n.,，;:：]|(?<=\d)[.,](?=\d))"
#: A measured condition stated as a noun before "시 알람": "대기열 500건 초과 1분 지속 시 알람".
_COND_NOUN_END = r"(?:초과|이상|이하|미만|지속|급증|급감|폭증|증가|감소|누락|실패|지연|중단|도달)"
ALARM_COND_RE = re.compile(
    rf"(?:{_ALARM}\s*(?:은|는)\s*{_MONITOR_PREFIX}(?P<t1>{_COND_CHAR}{{2,80}}?{_COND_END})"
    rf"\s*[,，]?\s*(?:곧바로\s*|바로\s*|즉시\s*)?{_ALARM_FIRES}"
    rf"|{_MONITOR_PREFIX}(?P<t2>{_COND_CHAR}{{2,80}}?{_COND_END})"
    rf"\s*[,，]?\s*(?:곧바로\s*|바로\s*|즉시\s*)?{_ALARM}\s*(?:이|가)\s*{_ALARM_FIRES}"
    rf"|{_MONITOR_PREFIX}(?P<t3>{_COND_CHAR}{{2,80}}?{_COND_NOUN_END})\s*(?:시|때)(?:에는|엔|에)?\s*"
    rf"(?:곧바로\s*|바로\s*|즉시\s*)?{_ALARM_STRICT})"
)

# --------------------------------------------------------------------------- dependency patterns

DEPENDS_RE = re.compile(
    rf"{_WORD_START}(?:(?P<s>{S})|(?P<sn>{_LAZY_NOUN}))\s*(?:서비스\s*)?(?:는|은)\s*"
    rf"(?:(?P<x>{S})|(?P<xn>{_NOUN_PHRASE}))\s*(?P<xs>서비스)?\s*(?:에|를|을)\s*의존"
)
DEPENDS_SUBJECTLESS_RE = re.compile(
    rf"{_WORD_START}(?:(?P<x>{S})|(?P<xn>{_NOUN_PHRASE}))\s*(?P<xs>서비스)?\s*(?:에|를|을)\s*의존"
)
DEPENDS_IMPACT_RE = re.compile(
    rf"{_WORD_START}(?:(?P<x>{S})|(?P<xn>{_LAZY_NOUN}))\s*(?P<xs>서비스)?\s*(?:시스템\s*)?장애\s*"
    rf"(?:시|가\s*나면|발생\s*시|가\s*발생하면)\s*[,，]?\s*(?:(?P<s>{S})|(?P<sn>{_LAZY_NOUN}))\s*(?:서비스\s*)?"
    rf"(?:도|까지|역시|가|이)?\s*(?:영향|중단|장애|불가|마비|멈)"
)

# --------------------------------------------------------------------------- document layout

_HEADING_RE = re.compile(r"[ \t]{0,3}(?P<hashes>#{1,6})[ \t]+(?P<body>.*?)[ \t]*#*[ \t]*")
_FENCE_RE = re.compile(r"[ \t]{0,3}(?:```|~~~)")
_TABLE_ROW_RE = re.compile(r"\s*\|.*\|\s*")
_TABLE_SEP_RE = re.compile(r"\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*")
_MARKER_RE = re.compile(r"(?:\s*>)*\s*(?:(?:[-*+•▪◦·]|\d{1,3}[.)]|[①-⑳]|[가-하][.)])\s+)?(?:\[[ xX]\]\s+)?")
#: "[결제 서비스]" section header. Markdown link text ("[past incident](url)", "[title][ref]") is not
#: one: a list of links to other pages must not open one service scope per link title.
_BRACKET_RE = re.compile(r"\[(?P<inner>[^\]\n]{1,40})\](?![(\[])")
#: A line that starts with a label and a colon ("담당: …", "결제: …", "**감지**: …").
_LABEL_RE = re.compile(r"(?P<label>[^:：\n|\[\]]{1,40}?)\s*[:：](?:\s+|$)")
#: What may follow a service name inside a label: "결제 서비스", "회원 인증 (Tier 1)".
_LABEL_TAIL_RE = re.compile(r"\s*(?:서비스|시스템)?\s*(?:[(（][^)）\n]{0,20}[)）])?[\s*`_]*")
#: Field labels (by their table-header role) that make a list line a labelled field.
#: ("담당:"/"연락:" are not among them: the person named there may well be the owner).
_FIELD_ROLES = frozenset({"contact", "fallback", "approver", "escalation", "executor", "action", "trigger"})
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\x00+")
_CLAUSE_SPLIT_RE = re.compile(r"[,，;；]")
_NEGATIVE_CELL_RE = re.compile(
    r"(?:-|—|–|없음|미정|미지정|공석|TBD|tbd|N/?A|n/?a|\?|확인\s*필요|해당\s*없음|x|X|\s*)"
)
#: A 역할 cell that makes the row's person the owner ("1차 담당", "담당자", "주 담당").
_OWNER_CELL_RE = re.compile(r"(?:1차|주|정|메인)?\s*(?:담당|책임|오너|[Oo]wner)\s*자?")
_NAME_CELL_RE = re.compile(r"[가-힣]{2,4}|[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}")
_CHAIN_RE = re.compile(r"→|->|=>|>")
_BOUNDARY_CHARS = frozenset(" \t\n\r\x00,.;:!?()[]{}<>|\"'“”‘’、。，：；（）「」『』→")
_TRIM_CHARS = " \t\r\n\x00"
_CELL_TRIM_CHARS = " \t\r\n\x00*`_"

#: Sentence-initial fillers that carry no meaning in a condition label.
_FILLER_WORDS = frozenset(
    {
        "만약", "만일", "혹시", "일단", "우선", "먼저", "그리고", "또는",
        "및", "그래도", "다시", "갑자기", "또", "즉시", "바로",
    }
)  # fmt: skip

#: Longest quote the parser emits when it has a choice: candidates (clause, sentence, line,
#: supporting passage) beyond this are not used; a single over-long statement is still quoted.
_MAX_QUOTE = 400

_BRACKET_LEVEL = 7
_INLINE_LEVEL = 8
_ROW_LEVEL = 9

#: Table header -> column role. First match wins, so specific headers come first
#: ("대체 채널" before "채널", "2차 담당" before "담당", "담당자 연락처" before "담당").
_HEADER_ROLES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (role, re.compile(pattern))
    for role, pattern in (
        # a directory: who (이름) and what they are (역할) — see ``_row_members``
        ("person", r"^(?:이름|성명|성함|name|인원)$"),
        ("role", r"^(?:역할|직책|role|position)$"),
        ("fallback", r"(?:대체|백업|보조|예비|비상).*(?:채널|연락|수단)"),
        ("approver", r"승인|결재|approv"),
        # "상향 시간 | 15분" states when to escalate, not to whom
        ("wait", r"(?:상향|에스컬|escalat)\w*(?:시간|기준|대기|timeout)"),
        ("escalation", r"상향|에스컬|escalat|2차"),
        ("executor", r"실행|수행|작업자|(?:롤백|조치|복구|작업)(?:실행)?(?:담당|책임)"),
        # a task-specific person column ("신고 담당", "공지 담당") does not own the service
        ("other", r"신고|통지|공지|보고|커뮤니케이션|홍보|대외"),
        ("contact", r"연락처|연락수단|연락방법|채널|contact|channel"),
        ("owner", r"담당|책임|오너|owner|1차|연락대상|연락"),
        # any mitigation the action lexicon knows ("페일오버 | 팀장 승인 후 김대리가 실행")
        (
            "action",
            r"롤백|조치|복구|원복|페일오버|절체|dr전환|재시작|재기동|재부팅|리스타트|플래그|스케일아웃|증설"
            r"|action|mitigat|rollback|failover|restart|reboot|featureflag|killswitch|scaleout",
        ),
        ("trigger", r"감지|알람|탐지|모니터|trigger|alert|alarm"),
        ("service", r"서비스|시스템|service|system|대상"),
        ("order", r"^(?:순서|단계|차수|순번|step|level|no\.?|#)$"),
        ("wait", r"대기|타임아웃|timeout|무응답|응답시간"),
    )
)

# --------------------------------------------------------------------------- small helpers


class _AttrSpec(NamedTuple):
    key: AttrKey
    value: AttrValue
    cands: tuple[Span, ...]


def _overlaps(span: Span, others: Iterable[Span]) -> bool:
    return any(span[0] < e and span[1] > s for s, e in others)


def _dedupe(spans: Iterable[Span | None]) -> tuple[Span, ...]:
    out: list[Span] = []
    for span in spans:
        if span is not None and span[1] > span[0] and span not in out:
            out.append(span)
    return tuple(out)


def _trim(text: str, start: int, end: int, chars: str = _TRIM_CHARS) -> Span:
    while start < end and text[start] in chars:
        start += 1
    while end > start and text[end - 1] in chars:
        end -= 1
    return start, end


def _working_text(doc: Document) -> str:
    """``doc.text`` with quarantined characters replaced by NUL (newlines kept for layout)."""
    chars = list(doc.text)
    for q_start, q_end in doc.quarantined:
        for i in range(max(q_start, 0), min(q_end, len(chars))):
            if chars[i] != "\n":
                chars[i] = "\x00"
    return "".join(chars)


def _line_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(text)))
    return spans


def _with_context(cands: Sequence[Span], evidence: Iterable[Span | None]) -> tuple[Span, ...]:
    """Put first the passage that states *every* part of a fact, when it is short enough.

    A relation often borrows a part from earlier text: the escalation source was named on the
    previous line, the minutes in a table header, the trigger a few lines up. The quote that
    backs the edge should contain all of them so the back-translation slot check can confirm
    it; the precise statement stays as the fallback candidate.
    """
    spans = [s for s in evidence if s is not None]
    if not cands or not spans:
        return tuple(cands)
    lead = cands[0]
    cover = (min(lead[0], *(s[0] for s in spans)), max(lead[1], *(s[1] for s in spans)))
    if cover == lead or cover[1] - cover[0] > _MAX_QUOTE:
        return tuple(cands)
    return _dedupe([cover, *cands])


def _code_lines(text: str, lines: Sequence[Span]) -> frozenset[int]:
    """Indices of fenced code-block lines (fences included). Their text is still read as prose
    ("kubectl rollout restart" is a real restart step) but "# comment" there is no heading and
    "|" is no table."""
    code: set[int] = set()
    inside = False
    for i, (start, end) in enumerate(lines):
        fence = _FENCE_RE.match(text, start, end) is not None
        if inside or fence:
            code.add(i)
        if fence:
            inside = not inside
    return frozenset(code)


def _sentences(text: str, start: int, end: int) -> list[Span]:
    out: list[Span] = []
    cur = start
    for m in _SENTENCE_SPLIT_RE.finditer(text, start, end):
        seg = _trim(text, cur, m.start())
        if seg[1] > seg[0]:
            out.append(seg)
        cur = m.end()
    seg = _trim(text, cur, end)
    if seg[1] > seg[0]:
        out.append(seg)
    return out


def _clean_label(text: str) -> str:
    """Display form for names built from text (trigger conditions, detection labels).

    Leading filler words ("만약", "일단") are dropped so equal conditions get equal names.
    """
    tokens = text.replace("*", " ").replace("`", " ").split()
    while len(tokens) > 1 and tokens[0] in _FILLER_WORDS:
        tokens.pop(0)
    return " ".join(tokens).strip(" ,.;:·-")


def _owner_matches(
    text: str, first_ordinals: Sequence[re.Match[str]], service: EntityRef | None
) -> dict[Span, re.Match[str]]:
    """Owner statements of a template keyed by the owner span (one statement per owner).

    A negative owner statement anywhere in the sentence ("담당자: 미정") suppresses them all:
    a sentence that says the owner is undecided is not evidence of an owner. A declaration
    whose 담당 is qualified by a task ("신고 담당: CISO", "DB 담당: 박과장") names who does that
    task, not who owns the service, so it is not an owner statement either.
    """
    if OWNER_NEGATIVE_RE.search(text) is not None:
        return {}
    owners: dict[Span, re.Match[str]] = {m.span("who"): m for m in first_ordinals}
    for rx in (OWNER_DECL_RE, FIRST_RESPONDER_RE, OWNER_SUBJECT_RE):
        for m in rx.finditer(text):
            if rx is OWNER_DECL_RE and not _owner_qualifier_ok(text, m.start(), service):
                continue
            if rx is OWNER_SUBJECT_RE and _task_topic(text, m.start("who")):
                continue  # "신고는 김CISO가 맡아요": 김CISO takes the report, not the service
            owners.setdefault(m.span("who"), m)
    return owners


def _task_topic(text: str, pos: int) -> bool:
    """Whether the clause before ``pos`` makes a task its topic ("신고는 …", "롤백은 …")."""
    start = max(text.rfind(ch, 0, pos) for ch in ",，;；") + 1
    return _TASK_TOPIC_RE.search(text, start, pos) is not None


#: Words that may stand right before 담당 in a *service* owner declaration ("결제 서비스
#: 담당자", "장애 대응 담당", "장애 시 담당자"). Any other word names a task or a component.
_OWNER_QUALIFIERS = frozenset(
    {
        "서비스", "시스템", "장애", "대응", "운영", "총괄", "전체",
        "1차", "주", "메인", "정", "시", "때", "경우",
    }
)  # fmt: skip
_QUALIFIER_BOUNDARY = frozenset(",;:/|·()[]{}<>-—–=.") | {S}
#: Endings that close a clause or mark a case ("…무시하고 담당자: X", "장애에서 담당: X"): the
#: word before 담당 is then no compound-noun qualifier of it.
_CLAUSE_ENDINGS: tuple[str, ...] = (
    "하고", "하며", "하면", "되면", "되고", "이고", "이며", "이면", "인데", "지만", "해서", "하여",
    "하되", "했고", "으며", "다", "요", "은", "는", "을", "를", "에서", "에는", "에게", "으로",
)  # fmt: skip


def _owner_qualifier_ok(text: str, start: int, service: EntityRef | None) -> bool:
    before = text[:start].rstrip()
    if before.endswith(("(", "（")):
        # "(담당: X)" qualifies the word before the parenthesis: "결제 서비스 (담당: X)" is the
        # service's owner, "KISA 신고(담당: X)" the reporter, "DB 작업(담당: X)" the DBA.
        before = before[:-1].rstrip()
        if not before or before[-1] in _QUALIFIER_BOUNDARY:
            return True
        return _owner_qualifier_word(before, service)
    if not before or before[-1] in _QUALIFIER_BOUNDARY:
        return True
    if before.rsplit(maxsplit=1)[-1].endswith(_CLAUSE_ENDINGS):
        return True
    return _owner_qualifier_word(before, service)


def _owner_qualifier_word(before: str, service: EntityRef | None) -> bool:
    """Whether the last word of ``before`` may qualify a *service* owner (see _OWNER_QUALIFIERS)."""
    token = before.rsplit(maxsplit=1)[-1]
    if token.endswith(("의", "쪽", "측")) and len(token) > 1:
        token = token[:-1]
    return (
        token in _OWNER_QUALIFIERS
        or token.endswith(("서비스", "시스템"))
        or (service is not None and (token == service.name or before.endswith(service.name)))
    )


#: Between two listed channels ("휴대폰, 이메일", "슬랙/메일").
_LIST_GAP_RE = re.compile(r"\s*(?:[,/·&]|및|와|과|또는|혹은|이나|나)?\s*")
#: "이메일로 연락", "메일로 공유해 주세요": the word is the means of contact.
_CHANNEL_USE_RE = re.compile(
    r"(?:으로|로)\s*(?:\S+\s+)?(?:연락|전화|알림|알려|공유|전달|발송|보내|통보|문의|호출)"
)
#: "연락처: 이메일", "연락 수단 = 메일".
_CHANNEL_FIELD_RE = re.compile(r"(?:연락\s*(?:처|수단|방법|채널)?|채널|수단)\s*[:：=]\s*$")
_PAREN_OPEN_RE = re.compile(r"\s*[(（]\s*")


def _as_channels(line: str, offset: int, mentions: list[Mention]) -> list[Mention]:
    """Read a catalog name that is also a channel word ("이메일" as a service) as the channel
    where the line uses it as one: listed next to another channel, in parentheses right after a
    person, after a contact field label, or followed by 로/으로 and a contact verb. Elsewhere
    ("## 이메일", "이메일 장애") the trusted catalog reading stands."""
    out = list(mentions)
    for i, m in enumerate(mentions):
        reading = channel_reading(m.name) if m.kind in (NodeKind.SERVICE, NodeKind.SYSTEM) else None
        if reading is None:
            continue
        start, end = m.start - offset, m.end - offset
        prev = mentions[i - 1] if i > 0 else None
        nxt = mentions[i + 1] if i + 1 < len(mentions) else None
        listed = (
            prev is not None
            and prev.kind is NodeKind.CHANNEL
            and _LIST_GAP_RE.fullmatch(line, prev.end - offset, start) is not None
        ) or (
            nxt is not None
            and nxt.kind is NodeKind.CHANNEL
            and _LIST_GAP_RE.fullmatch(line, end, nxt.start - offset) is not None
        )
        after_person = (
            prev is not None
            and prev.kind in _PEOPLE
            and _PAREN_OPEN_RE.fullmatch(line, prev.end - offset, start) is not None
        )
        used = _CHANNEL_USE_RE.match(line, end) is not None
        field_value = _CHANNEL_FIELD_RE.search(line, 0, start) is not None
        if listed or after_person or used or field_value:
            out[i] = Mention(NodeKind.CHANNEL, reading.name, m.start, m.end, medium=reading.medium)
    return out


_PAREN_TAIL_RE = re.compile(rf"[^()（）\n{P}{C}{A}{S}]{{0,20}}[)）]")


def _close_paren(text: str, start: int, end: int) -> int:
    """End of ``text[start:end]`` extended to close a parenthesis it opened ("실패율 알람(5% 초과
    3분 지속" -> "…지속)"), so a condition label and its quote read as written."""
    inner = text[start:end]
    if inner.count("(") + inner.count("（") <= inner.count(")") + inner.count("）"):
        return end
    close = _PAREN_TAIL_RE.match(text, end)
    return end if close is None else close.end()


def _after_placeholder(text: str, pos: int) -> bool:
    """Whether ``pos`` directly follows a mention placeholder (optionally with 이/가 and spaces)."""
    before = text[:pos].rstrip()
    if before.endswith(("이", "가")):
        before = before[:-1].rstrip()
    return bool(before) and before[-1] in (P, C, A)


@dataclass(frozen=True)
class _Template:
    """A segment with every mention replaced by one placeholder character.

    ``starts``/``ends`` map each template character back to its document range, so a regex
    match on the template converts to an exact document span.
    """

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    slots: dict[int, Mention]

    @classmethod
    def build(cls, work: str, seg: Span, mentions: Sequence[Mention]) -> _Template:
        chars: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        slots: dict[int, Mention] = {}

        def raw(upto: int, pos: int) -> int:
            while pos < upto:
                if work[pos] not in _TEMPLATE_SKIP:
                    chars.append(work[pos])
                    starts.append(pos)
                    ends.append(pos + 1)
                pos += 1
            return pos

        pos = seg[0]
        for m in mentions:
            pos = raw(m.start, pos)
            slots[len(chars)] = m
            chars.append(_PLACEHOLDER[m.kind])
            starts.append(m.start)
            ends.append(m.end)
            pos = m.end
        raw(seg[1], pos)
        return cls("".join(chars), tuple(starts), tuple(ends), slots)

    def span(self, ts: int, te: int) -> Span:
        return self.starts[ts], self.ends[te - 1]

    def mentions(self, ts: int, te: int) -> list[Mention]:
        return [self.slots[i] for i in range(ts, te) if i in self.slots]

    def group(self, match: re.Match[str], name: str) -> list[Mention]:
        start, end = match.span(name)
        return [] if start < 0 else self.mentions(start, end)

    def group_span(self, match: re.Match[str], name: str) -> Span | None:
        start, end = match.span(name)
        return None if start < 0 or end <= start else self.span(start, end)


@dataclass(frozen=True)
class _Unit:
    """Where a statement sits: its segment (sentence/cell), clauses and the whole line."""

    seg: Span
    line: Span
    clauses: tuple[Span, ...]

    @functools.cached_property
    def clause_starts(self) -> list[int]:
        return [c[0] for c in self.clauses]

    @functools.cached_property
    def clause_ends(self) -> list[int]:
        return [c[1] for c in self.clauses]


@dataclass(frozen=True)
class _TriggerHit:
    tspan: Span  # template span
    core: str  # condition label without the service ("장애 지속", "오류율 5% 이상")
    subject: Mention | None  # service named by the trigger itself


@dataclass(frozen=True)
class _DependencyHit:
    tspan: Span
    src: Mention | None  # None -> the scope's service
    dst: Mention


@dataclass
class _Contact:
    ref: EntityRef
    span: Span
    sentence: Span
    channel: EntityRef | None = None
    channel_span: Span | None = None


@dataclass
class _Trigger:
    ref: EntityRef
    span: Span
    sentence: Span
    #: contacts named in a sentence that states this trigger ("CPU 90% 초과 시 박과장에게 전화"),
    #: each with the trigger span of that statement. They are what the runbook says this trigger
    #: notifies, so they win over the scope's first contact.
    direct: list[tuple[_Contact, Span]] = field(default_factory=list)


@dataclass(frozen=True)
class _PendingOwner:
    """An owner read from "first contact in the scope" — emitted only if the scope never
    declares an owner explicitly ("담당: 박과장" anywhere in the section wins)."""

    service: EntityRef
    ref: EntityRef
    cands: tuple[Span, ...]


@dataclass(frozen=True)
class _PendingChannels:
    """A nameless contact declaration ("연락처: 슬랙") read before the scope named anyone."""

    channels: tuple[EntityRef, ...]
    cands: tuple[Span, ...]


@dataclass
class _Frame:
    """One level of the scope stack (document root, heading, bracket header, inline, row)."""

    level: int
    service: EntityRef | None = None
    owns_state: bool = False
    #: where the scope begins (heading/bracket/row line, or the sentence that opened it)
    start: int = 0
    #: a table row of a multi-service table: its (possibly missing) service is not inherited
    isolated: bool = False
    #: a one-line scope opened by a service label ("- 결제: … 알람 → 김대리 호출"): a labelled
    #: statement, so the person it calls is not taken for the owner (a guess made for prose only)
    structured: bool = False
    heading: Span | None = None
    owners: list[EntityRef] = field(default_factory=list)
    implicit_owners: list[_PendingOwner] = field(default_factory=list)
    pending_channels: list[_PendingChannels] = field(default_factory=list)
    contacts: list[_Contact] = field(default_factory=list)
    #: who the scope says receives its pages ("김대리가 슬랙으로 호출을 받는다"): what its
    #: triggers notify when their own sentence names no one
    paged: list[_Contact] = field(default_factory=list)
    last: list[tuple[EntityRef, Span | None]] = field(default_factory=list)
    triggers: list[_Trigger] = field(default_factory=list)
    primary: EntityRef | None = None
    actions: list[EntityRef] = field(default_factory=list)
    heading_actions: list[EntityRef] = field(default_factory=list)
    obligations: list[EntityRef] = field(default_factory=list)
    #: where each scope participant (owner, contact, channel) was first stated in this scope;
    #: lets a later statement quote the whole passage that supports it
    where: dict[EntityRef, Span] = field(default_factory=dict)

    def note(self, refs: Iterable[tuple[EntityRef, Span | None]]) -> None:
        for ref, span in refs:
            if span is not None:
                self.where.setdefault(ref, span)

    @property
    def declared_owner(self) -> bool:
        """Whether an owner was stated explicitly (not merely read from a first contact)."""
        return bool(self.owners) and not self.implicit_owners

    def declare_owners(self, refs: Iterable[EntityRef]) -> None:
        """Record explicitly stated owners; they replace any owner read from a first contact."""
        stated = list(refs)
        if not stated:
            return
        implicit = {p.ref for p in self.implicit_owners}
        self.owners = [o for o in self.owners if o not in implicit]
        self.implicit_owners = []
        self.owners += [r for r in dict.fromkeys(stated) if r not in self.owners]


@dataclass(frozen=True)
class _Row:
    """A markdown table data row with its cells grouped by column role."""

    line: Span
    cells: dict[str, list[tuple[int, Span]]]
    header: tuple[Span, ...]
    header_line: Span
    service: EntityRef | None


@dataclass(frozen=True)
class _LadderStep:
    """One row of an ordered contact table: who is called, how, and where it is written."""

    people: Sequence[tuple[EntityRef, Span]]
    channels: Sequence[tuple[EntityRef, Span]]
    cell: Span
    line: Span


# --------------------------------------------------------------------------- output sink


@dataclass
class _EntityAcc:
    quote: str
    attrs: dict[str, ExtractedAttr] = field(default_factory=dict)


class _Sink:
    """Collects entities/relations, grounding every quote before it is accepted.

    A quote candidate is accepted only if it does not touch quarantine; candidates whose first
    occurrence in the document is the place they were read from are preferred, because the
    assembler grounds quotes by searching the text.
    """

    def __init__(self, doc: Document) -> None:
        self._doc = doc
        self._text = doc.text
        self._entities: dict[tuple[NodeKind, str], _EntityAcc] = {}
        #: start of the scope whose facts are being emitted: repeated text before it belongs to
        #: another section, so a quote must not be grounded there (see :meth:`anchor`)
        self.floor = 0
        self._relations: dict[tuple[str, str, str, str, str, str], ExtractedRelation] = {}
        self._first: dict[Span, bool] = {}

    def anchor(self, cands: Sequence[Span], *, distinct: bool = False) -> str | None:
        """The quote for a fact, chosen from ``cands`` (most preferred first).

        ``distinct`` marks facts that depend on where they were read (relations, and actions,
        triggers and obligations named after their section's service): text that already
        occurred before the current scope — the same "| 담당 | 이수진 대리 |" row in an earlier
        section — is extended backwards until it occurs first where it was read, so the
        assembler does not ground it in another section. A repeat inside the same scope states
        the same fact and keeps its short quote.
        """
        trimmed = (_trim(self._text, s, e) for s, e in cands if 0 <= s < e <= len(self._text))
        usable = [(s, e) for s, e in trimmed if s < e and not self._doc.is_quarantined(s, e)]
        short = [(s, e) for s, e in usable if e - s <= _MAX_QUOTE]
        for s, e in short:
            if self._is_first(s, e):
                return self._text[s:e]
        # Repeated text: the most preferred clean candidate. The grounder picks the first
        # non-quarantined occurrence, which is identical text.
        if short:
            s, e = short[0]
            if distinct and self._text.find(self._text[s:e]) < self.floor:
                s, e = self._distinct(s, e)
            return self._text[s:e]
        if usable:
            s, e = min(usable, key=lambda c: c[1] - c[0])
            return self._text[s:e]
        return None

    def _distinct(self, start: int, end: int) -> Span:
        """Extend a repeated quote backwards, a line at a time, until its first occurrence is
        where it was read. Stops at quarantine and at the quote length limit, returning the
        quote unchanged when no clean, short enough extension is unique."""
        lo = start
        while lo > 0:
            line_start = self._text.rfind("\n", 0, lo) + 1
            lo = line_start if line_start < lo else self._text.rfind("\n", 0, lo - 1) + 1
            if end - lo > _MAX_QUOTE or self._doc.is_quarantined(lo, start):
                break
            extended = _trim(self._text, lo, end)
            if self._is_first(*extended):
                return extended
        return start, end

    def _is_first(self, start: int, end: int) -> bool:
        """Whether ``text[start:end]`` occurs nowhere earlier (cached: segments recur as candidates)."""
        hit = self._first.get((start, end))
        if hit is None:
            hit = self._first[(start, end)] = self._text.find(self._text[start:end]) == start
        return hit

    def entity(
        self, kind: NodeKind, name: str, cands: Sequence[Span], attrs: Iterable[_AttrSpec] = ()
    ) -> EntityRef | None:
        name = " ".join(name.split())
        if not name:
            return None
        acc = self._entities.get((kind, name))
        if acc is None:
            quote = self.anchor(cands, distinct=kind in _SCOPED_KINDS)
            if quote is None:
                return None
            acc = self._entities[(kind, name)] = _EntityAcc(quote)
        for spec in attrs:
            self._put_attr(acc.attrs, spec)
        return EntityRef(kind=kind, name=name)

    def attr(self, ref: EntityRef, spec: _AttrSpec) -> None:
        acc = self._entities.get((ref.kind, ref.name))
        if acc is not None:
            self._put_attr(acc.attrs, spec)

    def stated(self, kind: NodeKind, name: str, key: AttrKey) -> AttrValue | None:
        """The value already recorded for ``key`` of the entity ``(kind, name)``, if any."""
        acc = self._entities.get((kind, " ".join(name.split())))
        found = acc.attrs.get(key.value) if acc is not None else None
        return None if found is None else found.value

    def relation(
        self,
        kind: EdgeKind,
        src: EntityRef,
        dst: EntityRef,
        cands: Sequence[Span],
        attrs: Iterable[_AttrSpec] = (),
    ) -> None:
        src_kinds, dst_kinds = EDGE_SIGNATURES[kind]
        if src == dst or src.kind not in src_kinds or dst.kind not in dst_kinds:
            return
        quote = self.anchor(cands, distinct=True)
        if quote is None:
            return
        key = (kind.value, src.kind.value, src.name, dst.kind.value, dst.name, quote)
        merged: dict[str, ExtractedAttr] = {}
        if key in self._relations:
            merged = {a.key: a for a in self._relations[key].attrs}
        for spec in attrs:
            self._put_attr(merged, spec)
        self._relations[key] = ExtractedRelation(
            kind=kind, src=src, dst=dst, quote=quote, attrs=tuple(merged[k] for k in sorted(merged))
        )

    def _put_attr(self, attrs: dict[str, ExtractedAttr], spec: _AttrSpec) -> None:
        if spec.key.value in attrs:
            return  # first statement wins; later contradictions are the checker's business
        quote = self.anchor(spec.cands)
        if quote is not None:
            attrs[spec.key.value] = ExtractedAttr(key=spec.key.value, value=spec.value, quote=quote)

    def build(self) -> tuple[tuple[ExtractedEntity, ...], tuple[ExtractedRelation, ...]]:
        entities = tuple(
            ExtractedEntity(
                kind=kind, name=name, quote=acc.quote, attrs=tuple(acc.attrs[k] for k in sorted(acc.attrs))
            )
            for (kind, name), acc in sorted(self._entities.items(), key=lambda kv: (kv[0][0].value, kv[0][1]))
        )
        relations = tuple(self._relations[k] for k in sorted(self._relations))
        return entities, relations


# --------------------------------------------------------------------------- public API


class RuleExtractor:
    """Korean pattern-based extractor (LLM 0 calls). Same IR as the LLM extractor.

    >>> from jettae.domain import Document
    >>> doc = Document(doc_id="d", title="t", text="결제 장애가 계속되면 김대리에게 연락, 안 되면 롤백")
    >>> sorted(r.kind.value for r in RuleExtractor().extract(doc).relations)
    ['detected_by', 'mitigated_by', 'notifies', 'owned_by']
    """

    extractor: ClassVar[Extractor] = Extractor.RULE

    def extract(
        self,
        doc: Document,
        *,
        roster: Iterable[RosterEntry] = (),
        catalog: Iterable[CatalogService] = (),
    ) -> Extraction:
        walker = _Walker(doc, Lexicon(roster, catalog))
        walker.run()
        entities, relations = walker.sink.build()
        return Extraction(doc_id=doc.doc_id, extractor=Extractor.RULE, entities=entities, relations=relations)


# --------------------------------------------------------------------------- the walker


class _Walker:
    def __init__(self, doc: Document, lexicon: Lexicon) -> None:
        self.doc = doc
        self.lex = lexicon
        self.work = _working_text(doc)
        self.sink = _Sink(doc)
        self.lines = _line_spans(self.work)
        self._line_starts = [s for s, _ in self.lines]
        self._code = _code_lines(self.work, self.lines)
        #: line index -> (mention starts, mentions); lines are scanned once
        self._mention_cache: dict[int, tuple[list[int], list[Mention]]] = {}
        self.stack: list[_Frame] = [_Frame(level=0, owns_state=True)]
        #: triggers stated by the sentence being read (with that statement's span): a contact in
        #: the same sentence is what the trigger notifies
        self._live: list[tuple[_Trigger, Span]] = []
        #: the statement being read is a labelled field ("감지: … → 김대리 호출", a field-table
        #: cell): a contact there is never taken for the scope's owner
        self._in_field = False

    # ------------------------------------------------------------------ driving
    def run(self) -> None:
        title = self._title_index()
        i, n = 0, len(self.lines)
        while i < n:
            start, end = self.lines[i]
            line = self.work[start:end]
            in_code = i in self._code
            if (in_code and _FENCE_RE.match(line)) or not line.strip(_TRIM_CHARS):
                if not in_code:
                    self._pop(lambda f: f.level >= _INLINE_LEVEL)
                i += 1
                continue
            heading = None if in_code else _HEADING_RE.fullmatch(line)
            if heading is not None:
                self._heading(heading, start, is_title=i == title)
                i += 1
                continue
            if not in_code and self._is_table_start(i):
                j = i + 2
                while j < n and _TABLE_ROW_RE.fullmatch(self.work[self.lines[j][0] : self.lines[j][1]]):
                    j += 1
                self._table(self.lines[i:j])
                i = j
                continue
            self._prose(start, end)
            i += 1
        self._pop(lambda f: True)

    def _title_index(self) -> int:
        """A lone leading H1 is the document title, not a section (its content still counts)."""
        levels = [
            (i, len(m.group("hashes")))
            for i, (s, e) in enumerate(self.lines)
            if i not in self._code and (m := _HEADING_RE.fullmatch(self.work[s:e])) is not None
        ]
        if levels and levels[0][1] == 1 and sum(1 for _, lv in levels if lv == 1) == 1:
            return levels[0][0]
        return -1

    def _is_table_start(self, i: int) -> bool:
        if i + 1 >= len(self.lines):
            return False
        row = self.work[self.lines[i][0] : self.lines[i][1]]
        sep = self.work[self.lines[i + 1][0] : self.lines[i + 1][1]]
        return _TABLE_ROW_RE.fullmatch(row) is not None and _TABLE_SEP_RE.fullmatch(sep) is not None

    # ------------------------------------------------------------------ scope stack
    def _push(self, frame: _Frame) -> None:
        self.stack.append(frame)
        self.sink.floor = self._state().start

    def _pop(self, pred: Callable[[_Frame], bool]) -> None:
        while self.stack and pred(self.stack[-1]):
            frame = self.stack.pop()
            if frame.owns_state:
                self.sink.floor = frame.start  # its facts are quoted from inside it
                self._finalize(frame)
        self.sink.floor = self._state().start if self.stack else 0

    def _state(self) -> _Frame:
        return next(f for f in reversed(self.stack) if f.owns_state)

    def _service(self) -> EntityRef | None:
        for frame in reversed(self.stack):
            if frame.service is not None or frame.isolated:
                return frame.service
        return None

    def _heading_actions(self) -> list[EntityRef]:
        for frame in reversed(self.stack):
            if frame.heading_actions:
                return frame.heading_actions
        return []

    def _owner_cands(self, service: EntityRef, cands: Sequence[Span]) -> tuple[Span, ...]:
        """Quote candidates for ``service`` OWNED_BY someone.

        A service named by a channel word ("이메일") is itself a slot that the back-translation
        check compares (the rendering "이메일의 담당은 …" names a medium), so its owner statement
        is quoted together with the place the scope names the service, when that is near.
        """
        if not names_channel(service.name):
            return tuple(cands)
        named = next((f.where[service] for f in reversed(self.stack) if service in f.where), None)
        return _with_context(cands, [named])

    def _finalize(self, frame: _Frame) -> None:
        """Close a scope: emit the facts that needed the whole section to be read.

        * owners read from the first contact, unless the section declared an owner explicitly;
        * nameless contact declarations made before anyone was named, attached to the owner;
        * NOTIFIES for each trigger: the contacts its own sentence names, else who the scope
          says receives its pages, else the scope's first contact (with the stated channel).
        """
        for pending in frame.implicit_owners:
            self.sink.relation(EdgeKind.OWNED_BY, pending.service, pending.ref, pending.cands)
        targets = frame.owners or [c.ref for c in frame.contacts]
        for decl in frame.pending_channels:
            for target in targets:
                cands = _with_context(decl.cands, [frame.where.get(target)])
                for channel in decl.channels:
                    self.sink.relation(EdgeKind.REACHABLE_VIA, target, channel, cands)
        for trig in frame.triggers:
            pairs = trig.direct or [(contact, trig.span) for contact in frame.paged or frame.contacts]
            for contact, trigger_span in pairs:
                self._notify(trig.ref, trigger_span, contact)

    def _notify(self, trigger: EntityRef, trigger_span: Span, contact: _Contact) -> None:
        attrs: list[_AttrSpec] = []
        if contact.channel is not None:
            # evidence must show the channel: the contact clause only if it contains it
            ch_span = contact.channel_span
            covering = ch_span is not None and contact.span[0] <= ch_span[0] and ch_span[1] <= contact.span[1]
            evidence = _dedupe([ch_span, contact.span if covering else None, contact.sentence])
            attrs.append(_AttrSpec(AttrKey.CHANNEL, contact.channel.name, evidence))
        cands = _with_context(_dedupe([contact.span, contact.sentence]), [trigger_span, contact.channel_span])
        self.sink.relation(EdgeKind.NOTIFIES, trigger, contact.ref, cands, attrs)

    # ------------------------------------------------------------------ layout
    def _heading(self, m: re.Match[str], line_start: int, *, is_title: bool) -> None:
        level = len(m.group("hashes"))
        body = _trim(self.work, line_start + m.start("body"), line_start + m.end("body"))
        line = _trim(self.work, line_start, line_start + m.end())
        label = service_in_label(
            self.work[body[0] : body[1]], self.lex, offset=body[0], incident_suffix=not is_title
        )
        if is_title and label is None:
            for seg in _sentences(self.work, *body):
                self._sentence(seg, line)
            return
        self._pop(lambda f: f.level >= level)
        frame = _Frame(level=level, heading=body, start=line[0])
        if label is not None:
            frame.service = self._mention_ref(label, _dedupe([(label.start, label.end), body, line]))
        if frame.service is not None:
            frame.note([(frame.service, (label.start, label.end))] if label is not None else [])
        frame.owns_state = frame.service is not None or len(self.stack) == 1
        self._push(frame)
        for seg in _sentences(self.work, *body):
            self._sentence(seg, line, heading=True)

    def _prose(self, start: int, end: int) -> None:
        start, end = _trim(self.work, start, end)
        line = (start, end)
        marker = _MARKER_RE.match(self.work, start, end)
        start = marker.end() if marker is not None else start
        bracket = _BRACKET_RE.match(self.work, start, end)
        if bracket is not None:
            inner = (start + 1, bracket.end() - 1)
            label = service_in_label(self.work[inner[0] : inner[1]], self.lex, offset=inner[0])
            if label is not None:
                self._pop(lambda f: f.level >= _BRACKET_LEVEL)
                ref = self._mention_ref(label, _dedupe([(label.start, label.end), inner, line]))
                if ref is not None:
                    frame = _Frame(
                        level=_BRACKET_LEVEL, service=ref, owns_state=True, heading=inner, start=line[0]
                    )
                    frame.note([(ref, (label.start, label.end))])
                    self._push(frame)
                start = bracket.end()
        scoped = field_line = False
        labelled = _LABEL_RE.match(self.work, start, end)
        if labelled is not None:
            scoped = self._label_scope(labelled, line)
            start = labelled.end() if scoped else start
            field_line = not scoped and _header_role(labelled.group("label")) in _FIELD_ROLES
        self._in_field = field_line or scoped
        try:
            for seg in _sentences(self.work, start, end):
                self._sentence(seg, line)
        finally:
            self._in_field = False
            if scoped:
                self._pop(lambda f: f.level >= _INLINE_LEVEL)

    def _label_scope(self, labelled: re.Match[str], line: Span) -> bool:
        """A list line led by a service name ("- 결제: 실패율 5% 초과 시 알람 → 김대리 호출")
        states a fact of that service, wherever the line sits (a "## 감지 기준" section lists one
        line per service). Opens a scope for that line only; returns whether it did."""
        body = _trim(self.work, *labelled.span("label"), _CELL_TRIM_CHARS)
        text = self.work[body[0] : body[1]]
        found = service_in_label(text, self.lex, offset=body[0], incident_suffix=False)
        if found is None or found.start != body[0]:
            return False
        if _LABEL_TAIL_RE.fullmatch(self.work, found.end, body[1]) is None:
            return False
        current = self._service()
        if current is not None and current.name == found.name:
            return False  # the line restates its own section's service: nothing to scope
        ref = self._mention_ref(found, _dedupe([(found.start, found.end), body, line]))
        if ref is None:
            return False
        self._pop(lambda f: f.level >= _INLINE_LEVEL)
        frame = _Frame(
            level=_INLINE_LEVEL, service=ref, owns_state=True, isolated=True, structured=True, start=line[0]
        )
        frame.note([(ref, (found.start, found.end))])
        self._push(frame)
        return True

    # ------------------------------------------------------------------ mentions & spans
    def _text(self, span: Span) -> str:
        return self.doc.text[span[0] : span[1]]

    def _mentions(self, seg: Span) -> list[Mention]:
        idx = bisect.bisect_right(self._line_starts, seg[0]) - 1
        line_start, line_end = self.lines[idx]
        cached = self._mention_cache.get(idx)
        if cached is None:
            text = self.work[line_start:line_end]
            found = _as_channels(text, line_start, find_mentions(text, self.lex, offset=line_start))
            cached = self._mention_cache[idx] = ([m.start for m in found], found)
        starts, mentions = cached
        lo, hi = bisect.bisect_left(starts, seg[0]), bisect.bisect_left(starts, seg[1])
        return [m for m in mentions[lo:hi] if m.end <= seg[1]]

    def _unit(self, seg: Span, line: Span) -> _Unit:
        clauses: list[Span] = []
        cur = seg[0]
        for m in _CLAUSE_SPLIT_RE.finditer(self.work, seg[0], seg[1]):
            clause = _trim(self.work, cur, m.start())
            if clause[1] > clause[0]:
                clauses.append(clause)
            cur = m.end()
        last = _trim(self.work, cur, seg[1])
        if last[1] > last[0]:
            clauses.append(last)
        return _Unit(seg, line, tuple(clauses))

    def _word_span(self, span: Span, unit: _Unit) -> Span:
        start, end = span
        while start > unit.seg[0] and self.work[start - 1] not in _BOUNDARY_CHARS:
            start -= 1
        while end < unit.seg[1] and self.work[end] not in _BOUNDARY_CHARS:
            end += 1
        return _trim(self.work, start, end, _CELL_TRIM_CHARS)

    def _clause(self, span: Span, unit: _Unit) -> Span:
        """The clause(s) of the segment that ``span`` touches (clauses are sorted, disjoint)."""
        first = bisect.bisect_right(unit.clause_ends, span[0])
        last = bisect.bisect_left(unit.clause_starts, span[1])
        hits = unit.clauses[first:last]
        return (hits[0][0], hits[-1][1]) if hits else span

    def _cands(self, span: Span, unit: _Unit) -> tuple[Span, ...]:
        return _dedupe([self._word_span(span, unit), self._clause(span, unit), unit.seg, unit.line])

    def _mention_cands(self, m: Mention, unit: _Unit) -> tuple[Span, ...]:
        span = (m.start, m.end)
        return _dedupe([span, self._clause(span, unit), unit.seg, unit.line])

    def _mention_ref(self, m: Mention, cands: Sequence[Span]) -> EntityRef | None:
        attrs: list[_AttrSpec] = []
        own = ((m.start, m.end),)
        if m.kind is NodeKind.CHANNEL and m.medium is not None:
            attrs.append(_AttrSpec(AttrKey.MEDIUM, m.medium.value, own))
        if m.kind is NodeKind.PERSON and m.title is not None:
            attrs.append(_AttrSpec(AttrKey.TITLE, m.title, own))
        return self.sink.entity(m.kind, m.name, cands, attrs)

    def _refs(self, mentions: Iterable[Mention], unit: _Unit) -> list[tuple[EntityRef, Span]]:
        out: list[tuple[EntityRef, Span]] = []
        for m in mentions:
            ref = self._mention_ref(m, self._mention_cands(m, unit))
            if ref is not None and all(ref != r for r, _ in out):
                out.append((ref, (m.start, m.end)))
        return out

    def _action_ref(self, m: Mention, service: EntityRef | None, cands: Sequence[Span]) -> EntityRef | None:
        """Actions are named per service ("결제 롤백") so two services' rollbacks never merge."""
        name = m.name if service is None else f"{service.name} {m.name}"
        attrs = []
        if m.action is not None:
            attrs.append(_AttrSpec(AttrKey.ACTION_TYPE, m.action.value, ((m.start, m.end),)))
        return self.sink.entity(NodeKind.ACTION, name, cands, attrs)

    # ------------------------------------------------------------------ one sentence / cell
    def _sentence(self, seg: Span, line: Span, *, heading: bool = False) -> list[EntityRef]:
        """Read one statement; returns the (non-negated) actions it mentions."""
        t = _Template.build(self.work, seg, self._mentions(seg))
        unit = self._unit(seg, line)
        dependencies = self._dependency_hits(t)
        dep_spans = [h.tspan for h in dependencies]
        triggers = [h for h in self._trigger_hits(t) if not _overlaps(h.tspan, dep_spans)]
        self._open_inline_scope(triggers, unit, heading=heading)
        service = self._service()
        state = self._state()
        self._live = []
        for hit in triggers:
            stated = self._emit_trigger(t.span(*hit.tspan), hit.core, service, state, unit)
            if stated is not None:
                self._live.append(stated)
        for dep in dependencies:
            self._emit_dependency(dep, t, unit, service)
        self._response_flow(t, unit, service, state)
        actions = self._emit_actions(t, unit, service, state, heading=heading)
        self._approvals(t, unit, actions)
        self._executors(t, unit, actions)
        self._reachability(t, unit, state)
        self._fallbacks(t, unit, state)
        self._obligations(t, unit, service, state)
        self._holds_role(t, unit)
        return [ref for _, ref in actions]

    # ------------------------------------------------------------------ triggers & scope
    def _trigger_hits(self, t: _Template) -> list[_TriggerHit]:
        hits: list[_TriggerHit] = []
        for rx, suffix in ((PERSIST_RE, "지속"), (OCCUR_RE, "발생")):
            for m in rx.finditer(t.text):
                subject = self._subject(t, m)
                if subject is None and _after_placeholder(t.text, m.start("kind")):
                    continue  # "슬랙 장애 시 전화" is a channel outage, not a service trigger
                start = m.start() if subject is not None else m.start("kind")
                hits.append(_TriggerHit((start, m.end()), f"{m.group('kind')} {suffix}", subject))
        for m in ALERT_RE.finditer(t.text):
            named = t.group(m, "svc")
            start = m.start() if named else m.start("t")
            end = _close_paren(t.text, m.start("t"), m.end())
            core = _clean_label(self._text(t.span(m.start("t"), end)))
            hits.append(_TriggerHit((start, end), core, named[0] if named else None))
        hits += self._alarm_conditions(t)
        for m in DETECT_DECL_RE.finditer(t.text):
            v_start, v_end = m.span("v")
            while v_start < v_end and t.text[v_start].isspace():
                v_start += 1
            while v_end > v_start and t.text[v_end - 1] in " .,;:→>-=":
                v_end -= 1  # "감지: 주문 실패율 알람 → 정민혁 과장 호출": the arrow starts the response
            if v_end - v_start < 2:
                continue
            v_end = _close_paren(t.text, v_start, v_end)
            core = _clean_label(self._text(t.span(v_start, v_end)))
            if core:
                hits.append(_TriggerHit((v_start, v_end), core, None))
        chosen: list[_TriggerHit] = []
        for hit in sorted(hits, key=lambda h: (h.tspan[0], -(h.tspan[1] - h.tspan[0]))):
            if not _overlaps(hit.tspan, [c.tspan for c in chosen]):
                chosen.append(hit)
        return chosen

    def _alarm_conditions(self, t: _Template) -> list[_TriggerHit]:
        """Alarms that go off under a stated condition (``ALARM_COND_RE``); the condition is the
        trigger's label, and a service it starts with is the trigger's own subject."""
        hits: list[_TriggerHit] = []
        for m in ALARM_COND_RE.finditer(t.text):
            group = next(g for g in ("t1", "t2", "t3") if m.group(g) is not None)
            c_start, c_end = m.span(group)
            core = _clean_label(self._text(t.span(c_start, c_end)))
            if not core:
                continue
            lead = t.slots.get(c_start)
            subject = lead if lead is not None and lead.kind in (NodeKind.SERVICE, NodeKind.SYSTEM) else None
            hits.append(_TriggerHit(m.span(), core, subject))
        return hits

    def _subject(self, t: _Template, m: re.Match[str]) -> Mention | None:
        svc = t.group(m, "svc")
        if svc:
            return svc[0]
        noun_span = t.group_span(m, "noun")
        if noun_span is None:
            return None
        noun = self.work[noun_span[0] : noun_span[1]]
        if not is_service_noun(noun, self.lex):
            return None
        return Mention(NodeKind.SERVICE, noun, noun_span[0], noun_span[1])

    def _open_inline_scope(self, triggers: Sequence[_TriggerHit], unit: _Unit, *, heading: bool) -> None:
        """A trigger that names its service ("결제 장애가 계속되면") scopes the rest of the
        paragraph when no heading did. Never overrides a service set by structure."""
        if self._service() is not None or self.stack[-1].isolated:
            return
        subject = next((h.subject for h in triggers if h.subject is not None), None)
        if subject is None:
            return
        ref = self._mention_ref(subject, self._mention_cands(subject, unit))
        if ref is None:
            return
        top = self.stack[-1]
        if heading or top.level == _ROW_LEVEL:
            top.service = ref
            top.owns_state = True
        else:
            self._pop(lambda f: f.level >= _INLINE_LEVEL)
            top = _Frame(level=_INLINE_LEVEL, service=ref, owns_state=True, start=unit.seg[0])
            self._push(top)
        top.note([(ref, (subject.start, subject.end))])

    def _emit_trigger(
        self, span: Span, core: str, service: EntityRef | None, state: _Frame, unit: _Unit
    ) -> tuple[_Trigger, Span] | None:
        """Emit a trigger (+ DETECTED_BY) and register it in the scope; returns the scope's
        record of it with the span of this statement, or ``None`` when it could not be quoted."""
        span = _trim(self.work, *span)
        name = core if service is None or service.name in core else f"{service.name} {core}"
        attrs = [_AttrSpec(AttrKey.CONDITION, self.doc.text[span[0] : span[1]], (span,))]
        delay = find_delay(self.work[span[0] : span[1]], span[0])
        if delay is not None:
            attrs.append(_AttrSpec(AttrKey.DELAY_MINUTES, delay.minutes, ((delay.start, delay.end), span)))
        cands = self._cands(span, unit)
        ref = self.sink.entity(NodeKind.TRIGGER, name, cands, attrs)
        if ref is None:
            return None
        if service is not None:
            self.sink.relation(EdgeKind.DETECTED_BY, service, ref, cands)
        word = self._word_span(span, unit)
        record = next((trig for trig in state.triggers if trig.ref == ref), None)
        if record is None:  # the first statement of a trigger is its scope-level evidence
            record = _Trigger(ref, word, unit.seg)
            state.triggers.append(record)
        return record, word

    # ------------------------------------------------------------------ dependencies
    def _dependency_hits(self, t: _Template) -> list[_DependencyHit]:
        hits: list[_DependencyHit] = []
        for rx in (DEPENDS_RE, DEPENDS_IMPACT_RE, DEPENDS_SUBJECTLESS_RE):
            for m in rx.finditer(t.text):
                if _overlaps(m.span(), [h.tspan for h in hits]):
                    continue
                dst = self._dependency_end(t, m, "x", "xn", target=True)
                src = self._dependency_end(t, m, "s", "sn", target=False) if "s" in rx.groupindex else None
                if dst is None or ("s" in rx.groupindex and src is None):
                    continue
                hits.append(_DependencyHit(m.span(), src, dst))
        return hits

    def _dependency_end(
        self, t: _Template, m: re.Match[str], ph: str, noun: str, *, target: bool
    ) -> Mention | None:
        mentioned = t.group(m, ph)
        if mentioned:
            return mentioned[0]
        span = t.group_span(m, noun)
        if span is None:
            return None
        text = " ".join(self.work[span[0] : span[1]].split())
        if not all(is_service_noun(token, self.lex) for token in text.split(" ")):
            return None
        kind = NodeKind.SERVICE if (not target or m.group("xs")) else NodeKind.SYSTEM
        return Mention(kind, text, span[0], span[1])

    def _emit_dependency(
        self, hit: _DependencyHit, t: _Template, unit: _Unit, service: EntityRef | None
    ) -> None:
        span = t.span(*hit.tspan)
        cands = self._cands(span, unit)
        src = service if hit.src is None else self._mention_ref(hit.src, self._mention_cands(hit.src, unit))
        dst = self._mention_ref(hit.dst, self._mention_cands(hit.dst, unit))
        if src is not None and dst is not None:
            self.sink.relation(EdgeKind.DEPENDS_ON, src, dst, cands)

    # ------------------------------------------------------------------ owner / contact / escalation
    def _response_flow(self, t: _Template, unit: _Unit, service: EntityRef | None, state: _Frame) -> None:
        """Owner declarations, contacts and escalations, applied in reading order because the
        escalation source is "whoever was contacted last" at that point of the sentence."""
        # "슬랙이 안 되면 박팀장에게 전화" is a channel outage: 박팀장 is the next contact, not an
        # escalation after someone failed to answer (that would be a guess).
        channel_down = [m.span() for m in CHANNEL_DOWN_RE.finditer(t.text)]
        escalations = [
            m for m in ESCALATE_RE.finditer(t.text) if not any(s <= m.start() < e for s, e in channel_down)
        ]
        esc_spans = [m.span() for m in escalations]
        events: list[tuple[int, int, Callable[[], None]]] = [
            (m.start(), 1, functools.partial(self._apply_escalation, m, t, unit, state)) for m in escalations
        ]
        contacts: dict[Span, list[re.Match[str]]] = {}
        prohibited: set[Span] = set()
        for rx in (CONTACT_RE, CONTACT_VIA_FIRST_RE, CALL_RE, CALL_PAREN_RE, CONTACT_PASSIVE_RE):
            for m in rx.finditer(t.text):
                if _overlaps(m.span("who"), esc_spans):
                    continue
                contacts.setdefault(m.span("who"), []).append(m)
                if PROHIBITED_AFTER_RE.match(t.text, m.end()):
                    prohibited.add(m.span("who"))  # "김대리에게 연락하지 마세요" is no contact
        for who, ms in contacts.items():
            if who not in prohibited:
                events.append(
                    (who[0], 2, functools.partial(self._apply_contact, ms, t, unit, service, state))
                )
        ordinals = list(ORDINAL_RE.finditer(t.text))
        events += [
            (m.start(), 1, functools.partial(self._apply_ordinal, m, t, unit, state))
            for m in ordinals
            if m.group("ord") != "1"
        ]
        first_ordinals = [m for m in ordinals if m.group("ord") == "1"]
        for who, m in _owner_matches(t.text, first_ordinals, service).items():
            events.append((who[0], 0, functools.partial(self._apply_owner, m, t, unit, service, state)))
        for _, _, apply in sorted(events, key=lambda e: (e[0], e[1])):
            apply()

    def _apply_owner(
        self, m: re.Match[str], t: _Template, unit: _Unit, service: EntityRef | None, state: _Frame
    ) -> None:
        cands = self._cands(t.span(*m.span()), unit)
        refs = self._refs(t.group(m, "who"), unit)
        for ref, _ in refs:
            if service is not None:
                self.sink.relation(EdgeKind.OWNED_BY, service, ref, self._owner_cands(service, cands))
        state.declare_owners(ref for ref, _ in refs)
        state.note(refs)
        if not state.last:
            state.last = [(ref, span) for ref, span in refs]

    def _apply_contact(
        self, ms: Sequence[re.Match[str]], t: _Template, unit: _Unit, service: EntityRef | None, state: _Frame
    ) -> None:
        span = self._word_span(t.span(min(m.start() for m in ms), max(m.end() for m in ms)), unit)
        cands = self._cands(span, unit)
        people = self._refs(t.group(ms[0], "who"), unit)
        vias: list[Mention] = []
        for m in ms:
            vias += [v for v in t.group(m, "via") if v not in vias] if "via" in m.re.groupindex else []
        channels = self._refs(vias, unit)
        primary, primary_span = channels[0] if channels else (None, None)
        # "김대리가 슬랙으로 호출을 받는다" states where the page goes, not whom the runbook tells
        # you to contact: it neither takes the first-contact slot nor seeds an owner.
        paged = all(m.re is CONTACT_PASSIVE_RE for m in ms)
        first = not state.contacts and not paged
        for ref, _ in people:
            contact = _Contact(ref, span, unit.seg, primary, primary_span)
            for channel, _ in channels:
                self.sink.relation(EdgeKind.REACHABLE_VIA, ref, channel, cands)
            if first:
                state.contacts.append(contact)
            if paged and all(c.ref != ref for c in state.paged):
                state.paged.append(contact)
            for trig, trigger_span in self._live:
                if all(c.ref != ref for c, _ in trig.direct):
                    trig.direct.append((contact, trigger_span))
        if first and service is not None and not state.owners and not (self._in_field or state.structured):
            # The first contact owns the scope unless the section declares an owner (decided
            # when the scope closes). A team is an owner only when written as one ("담당: SRE팀").
            # Prose only: a labelled field names the owner in its own field when there is one.
            for ref, _ in people:
                if ref.kind is not NodeKind.TEAM:
                    state.implicit_owners.append(
                        _PendingOwner(service, ref, self._owner_cands(service, cands))
                    )
                    state.owners.append(ref)
        if channels and state.primary is None:
            state.primary = channels[0][0]
        state.note([*people, *channels])
        if people:
            state.last = list(people)

    def _apply_escalation(
        self,
        m: re.Match[str],
        t: _Template,
        unit: _Unit,
        state: _Frame,
        default: tuple[int, Span] | None = None,
    ) -> None:
        span = self._word_span(t.span(*m.span()), unit)
        minutes_group = "m1" if m.group("m1") else "m2" if m.group("m2") else None
        attrs: list[_AttrSpec] = []
        minutes_span: Span | None = None
        if minutes_group is not None:
            value = parse_minutes(m.group(minutes_group))
            minutes_span = t.group_span(m, minutes_group)
            if value is not None:
                attrs.append(_AttrSpec(AttrKey.AFTER_MINUTES, value, _dedupe([minutes_span, span])))
        elif default is not None:
            minutes_span = default[1]
            attrs.append(_AttrSpec(AttrKey.AFTER_MINUTES, default[0], (default[1],)))
        explicit: list[tuple[EntityRef, Span | None]] = [
            (r, s) for r, s in self._refs(t.group(m, "src"), unit)
        ]
        targets = self._refs(t.group(m, "dst"), unit)
        self._escalate(explicit, targets, span, unit, state=state, attrs=attrs, evidence=[minutes_span])
        for dst, _ in targets:
            for channel, _ in self._refs(t.group(m, "via"), unit):
                self.sink.relation(EdgeKind.REACHABLE_VIA, dst, channel, self._cands(span, unit))

    def _apply_ordinal(self, m: re.Match[str], t: _Template, unit: _Unit, state: _Frame) -> None:
        """Next step of a numbered contact order ("2차: 박팀장", "최종: CTO"); no time is stated."""
        span = self._word_span(t.span(*m.span()), unit)
        self._escalate([], self._refs(t.group(m, "who"), unit), span, unit, state=state, attrs=())

    def _escalate(
        self,
        explicit: Sequence[tuple[EntityRef, Span | None]],
        targets: Sequence[tuple[EntityRef, Span]],
        span: Span,
        unit: _Unit,
        *,
        state: _Frame,
        attrs: Sequence[_AttrSpec],
        evidence: Sequence[Span | None] = (),
    ) -> None:
        """ESCALATES_TO from the stated source, else from whoever was contacted last (or the
        owner). Without any source the step is remembered but no edge is invented."""
        sources = list(explicit) or state.last or [(o, state.where.get(o)) for o in state.owners]
        for src, src_span in sources:
            cands = _with_context(self._cands(span, unit), [src_span, *evidence])
            for dst, _ in targets:
                self.sink.relation(EdgeKind.ESCALATES_TO, src, dst, cands, attrs)
        state.note(targets)
        if targets:
            state.last = [(r, s) for r, s in targets]

    # ------------------------------------------------------------------ actions
    def _emit_actions(
        self, t: _Template, unit: _Unit, service: EntityRef | None, state: _Frame, *, heading: bool
    ) -> list[tuple[Mention, EntityRef]]:
        out: list[tuple[Mention, EntityRef]] = []
        for idx in sorted(t.slots):
            m = t.slots[idx]
            if m.kind is not NodeKind.ACTION:
                continue
            if NEGATED_ACTION_RE.match(t.text, idx) and not _UNAPPROVED_BEFORE_RE.search(t.text, 0, idx):
                continue  # "롤백 불가" — but "승인 없이 롤백 금지" only forbids the unapproved one
            ref = self._action_ref(m, service, self._mention_cands(m, unit))
            if ref is None:
                continue
            out.append((m, ref))
            if service is not None:
                clause = self._clause((m.start, m.end), unit)
                self.sink.relation(
                    EdgeKind.MITIGATED_BY, service, ref, _dedupe([clause, unit.seg, unit.line])
                )
            if ref not in state.actions:
                state.actions.append(ref)
            if heading and ref not in self.stack[-1].heading_actions:
                self.stack[-1].heading_actions.append(ref)
        return out

    def _action_targets(
        self, acts: Sequence[Mention], span: Span, unit: _Unit, actions: Sequence[tuple[Mention, EntityRef]]
    ) -> list[EntityRef]:
        """Which actions an approval/executor statement is about: the ones it names, else the
        ones in its clause, else the sentence's, else the section heading's (e.g. "### 롤백"),
        else — only when the clause says so generically ("조치는 팀장 승인 후") — the actions
        already listed in this scope."""
        if acts:
            return [ref for m, ref in actions if m in acts]
        clause = self._clause(span, unit)
        in_clause = [ref for m, ref in actions if clause[0] <= m.start and m.end <= clause[1]]
        found = in_clause or [ref for _, ref in actions] or list(self._heading_actions())
        if not found and _GENERIC_ACTION_RE.search(self.work, clause[0], clause[1]):
            found = list(self._state().actions)
        return found

    def _approval_polarity(self, t: _Template, pos: int, unit: _Unit) -> tuple[bool, int]:
        """Polarity of the approval word ending at template index ``pos``, and where the
        statement of that polarity ends in the document.

        "승인 후" requires approval; a waiver right after it ("승인 불필요", "승인 없이 가능",
        "승인을 받지 않아도") does not — unless the rest of the clause prohibits acting that way
        ("승인 없이 진행 금지"), which is a requirement stated twice negatively. When a waiver is
        read, the statement runs to the end of its clause so the quote carries both negations.
        """
        waiver = NOT_REQUIRED_RE.match(t.text, pos)
        if waiver is None:
            return True, t.ends[pos - 1]
        after = t.ends[waiver.end() - 1]
        clause_end = max(after, self._clause((after - 1, after), unit)[1])
        return PROHIBITION_RE.search(self.work, after, clause_end) is not None, clause_end

    def _approvals(self, t: _Template, unit: _Unit, actions: Sequence[tuple[Mention, EntityRef]]) -> None:
        done: set[tuple[EntityRef, EntityRef]] = set()
        for rx in (
            APPROVE_ACTION_RE,
            APPROVE_SUBJECT_RE,
            APPROVE_DECL_RE,
            APPROVE_JUDGE_RE,
            APPROVE_JUDGE_SUBJECT_RE,
        ):
            for m in rx.finditer(t.text):
                required, statement_end = self._approval_polarity(t, m.end("ap"), unit)
                if not required:
                    continue  # "롤백은 박팀장 승인 불필요" waives approval; nobody approves
                span = self._word_span(t.span(*m.span()), unit)
                span = (span[0], max(span[1], statement_end))
                acts = t.group(m, "act") if "act" in rx.groupindex else []
                targets = self._action_targets(acts, span, unit, actions)
                approvers = [w for w in t.group(m, "who") if w.kind in _APPROVER_KINDS]
                cands = self._cands(span, unit)
                for who, _ in self._refs(approvers, unit):
                    for action in targets:
                        if (action, who) in done:
                            continue
                        done.add((action, who))
                        self.sink.attr(action, _AttrSpec(AttrKey.REQUIRES_APPROVAL, True, cands))
                        self.sink.relation(EdgeKind.APPROVED_BY, action, who, cands)
        for m in NO_APPROVAL_RE.finditer(t.text):
            span = self._word_span(t.span(*m.span()), unit)
            clause = self._clause(span, unit)
            required, _ = self._approval_polarity(t, m.end(), unit)
            for mention, ref in actions:
                if clause[0] <= mention.start and mention.end <= clause[1]:
                    self.sink.attr(
                        ref, _AttrSpec(AttrKey.REQUIRES_APPROVAL, required, _dedupe([clause, span]))
                    )

    def _executors(self, t: _Template, unit: _Unit, actions: Sequence[tuple[Mention, EntityRef]]) -> None:
        for rx in (
            EXECUTE_ACTION_RE,
            EXECUTE_ACTION_DECL_RE,
            EXECUTE_SUBJECT_RE,
            EXECUTE_DECL_RE,
            EXECUTE_TOPIC_RE,
        ):
            for m in rx.finditer(t.text):
                if PROHIBITED_AFTER_RE.match(t.text, m.end()):
                    continue  # "김대리가 롤백을 수행하지 않는다"
                span = self._word_span(t.span(*m.span()), unit)
                acts = t.group(m, "act") if "act" in rx.groupindex else []
                cands = self._cands(span, unit)
                for who, _ in self._refs(t.group(m, "who"), unit):
                    for action in self._action_targets(acts, span, unit, actions):
                        self.sink.relation(EdgeKind.EXECUTED_BY, action, who, cands)

    # ------------------------------------------------------------------ channels
    def _reachability(self, t: _Template, unit: _Unit, state: _Frame) -> None:
        fallback_decls = [m.span() for m in FALLBACK_DECL_RE.finditer(t.text)]
        for m in PAREN_CONTACT_RE.finditer(t.text):
            channels = [c for c in t.group(m, "inner") if c.kind is NodeKind.CHANNEL]
            self._link_channels(t.group(m, "who"), channels, t.span(*m.span()), unit, state)
        colon = PERSON_COLON_RE.match(t.text)
        if colon is not None:
            self._link_channels(
                t.group(colon, "who"), t.group(colon, "via"), t.span(*colon.span()), unit, state
            )
        for m in CONTACT_DECL_RE.finditer(t.text):
            if _overlaps(m.span(), fallback_decls):
                continue
            span = t.span(*m.span())
            who = t.group(m, "who")
            if not who:
                # "담당자: 김대리 / 연락처: 슬랙" — the person named just before in the sentence;
                # the quote runs from that name so it states both ends of the edge
                before = [t.slots[i] for i in sorted(t.slots) if i < m.start() and t.slots[i].kind in _PEOPLE]
                who = before[-1:]
                span = (who[0].start, span[1]) if who else span
            if who:
                self._link_channels(who, t.group(m, "via"), span, unit, state)
                continue
            # "연락처: 슬랙" without a name belongs to the scope's owner (or first contact); when
            # nobody is named yet it waits for the scope to name its owner
            channel_refs = self._refs(t.group(m, "via"), unit)
            targets = state.owners or [c.ref for c in state.contacts]
            if not targets and channel_refs:
                decl = _PendingChannels(tuple(ref for ref, _ in channel_refs), self._cands(span, unit))
                state.pending_channels.append(decl)
            for target in targets:
                cands = _with_context(self._cands(span, unit), [state.where.get(target)])
                for channel, _ in channel_refs:
                    self.sink.relation(EdgeKind.REACHABLE_VIA, target, channel, cands)
            if channel_refs and state.primary is None:
                state.primary = channel_refs[0][0]
            state.note(channel_refs)

    def _link_channels(
        self, people: Sequence[Mention], channels: Sequence[Mention], span: Span, unit: _Unit, state: _Frame
    ) -> None:
        cands = self._cands(span, unit)
        channel_refs = self._refs(channels, unit)
        people_refs = self._refs(people, unit)
        for who, _ in people_refs:
            for channel, _ in channel_refs:
                self.sink.relation(EdgeKind.REACHABLE_VIA, who, channel, cands)
        if channel_refs and state.primary is None:
            state.primary = channel_refs[0][0]
        state.note([*people_refs, *channel_refs])

    def _fallbacks(self, t: _Template, unit: _Unit, state: _Frame) -> None:
        for idx in sorted(t.slots):
            if t.slots[idx].kind is not NodeKind.CHANNEL:
                continue
            m = FALLBACK_RE.match(t.text, idx)
            if m is None:
                continue
            cands = self._cands(t.span(*m.span()), unit)
            for down, _ in self._refs([t.slots[idx]], unit):
                for dst, _ in self._refs(t.group(m, "b"), unit):
                    self.sink.relation(EdgeKind.FALLBACK_TO, down, dst, cands)
        for m in FALLBACK_DECL_RE.finditer(t.text):
            targets = t.group(m, "b")
            # The primary is the channel named earlier in the same sentence ("주 채널은 슬랙,
            # 대체 채널은 문자"), else the scope's primary channel.
            earlier = [
                t.slots[i] for i in sorted(t.slots) if i < m.start() and t.slots[i].kind is NodeKind.CHANNEL
            ]
            refs = self._refs(earlier[-1:], unit)
            src, src_span = refs[0] if refs else (state.primary, None)
            if src is None:
                continue
            src_span = src_span or state.where.get(src)
            cands = _with_context(self._cands(t.span(*m.span()), unit), [src_span])
            for dst, _ in self._refs(targets, unit):
                self.sink.relation(EdgeKind.FALLBACK_TO, src, dst, cands)
        for m in FALLBACK_SEQ_RE.finditer(t.text):
            cands = self._cands(t.span(*m.span()), unit)
            for src, _ in self._refs(t.group(m, "a"), unit):
                for dst, _ in self._refs(t.group(m, "b"), unit):
                    self.sink.relation(EdgeKind.FALLBACK_TO, src, dst, cands)
        self._ranked_channels(t, unit, state)

    def _ranked_channels(self, t: _Template, unit: _Unit, state: _Frame) -> None:
        """ "슬랙(1순위), 문자(대체)": the channel ranked first falls back to each channel marked as
        a backup in the same statement. Without a first-ranked channel nothing is linked."""
        primary: tuple[int, int, Mention] | None = None
        backups: list[tuple[int, int, Mention]] = []
        for m in RANKED_CHANNEL_RE.finditer(t.text):
            note = m.group("note").strip()
            entry = (m.start(), m.end(), t.slots[m.start()])
            if _PRIMARY_NOTE_RE.fullmatch(note):
                primary = primary or entry
            elif _SECONDARY_NOTE_RE.fullmatch(note):
                backups.append(entry)
        if primary is None:
            return
        refs = self._refs([primary[2]], unit)
        if not refs:
            return
        src = refs[0][0]
        if state.primary is None:
            state.primary = src
        for start, end, mention in backups:
            cands = self._cands(t.span(min(primary[0], start), max(primary[1], end)), unit)
            for dst, _ in self._refs([mention], unit):
                self.sink.relation(EdgeKind.FALLBACK_TO, src, dst, cands)

    # ------------------------------------------------------------------ obligations
    def _obligations(self, t: _Template, unit: _Unit, service: EntityRef | None, state: _Frame) -> None:
        recipients = recipients_in(t.text)
        hours = _HOURS_RE.search(t.text)
        sentence_obligations: list[EntityRef] = []
        if recipients and (hours is not None or _REPORT_VERB_RE.search(t.text)):
            ref = self._emit_obligation(t, unit, recipients, hours)
            if ref is not None:
                sentence_obligations.append(ref)
                if service is not None:
                    self.sink.relation(
                        EdgeKind.MUST_REPORT, service, ref, self._obligation_cands(t, unit, recipients, hours)
                    )
                state.obligations = [ref]
        targets = sentence_obligations or state.obligations
        if not targets:
            return
        for rx in (
            REPORTER_SUBJECT_RE,
            REPORTER_OBJECT_RE,
            REPORTER_DECL_RE,
            REPORTER_PAREN_RE,
            REPORTER_ROLE_RE,
        ):
            for m in rx.finditer(t.text):
                cands = self._cands(t.span(*m.span()), unit)
                for who, _ in self._refs(t.group(m, "who"), unit):
                    for obligation in targets:
                        self.sink.relation(EdgeKind.REPORTED_BY, obligation, who, cands)

    def _obligation_cands(
        self,
        t: _Template,
        unit: _Unit,
        recipients: Sequence[tuple[str, int, int]],
        hours: re.Match[str] | None,
    ) -> tuple[Span, ...]:
        starts = [r[1] for r in recipients] + ([hours.start()] if hours else [])
        ends = [r[2] for r in recipients] + ([hours.end()] if hours else [])
        return self._cands(t.span(min(starts), max(ends)), unit)

    def _emit_obligation(
        self,
        t: _Template,
        unit: _Unit,
        recipients: Sequence[tuple[str, int, int]],
        hours: re.Match[str] | None,
    ) -> EntityRef | None:
        labels = list(dict.fromkeys(label for label, _, _ in recipients))
        first = t.span(recipients[0][1], recipients[0][2])
        attrs = [_AttrSpec(AttrKey.RECIPIENT, "·".join(labels), _dedupe([first, unit.seg]))]
        if hours is not None:
            attrs.append(
                _AttrSpec(
                    AttrKey.DEADLINE_MINUTES,
                    int(hours.group("h")) * 60,
                    _dedupe([t.span(*hours.span()), unit.seg]),
                )
            )
        incident = self._incident(t, recipients, labels)
        if incident is not None:
            itype, ispan = incident
            attrs.append(_AttrSpec(AttrKey.INCIDENT_TYPE, itype.value, _dedupe([ispan, unit.seg])))
        kind_label = {IncidentType.BREACH: "침해사고", IncidentType.PRIVACY_LEAK: "개인정보 유출"}.get(
            incident[0] if incident else IncidentType.OUTAGE, ""
        )
        verb = "통지" if "통지" in t.text and "신고" not in t.text else "신고"
        name = " ".join(x for x in ("·".join(labels), kind_label, verb) if x)
        if hours is not None:
            # One duty stated with two deadlines ("KISA 24시간" in one section, "48시간" in
            # another) is two statements to check: a shared node would keep only the first.
            earlier = self.sink.stated(NodeKind.OBLIGATION, name, AttrKey.DEADLINE_MINUTES)
            if earlier is not None and earlier != int(hours.group("h")) * 60:
                name = f"{name} ({hours.group('h')}시간)"
        return self.sink.entity(
            NodeKind.OBLIGATION, name, self._obligation_cands(t, unit, recipients, hours), attrs
        )

    def _incident(
        self, t: _Template, recipients: Sequence[tuple[str, int, int]], labels: Sequence[str]
    ) -> tuple[IncidentType, Span] | None:
        """Incident type from the sentence's own words, else the enclosing headings, else the
        recipient (개인정보보호위원회 only handles privacy). The evidence span goes with it."""
        blanked = list(t.text)
        for _, start, end in recipients:
            blanked[start:end] = " " * (end - start)
        sentence = "".join(blanked)
        word = incident_word_span(sentence)
        if word is not None:
            itype = incident_type_of(sentence, labels)
            if itype is not None:
                return itype, t.span(*word)
        for frame in reversed(self.stack):
            if frame.heading is None:
                continue
            text = self.work[frame.heading[0] : frame.heading[1]]
            hword = incident_word_span(text)
            itype = incident_type_of(text, labels) if hword is not None else None
            if itype is not None and hword is not None:
                return itype, (frame.heading[0] + hword[0], frame.heading[0] + hword[1])
        itype = incident_type_of("", labels)
        if itype is None:
            return None
        return itype, t.span(recipients[0][1], recipients[0][2])

    def _holds_role(self, t: _Template, unit: _Unit) -> None:
        for rx in (ROLE_COLON_RE, ROLE_PAREN_RE):
            for m in rx.finditer(t.text):
                role, who = t.group(m, "role"), t.group(m, "who")
                if role[0].kind is not NodeKind.ROLE or who[0].kind is not NodeKind.PERSON:
                    continue
                refs = self._refs([who[0], role[0]], unit)
                if len(refs) == 2:
                    self.sink.relation(
                        EdgeKind.HOLDS_ROLE, refs[0][0], refs[1][0], self._cands(t.span(*m.span()), unit)
                    )

    # ------------------------------------------------------------------ tables
    def _cells(self, row: Span) -> list[Span]:
        start, end = row
        text = self.work[start:end]
        pipes = [i for i, ch in enumerate(text) if ch == "|" and (i == 0 or text[i - 1] != "\\")]
        return [
            _trim(self.work, start + a + 1, start + b, _CELL_TRIM_CHARS) for a, b in itertools.pairwise(pipes)
        ]

    def _table(self, rows: Sequence[Span]) -> None:
        header = self._cells(rows[0])
        roles = [_header_role(self.work[s:e]) for s, e in header]
        header_line = _trim(self.work, *rows[0])
        if "service" not in roles and "order" in roles and "owner" in roles:
            self._ladder(roles, rows[2:])
            return
        if len(roles) >= 2 and all(role == "other" for role in roles):  # 항목 | 내용
            self._fields(rows[2:])
            return
        for row in rows[2:]:
            self._table_row(header, header_line, roles, row)

    def _fields(self, rows: Sequence[Span]) -> None:
        """A field table (항목 | 내용): each row states one field of the enclosing section, typed
        by its first cell ("담당 | 이수진 대리", "상향 | 10분 미응답 시 박재현 팀장 상향"). Rows
        share the section's scope state, so a later row can build on an earlier one (the
        escalation starts from the owner the 담당 row named)."""
        for row in rows:
            cells = self._cells(row)
            if len(cells) < 2 or cells[1][1] <= cells[1][0]:
                continue
            key, value = cells[0], cells[1]
            line = _trim(self.work, *row)
            role = _header_role(self.work[key[0] : key[1]])
            if not _NEGATIVE_CELL_RE.fullmatch(self.work[value[0] : value[1]]):
                self._in_field = role in _FIELD_ROLES
                try:
                    self._field(role, key, value, line)
                finally:
                    self._in_field = False
            for note in cells[2:]:
                if note[1] > note[0]:
                    self._sentence(note, line)

    def _field(self, role: str, key: Span, value: Span, line: Span) -> None:
        """Read one field row according to the role its key names (see ``_HEADER_ROLES``)."""
        state, service = self._state(), self._service()
        people = self._people_in_cell(value, line) if role in ("owner", "executor") else []
        if role == "owner" and people:
            self._field_owners(people, value, line, service, state)
            self._sentence(value, line)  # "이수진 대리 (슬랙, 전화)" also states how to reach them
        elif role in ("owner", "contact"):  # "연락 | 사내 메신저(1순위) → 안 되면 전화"
            self._field_channels(value, line, state)
            self._sentence(value, line)
        elif role == "fallback":
            self._field_fallbacks(value, line, state)
        elif role == "trigger":
            self._trigger_cell(value, line, service, state)
        elif role == "escalation":
            self._escalation_cell(value, line, key, state, None)
        elif role == "approver":
            row = _Row(line=line, cells={}, header=(key,), header_line=line, service=service)
            self._approver_cell(row, value, key, [])
        elif role == "executor":
            for action in self._header_actions(key, line, service) or self._heading_actions():
                for who, _ in people:
                    self.sink.relation(EdgeKind.EXECUTED_BY, action, who, _dedupe([value, line]))
        else:  # "롤백 | 팀장 승인 후 진행": the key names what the value is about
            self._sentence((key[0], value[1]) if role == "action" else value, line)

    def _field_fallbacks(self, value: Span, line: Span, state: _Frame) -> None:
        primary = state.primary
        if primary is None:
            return
        for channel, _ in self._channels_in_cell(value, line):
            cands = _with_context(_dedupe([value, line]), [state.where.get(primary)])
            self.sink.relation(EdgeKind.FALLBACK_TO, primary, channel, cands)

    def _field_owners(
        self,
        people: Sequence[tuple[EntityRef, Span]],
        value: Span,
        line: Span,
        service: EntityRef | None,
        state: _Frame,
    ) -> None:
        if service is not None:
            for ref, _ in people:
                self.sink.relation(
                    EdgeKind.OWNED_BY, service, ref, self._owner_cands(service, _dedupe([value, line]))
                )
        state.declare_owners(ref for ref, _ in people)
        state.note(people)
        if not state.last:
            state.last = list(people)
        if not state.contacts:
            state.contacts = [_Contact(ref, value, line) for ref, _ in people]

    def _field_channels(self, value: Span, line: Span, state: _Frame) -> None:
        """A contact field without a name: the channels reach the section's owner."""
        channels = self._channels_in_cell(value, line)
        if not channels:
            return
        targets = state.owners or [c.ref for c in state.contacts]
        if not targets:
            state.pending_channels.append(
                _PendingChannels(tuple(ref for ref, _ in channels), _dedupe([value, line]))
            )
        for target in targets:
            cands = _with_context(_dedupe([value, line]), [state.where.get(target)])
            for channel, _ in channels:
                self.sink.relation(EdgeKind.REACHABLE_VIA, target, channel, cands)
        if state.primary is None:
            state.primary = channels[0][0]
        state.note(channels)

    def _row_cells(self, roles: Sequence[str], row: Span) -> dict[str, list[tuple[int, Span]]]:
        by_role: dict[str, list[tuple[int, Span]]] = {}
        for idx, cell in enumerate(self._cells(row)):
            if idx >= len(roles) or cell[1] <= cell[0]:
                continue
            if _NEGATIVE_CELL_RE.fullmatch(self.work[cell[0] : cell[1]]):
                continue
            by_role.setdefault(roles[idx], []).append((idx, cell))
        return by_role

    def _people_in_cell(
        self, cell: Span, line: Span, kinds: frozenset[NodeKind] = _PEOPLE
    ) -> list[tuple[EntityRef, Span]]:
        """People named in a cell whose column says "this is a person". A bare 2–4 syllable
        Korean name is accepted only here, because the header — not a guess — types it."""
        unit = _Unit(cell, line, (cell,))
        mentions = [m for m in self._mentions(cell) if m.kind in _PEOPLE]
        if mentions:
            return self._refs([m for m in mentions if m.kind in kinds], unit)
        text = self.work[cell[0] : cell[1]]
        if (
            NodeKind.PERSON in kinds
            and _NAME_CELL_RE.fullmatch(text)
            and text not in NON_NAME_WORDS
            and text not in NON_SERVICE_WORDS
            and not self.lex.is_service(text)
            and not find_mentions(text, self.lex)
        ):
            ref = self.sink.entity(NodeKind.PERSON, text, _dedupe([cell, line]))
            return [] if ref is None else [(ref, cell)]
        return []

    def _channels_in_cell(self, cell: Span, line: Span) -> list[tuple[EntityRef, Span]]:
        unit = _Unit(cell, line, (cell,))
        return self._refs([m for m in self._mentions(cell) if m.kind is NodeKind.CHANNEL], unit)

    def _table_row(self, header: Sequence[Span], header_line: Span, roles: Sequence[str], row: Span) -> None:
        """One data row: it opens a row scope (its own service in a multi-service table) and
        reads each cell according to its column's role."""
        line = _trim(self.work, *row)
        cells = self._row_cells(roles, row)
        multi_service = "service" in roles
        named = self._row_service(cells, line) if multi_service else None
        service = named[0] if named is not None else None if multi_service else self._service()
        frame = _Frame(
            level=_ROW_LEVEL, service=service, owns_state=True, isolated=multi_service, start=line[0]
        )
        frame.note([named] if named is not None else [])
        self._pop(lambda f: f.level >= _ROW_LEVEL)
        self._push(frame)
        ctx = _Row(line=line, cells=cells, header=tuple(header), header_line=header_line, service=service)
        self._row_people(ctx, frame)
        actions: list[EntityRef] = []
        for _, cell in cells.get("trigger", []):
            self._trigger_cell(cell, line, service, frame)
        for _, cell in cells.get("action", []):
            actions += self._sentence(cell, line)
        for idx, cell in cells.get("approver", []):
            self._approver_cell(ctx, cell, header[idx], actions)
        for idx, cell in cells.get("executor", []):
            targets = (
                actions or self._header_actions(header[idx], header_line, service) or self._heading_actions()
            )
            for who, _ in self._people_in_cell(cell, line):
                for action in targets:
                    self.sink.relation(EdgeKind.EXECUTED_BY, action, who, _dedupe([cell, line]))
        waits = [
            (d.minutes, (d.start, d.end))
            for _, cell in cells.get("wait", [])
            for d in find_durations(self.work[cell[0] : cell[1]], cell[0])
        ]
        for idx, cell in cells.get("escalation", []):
            self._escalation_cell(cell, line, header[idx], frame, waits[0] if waits else None)
        for _, cell in cells.get("other", []):
            self._sentence(cell, line)
        self._pop(lambda f: f is frame)
        if not multi_service:
            self._merge_into_scope(frame)

    def _row_service(
        self, cells: dict[str, list[tuple[int, Span]]], line: Span
    ) -> tuple[EntityRef, Span] | None:
        """The service a row's 서비스 column names, with where it is written."""
        for _, cell in cells.get("service", []):
            label = service_from_cell(self.work[cell[0] : cell[1]], self.lex, offset=cell[0])
            if label is not None:
                ref = self._mention_ref(label, _dedupe([(label.start, label.end), cell, line]))
                return None if ref is None else (ref, (label.start, label.end))
        return None

    def _row_people(self, row: _Row, frame: _Frame) -> None:
        """Owner, person, role, contact-channel and fallback columns: who the row is about and
        how to reach them — the row's response entry point."""
        # A bare "연락" column holds people in some tables and channels in others: a cell of an
        # owner-typed column that names no one but a channel ("| 연락 | 슬랙 |") is a contact cell.
        channel_cells = [cell for _, cell in row.cells.get("contact", [])]
        for _, cell in row.cells.get("owner", []):
            people = self._people_in_cell(cell, row.line)
            for ref, span in people:
                self._row_owner(row, frame, ref, span, _dedupe([cell, row.line]))
            if not people:
                channel_cells.append(cell)
        members = self._row_members(row, frame)
        channels = [
            (ref, cell) for cell in channel_cells for ref, _ in self._channels_in_cell(cell, row.line)
        ]
        reachable = [*frame.owners, *(ref for ref, _ in members if ref not in frame.owners)]
        for person in reachable:
            for channel, cell in channels:
                cands = _with_context(_dedupe([cell, row.line]), [frame.where.get(person)])
                self.sink.relation(EdgeKind.REACHABLE_VIA, person, channel, cands)
        frame.note(channels)
        frame.primary = channels[0][0] if channels else None
        primary = channels[0] if channels else (None, None)
        frame.contacts = [
            _Contact(ref, frame.where.get(ref, row.line), row.line, *primary) for ref in frame.owners
        ]
        for _, cell in row.cells.get("fallback", []):
            for channel, _ in self._channels_in_cell(cell, row.line):
                if frame.primary is not None:
                    cands = _with_context(_dedupe([cell, row.line]), [frame.where.get(frame.primary)])
                    self.sink.relation(EdgeKind.FALLBACK_TO, frame.primary, channel, cands)

    def _row_owner(
        self, row: _Row, frame: _Frame, ref: EntityRef, span: Span, cands: tuple[Span, ...]
    ) -> None:
        if row.service is not None:
            self.sink.relation(EdgeKind.OWNED_BY, row.service, ref, self._owner_cands(row.service, cands))
        if ref not in frame.owners:
            frame.owners.append(ref)
            frame.last.append((ref, span))
            frame.note([(ref, cands[0])])

    def _row_members(self, row: _Row, frame: _Frame) -> list[tuple[EntityRef, Span]]:
        """A directory row (이름 | 역할 | 연락처): the named person holds the roles the 역할 cell
        names, and owns the section only when that cell says so ("1차 담당", "담당자")."""
        members = [p for _, cell in row.cells.get("person", []) for p in self._people_in_cell(cell, row.line)]
        role_cells = [cell for _, cell in row.cells.get("role", [])]
        owner_cell = next((c for c in role_cells if _OWNER_CELL_RE.fullmatch(self.work[c[0] : c[1]])), None)
        for ref, span in members:
            frame.note([(ref, span)])
            if owner_cell is not None:
                self._row_owner(row, frame, ref, span, _with_context(_dedupe([span, row.line]), [owner_cell]))
            for cell in role_cells:
                for role, role_span in self._people_in_cell(cell, row.line, frozenset({NodeKind.ROLE})):
                    cands = _with_context(_dedupe([span, row.line]), [role_span])
                    self.sink.relation(EdgeKind.HOLDS_ROLE, ref, role, cands)
        return members

    def _merge_into_scope(self, row: _Frame) -> None:
        """Rows of a single-service table describe the enclosing section's response path."""
        state = self._state()
        state.declare_owners(row.owners)
        if not state.contacts:
            state.contacts = list(row.contacts)
        if row.last:
            state.last = list(row.last)
        if state.primary is None:
            state.primary = row.primary
        state.actions += [a for a in row.actions if a not in state.actions]
        state.note(row.where.items())

    def _header_actions(
        self, header_cell: Span, header_line: Span, service: EntityRef | None
    ) -> list[EntityRef]:
        unit = _Unit(header_cell, header_line, (header_cell,))
        refs: list[EntityRef] = []
        for m in self._mentions(header_cell):
            if m.kind is NodeKind.ACTION:
                ref = self._action_ref(m, service, self._mention_cands(m, unit))
                if ref is not None:
                    refs.append(ref)
        return refs

    def _approver_cell(
        self, row: _Row, cell: Span, header_cell: Span, row_actions: Sequence[EntityRef]
    ) -> None:
        """Approvers approve the row's actions, else the action named by the column header
        ("롤백 승인"), else the section heading's action; with none of those nothing is linked."""
        approvers = self._people_in_cell(cell, row.line, _APPROVER_KINDS)
        if not approvers:
            return
        header_actions = (
            [] if row_actions else self._header_actions(header_cell, row.header_line, row.service)
        )
        actions = list(row_actions) or header_actions or self._heading_actions()
        cands = _dedupe([cell, row.line])
        for action in header_actions:
            if row.service is not None:
                self.sink.relation(
                    EdgeKind.MITIGATED_BY, row.service, action, _dedupe([header_cell, row.header_line])
                )
        for action in actions:
            self.sink.attr(action, _AttrSpec(AttrKey.REQUIRES_APPROVAL, True, cands))
            for who, _ in approvers:
                self.sink.relation(EdgeKind.APPROVED_BY, action, who, cands)

    def _escalation_cell(
        self, cell: Span, line: Span, header_cell: Span, frame: _Frame, row_wait: tuple[int, Span] | None
    ) -> None:
        """An escalation column cell. Its waiting time is the one the cell states, else the
        column header's ("상향(15분 무응답)"), else the row's waiting-time column ("상향 시간")."""
        header_durations = find_durations(self.work[header_cell[0] : header_cell[1]], header_cell[0])
        default = (
            (header_durations[0].minutes, (header_durations[0].start, header_durations[0].end))
            if header_durations
            else row_wait
        )
        t = _Template.build(self.work, cell, self._mentions(cell))
        unit = _Unit(cell, line, (cell,))
        hits = list(ESCALATE_RE.finditer(t.text))
        if hits:
            for m in hits:
                self._apply_escalation(m, t, unit, frame, default)
            return
        people = self._people_in_cell(cell, line)
        if people:
            self._escalation_chain(cell, line, people, frame, default)
        self._reachability(t, unit, frame)

    def _escalation_chain(
        self,
        cell: Span,
        line: Span,
        people: Sequence[tuple[EntityRef, Span]],
        frame: _Frame,
        default: tuple[int, Span] | None,
    ) -> None:
        """A cell listing "박팀장, 최실장" escalates to both at once; "10분 후 박팀장 → 30분 후
        CTO" is a chain whose steps each carry the time written in their own segment. The default
        time applies to the first step only; a later step without a stated time gets none
        (absent ≠ guessed)."""
        text = self.work[cell[0] : cell[1]]
        cuts = [cell[0] + m.start() for m in _CHAIN_RE.finditer(text)]
        durations = find_durations(text, cell[0])
        sources: list[tuple[EntityRef, Span | None]] = frame.last or [
            (o, frame.where.get(o)) for o in frame.owners
        ]
        first = True
        for lo, hi in zip([cell[0], *cuts], [*cuts, cell[1]], strict=True):
            targets = [(ref, span) for ref, span in people if lo <= span[0] < hi]
            if not targets:
                continue
            stated = next(
                ((d.minutes, (d.start, d.end)) for d in durations if lo <= d.start and d.end <= hi), None
            )
            minutes = stated or (default if first else None)
            first = False
            attrs = (
                []
                if minutes is None
                else [_AttrSpec(AttrKey.AFTER_MINUTES, minutes[0], _dedupe([minutes[1], cell]))]
            )
            for ref, _ in targets:
                for src, src_span in sources:
                    evidence = [src_span, None if minutes is None else minutes[1]]
                    cands = _with_context(_dedupe([cell, line]), evidence)
                    self.sink.relation(EdgeKind.ESCALATES_TO, src, ref, cands, attrs)
            sources = [(ref, span) for ref, span in targets]
        frame.last = sources

    def _trigger_cell(self, cell: Span, line: Span, service: EntityRef | None, frame: _Frame) -> None:
        """A detection cell: its trigger(s) and, when the cell also says who the alarm calls
        ("… 알람 → 김대리 호출(슬랙)"), that call — what the trigger notifies. The call is a
        field statement: it never makes the person the owner."""
        t = _Template.build(self.work, cell, self._mentions(cell))
        unit = _Unit(cell, line, (cell,))
        calls = any(m.kind in _PEOPLE for m in t.slots.values())
        live: list[tuple[_Trigger, Span]] = []
        hits = self._trigger_hits(t)
        for hit in hits:
            stated = self._emit_trigger(t.span(*hit.tspan), hit.core, service, frame, unit)
            if stated is not None:
                live.append(stated)
        if not hits:
            arrow = _CHAIN_RE.search(self.work, cell[0], cell[1]) if calls else None
            condition = _trim(self.work, cell[0], arrow.start()) if arrow is not None else cell
            if condition[1] - condition[0] < 2:
                condition = cell
            core = _clean_label(self.doc.text[condition[0] : condition[1]])
            stated = self._emit_trigger(condition, core, service, frame, unit) if core else None
            if stated is not None:
                live.append(stated)
        if not (live and calls):
            return
        in_field, self._in_field, self._live = self._in_field, True, live
        try:
            self._response_flow(t, unit, service, frame)
        finally:
            self._in_field, self._live = in_field, []

    def _ladder(self, roles: Sequence[str], rows: Sequence[Span]) -> None:
        """An ordered contact table (순서 | 담당자 | 연락 수단 | 대기): row 1 is the first contact,
        each next row is an escalation after the previous row's waiting time."""
        state = self._state()
        service = self._service()
        previous: list[tuple[EntityRef, Span]] = []
        wait: tuple[int, Span] | None = None
        for row in rows:
            line = _trim(self.work, *row)
            cells = self._row_cells(roles, row)
            owner_cells = cells.get("owner", [])
            people = [p for _, cell in owner_cells for p in self._people_in_cell(cell, line)]
            channels = [c for _, cell in cells.get("contact", []) for c in self._channels_in_cell(cell, line)]
            if people:
                step = _LadderStep(people=people, channels=channels, cell=owner_cells[0][1], line=line)
                self._ladder_step(step, previous, wait, service, state)
                previous = people
            durations = [
                d
                for _, cell in cells.get("wait", [])
                for d in find_durations(self.work[cell[0] : cell[1]], cell[0])
            ]
            wait = (durations[0].minutes, (durations[0].start, durations[0].end)) if durations else None

    def _ladder_step(
        self,
        step: _LadderStep,
        previous: Sequence[tuple[EntityRef, Span]],
        wait: tuple[int, Span] | None,
        service: EntityRef | None,
        state: _Frame,
    ) -> None:
        cands = _dedupe([step.cell, step.line])
        for ref, person_span in step.people:
            for channel, channel_span in step.channels:
                # the quote must name the person too, not only the channel cell
                via = _with_context(_dedupe([channel_span, step.line]), [person_span])
                self.sink.relation(EdgeKind.REACHABLE_VIA, ref, channel, via)
        if not previous:
            if not state.declared_owner:
                for ref, _ in step.people:
                    if service is not None:
                        self.sink.relation(EdgeKind.OWNED_BY, service, ref, self._owner_cands(service, cands))
                state.declare_owners(ref for ref, _ in step.people)
            if not state.contacts:
                primary = step.channels[0] if step.channels else (None, None)
                state.contacts = [_Contact(ref, step.cell, step.line, *primary) for ref, _ in step.people]
        else:
            attrs = [_AttrSpec(AttrKey.AFTER_MINUTES, wait[0], (wait[1],))] if wait is not None else []
            for src, src_span in previous:
                evidence = [src_span, wait[1] if wait is not None else None]
                for dst, _ in step.people:
                    self.sink.relation(EdgeKind.ESCALATES_TO, src, dst, _with_context(cands, evidence), attrs)
        if step.channels and state.primary is None:
            state.primary = step.channels[0][0]
        state.last = list(step.people)


def _header_role(text: str) -> str:
    key = re.sub(r"[\s*`_]", "", text).lower()
    for role, pattern in _HEADER_ROLES:
        if pattern.search(key):
            return role
    return "other"


__all__ = ["RuleExtractor"]
