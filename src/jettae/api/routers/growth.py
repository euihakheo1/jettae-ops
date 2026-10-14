"""Pre-launch sign-ups and demand measurement (사전 신청·수요 측정).

Both write endpoints are public (the landing page is) and both are built to hold as little personal
data as possible: a sign-up is stored only with privacy consent (the store's model refuses it
otherwise), and events are limited to an allow-list of names, each with a few enumerated or bounded
properties (:data:`jettae.api.schemas.EVENT_PROPS`) and a screen path from a fixed list
(:data:`jettae.api.schemas.EVENT_SCREEN_PATHS`). Neither echoes what was sent. Both are
rate-limited per client address (``JETTAE_EVENTS_PER_CLIENT_PER_MINUTE``,
``JETTAE_SIGNUPS_PER_CLIENT_PER_HOUR``; 429 with ``Retry-After``), so a script can neither fill the
database nor inflate the demand numbers cheaply; on top of that, at most ``JETTAE_EVENTS_PER_DAY``
events are stored per Korean calendar day site-wide (the rest are answered 202 and dropped, with
one warning in the log per day), so even a client rotating addresses cannot fill the events table.

A sign-up for an address that already has one is answered exactly like a new one but changes
nothing: there is no e-mail verification, so whoever knows an address must not be able to rewrite
that person's consents — nor learn from the answer whether the address signed up. The aggregate
counts are for team leads and admins only; the sign-up list itself is personal data and is for admins
only — every read of it, and every deletion (withdrawal of consent), is written to the audit log.

The CSV export is meant to be opened in Excel by a Korean founder: UTF-8 *with* a BOM (without it
Excel reads Korean as mojibake), and every cell that a spreadsheet would run as a formula (``=``,
``+``, ``-``, ``@``, tab, carriage return — anyone can type them into the public form) is prefixed
with ``'`` so it stays text (CSV injection).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Final, Literal

from fastapi import APIRouter, Depends, Query, Response
from starlette.concurrency import run_in_threadpool

from jettae.api.deps import ServiceDep
from jettae.api.errors import ApiProblem, ErrorCode, error_responses
from jettae.api.ratelimit import RATE_LIMITED_RESPONSE, EventCapDep, limit_events, limit_signups
from jettae.api.schemas import Accepted, EventRequest, SignupDeleted, SignupRequest
from jettae.api.security import AdminDep, DeciderDep
from jettae.orchestrator import DemandMetrics
from jettae.store import Signup

__all__ = ["SIGNUP_CSV_COLUMNS", "TAG", "TAG_DESCRIPTION", "router", "signups_csv"]

TAG: Final = "사전 신청·수요"
TAG_DESCRIPTION: Final = (
    "사전 신청(개인정보 수집·이용 동의 필수)과 수요 측정 이벤트(허용 목록), 명단 내려받기·삭제(관리자)"
)

router = APIRouter(tags=[TAG])

#: Columns of the sign-up CSV, in order.
SIGNUP_CSV_COLUMNS: Final = (
    "created_at",
    "email",
    "organization",
    "team_size",
    "consent_marketing",
    "source",
    "message",
)
#: First characters that make Excel, LibreOffice or Google Sheets evaluate a cell as a formula.
_FORMULA_START: Final = ("=", "+", "-", "@", "\t", "\r")
_BOM: Final = "\ufeff"
_CSV_HEADERS: Final = {"Content-Disposition": 'attachment; filename="signups.csv"'}


def _cell(value: str | None) -> str:
    if value is None:
        return ""
    return f"'{value}" if value.startswith(_FORMULA_START) else value


def signups_csv(signups: Sequence[Signup]) -> bytes:
    """The sign-ups as CSV for Excel: UTF-8 with BOM, CRLF rows, formula-like cells neutralised."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(SIGNUP_CSV_COLUMNS)
    for signup in signups:
        writer.writerow(
            [
                signup.created_at.isoformat(),
                _cell(signup.email),
                _cell(signup.organization),
                _cell(signup.team_size),
                "true" if signup.consent_marketing else "false",
                _cell(signup.source),
                _cell(signup.message),
            ]
        )
    return (_BOM + buffer.getvalue()).encode("utf-8")


@router.post(
    "/signups",
    status_code=201,
    operation_id="createSignup",
    summary="사전 신청",
    responses={**error_responses(422), 429: RATE_LIMITED_RESPONSE},
    dependencies=[Depends(limit_signups)],
)
def create_signup(body: SignupRequest, service: ServiceDep) -> Accepted:
    """상세 진단 사전 신청을 저장합니다.

    같은 이메일로 이미 신청했다면 처음 신청한 내용(동의 여부·시각 포함)을 그대로 두며, 응답은 새 신청과
    똑같습니다(신청 여부를 알려 주지 않음). 내용을 바꾸거나 철회하려면 운영자에게 요청하세요.
    개인정보 수집·이용 동의(`consent_privacy: true`)가 없으면 저장하지 않고 422로 알립니다.
    같은 곳에서 너무 자주 보내면 429입니다(`JETTAE_SIGNUPS_PER_CLIENT_PER_HOUR`).
    """
    if not body.consent_privacy:
        raise ApiProblem(
            422,
            "개인정보 수집·이용에 동의해야 사전 신청을 할 수 있습니다. 동의 항목에 체크해 주세요.",
            code=ErrorCode.CONSENT_REQUIRED,
        )
    signup = Signup(
        email=body.email,
        organization=body.organization,
        team_size=body.team_size,
        message=body.message,
        consent_privacy=True,
        consent_marketing=body.consent_marketing,
        created_at=service.clock(),
        source=body.source,
    )
    service.signup(signup)
    return Accepted()


@router.post(
    "/events",
    status_code=202,
    operation_id="recordEvent",
    summary="수요 측정 이벤트",
    responses={**error_responses(422), 429: RATE_LIMITED_RESPONSE},
    dependencies=[Depends(limit_events)],
)
async def record_event(body: EventRequest, service: ServiceDep, cap: EventCapDep) -> Accepted:
    """허용된 이름의 이벤트만, 이벤트마다 정해진 항목·값의 props와 함께 기록합니다.

    허용 이름: landing_view·landing_first_view(브라우저당 첫 방문 1회)·upload_click·signup_click·
    demo_start·report_download. 무료 검진 횟수(runbook_upload)는 서버가 직접 기록하므로 받지 않습니다.
    path는 화면 경로 `/`·`/app`·`/app/report`·`/privacy` 중 하나만 받습니다(쿼리·# 뒤는 버림).
    props: landing_view·landing_first_view는 referrer(direct·external), upload_click은 files(0~100),
    signup_click은 team_size(가입 양식의 팀 규모)·marketing(true/false), demo_start는
    from(landing·workspace_home), report_download는 kind(md·html·sarif)만 받습니다.
    같은 곳에서 너무 자주 보내면 429입니다(`JETTAE_EVENTS_PER_CLIENT_PER_MINUTE`).
    사이트 전체로 하루(한국 시간) `JETTAE_EVENTS_PER_DAY`건을 넘으면 202로 답하되 저장하지 않습니다.
    """
    if cap.take(service.clock()):
        await run_in_threadpool(service.record_event, body.name, body.stored_props())
    return Accepted()


@router.get(
    "/metrics/demand",
    operation_id="getDemandMetrics",
    summary="수요 지표",
    responses=error_responses(401, 403, 422),
)
def get_demand_metrics(service: ServiceDep, identity: DeciderDep) -> DemandMetrics:
    """사전 신청 인원과 이벤트별 횟수(팀 리드·관리자). 무료 검진 횟수는 `runbook_upload`입니다."""
    return service.demand_metrics()


@router.get(
    "/signups",
    operation_id="listSignups",
    summary="사전 신청 명단(관리자)",
    response_model=list[Signup],
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "사전 신청 명단(JSON 배열, 또는 format=csv면 엑셀용 CSV 첨부 파일)",
        },
        **error_responses(401, 403, 422),
    },
)
def list_signups(
    service: ServiceDep,
    identity: AdminDep,
    export_format: Literal["json", "csv"] = Query(
        default="json", alias="format", description="json(기본) 또는 csv(UTF-8 BOM, 엑셀용)"
    ),
) -> list[Signup] | Response:
    """사전 신청 명단을 신청 시각순으로 돌려줍니다(관리자 전용 — 개인정보).

    `format=csv`면 `signups.csv` 첨부 파일(UTF-8 BOM, 엑셀에서 한글이 깨지지 않음)로 내려줍니다.
    조회할 때마다 누가 몇 건을 어떤 형식으로 봤는지 감사 로그에 남습니다.
    """
    signups = service.list_signups(actor=identity.user, export_format=export_format)
    if export_format == "csv":
        return Response(
            content=signups_csv(signups), media_type="text/csv; charset=utf-8", headers=_CSV_HEADERS
        )
    return signups


@router.delete(
    "/signups/{email}",
    operation_id="deleteSignup",
    summary="사전 신청 삭제(관리자)",
    responses=error_responses(401, 403, 422),
)
def delete_signup(email: str, service: ServiceDep, identity: AdminDep) -> SignupDeleted:
    """한 사람의 사전 신청을 지웁니다(관리자 전용 — 동의 철회·삭제 요청 처리).

    이메일은 신청할 때와 같은 방식으로 맞춰 찾습니다(대소문자·전각 문자 무시). 신청이 없었으면
    `deleted: false`입니다. 감사 로그에는 이메일 대신 그 짧은 해시(SHA-256 앞 12자리)만 남습니다.
    """
    return SignupDeleted(deleted=service.delete_signup(email, actor=identity.user))
