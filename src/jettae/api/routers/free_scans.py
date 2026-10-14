"""The anonymous free runbook scan of the public site (무료 런북 검진).

One call takes runbook files (and, optionally, the structured sources the regular upload accepts),
creates a private workspace for them, diagnoses it with the server's configured LLM mode and
answers with an **access key**. The key is the only way an anonymous visitor gets back into that
workspace (``X-Jettae-Workspace-Key``, see :mod:`jettae.api.access`); it is shown once and stored
only as a digest.

Files go through the same reader as ``POST /workspaces/{ws}/inputs`` (:func:`load_upload`: one
``JETTAE_MAX_UPLOAD_MB`` budget, same formats, PII masking and injection quarantine), and the text
of each runbook — and of all of them together — is capped at ``JETTAE_MAX_DOCUMENT_CHARS``, and
each structured source at :data:`MAX_FREE_SCAN_ROWS` rows: the diagnosis cost grows faster than its
input (with the square of the roster). Every run is a full diagnosis, so for viewers and
anonymous callers it is metered like any free scan: at most
:data:`~jettae.api.ratelimit.FREE_SCAN_ATTEMPTS_PER_MINUTE` attempts per minute and
``JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY`` scans per day from one client address, the shared daily
budget (429 once spent), and a free scan slot (503 while ``JETTAE_MAX_CONCURRENT_SCANS`` run).
Raw runbook text is not stored unless ``JETTAE_STORE_RAW_DOCUMENTS``, the structured inputs are not
kept after the diagnosis, and the workspace is deleted after ``JETTAE_FREE_SCAN_RETENTION_DAYS``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Final

from fastapi import APIRouter, File, Request, UploadFile

from jettae.api.deps import ServiceDep, SettingsDep
from jettae.api.errors import error_responses
from jettae.api.ratelimit import (
    BUSY_RESPONSE,
    RATE_LIMITED_RESPONSE,
    ClientDep,
    ScanSlotsDep,
    limit_free_scan_attempts,
)
from jettae.api.routers.workspaces import LoadedUpload, load_upload
from jettae.api.schemas import FreeScanResult
from jettae.api.security import IdentityDep, QuotaDep, Role, free_scan
from jettae.domain import Document

__all__ = ["MAX_FREE_SCAN_ROWS", "TAG", "TAG_DESCRIPTION", "check_rows", "check_total_chars", "router"]

TAG: Final = "무료 검진"
TAG_DESCRIPTION: Final = (
    "익명 무료 런북 검진 — 파일을 올리면 비공개 워크스페이스를 만들어 진단하고 접근 키를 돌려줌(일일 한도)"
)

router = APIRouter(tags=[TAG])

#: Rows a free scan may bring per structured source (roster, on-call, catalog). The checks match
#: the rows against each other and the graph, so the cost grows with their square: 1,000 roster rows
#: take seconds, 2,000 about four times as long, and the 35,000 rows of a 2 MB roster hours.
MAX_FREE_SCAN_ROWS: Final = 1000


def check_rows(upload: LoadedUpload, max_rows: int = MAX_FREE_SCAN_ROWS) -> None:
    """``ValueError`` (422) when a structured source of a free scan has more than ``max_rows`` rows."""
    for label, rows in (
        ("인사 명부(roster)", upload.roster),
        ("당직표(oncall)", upload.oncall),
        ("서비스 카탈로그(catalog)", upload.catalog),
    ):
        if rows is not None and len(rows) > max_rows:
            raise ValueError(
                f"무료 검진의 {label}는 {max_rows:,}행까지 받습니다(올린 파일: {len(rows):,}행). "
                "점검할 팀의 행만 남겨 다시 올려 주세요."
            )


def check_total_chars(documents: Sequence[Document], max_chars: int | None) -> None:
    """``ValueError`` (422, like the loader's per-file check) when the runbooks of one scan hold more
    than ``max_chars`` characters together (None = no cap)."""
    if max_chars is None:
        return
    total = sum(len(doc.text) for doc in documents)
    if total > max_chars:
        raise ValueError(
            f"올린 런북의 텍스트가 모두 합쳐 너무 깁니다({total:,}자, 최대 {max_chars:,}자). "
            "핵심 런북만 골라 나눠서 올려 주세요."
        )


@router.post(
    "/free-scans",
    status_code=201,
    operation_id="createFreeScan",
    summary="무료 런북 검진",
    responses={
        **error_responses(401, 413, 422, 502),
        429: {
            **RATE_LIMITED_RESPONSE,
            "description": (
                "무료 진단 한도 초과(free_scan_limit) 또는 요청이 너무 잦음(rate_limited) — Retry-After 참고"
            ),
        },
        503: BUSY_RESPONSE,
    },
)
async def create_free_scan(
    *,
    request: Request,
    service: ServiceDep,
    settings: SettingsDep,
    identity: IdentityDep,
    quota: QuotaDep,
    slots: ScanSlotsDep,
    client: ClientDep,
    files: Annotated[list[UploadFile], File(description="런북 파일(.md .txt .pdf .docx .html) — 하나 이상")],
    roster: Annotated[UploadFile | None, File(description="인사 명부(CSV/YAML)")] = None,
    oncall: Annotated[UploadFile | None, File(description="당직표(CSV/YAML)")] = None,
    catalog: Annotated[
        UploadFile | None, File(description="서비스 카탈로그(YAML/JSON: Backstage·compose·Jaeger)")
    ] = None,
    policy: Annotated[UploadFile | None, File(description="정책(YAML)")] = None,
) -> FreeScanResult:
    """런북을 올려 바로 진단합니다(multipart/form-data, 로그인 불필요).

    비공개 워크스페이스를 새로 만들어 진단하고, 그 워크스페이스를 여는 **접근 키**를 돌려줍니다.
    접근 키는 다시 보여 주지 않습니다. 결과 화면·리포트를 열 때 `X-Jettae-Workspace-Key` 헤더로 보내세요.
    보기 전용·익명 요청은 하루(한국 시간) 무료 진단 한도(`JETTAE_DAILY_FREE_SCAN_LIMIT`, 접속 주소당
    `JETTAE_FREE_SCANS_PER_CLIENT_PER_DAY`)를 쓰며, 입력 오류로 실패해도 한 번으로 셉니다.
    동시에 돌 수 있는 진단 수(`JETTAE_MAX_CONCURRENT_SCANS`)가 차 있으면 503입니다.
    모든 파일을 합쳐 `JETTAE_MAX_UPLOAD_MB`까지 받으며 넘으면 413, 런북 텍스트가 파일마다·모두 합쳐
    `JETTAE_MAX_DOCUMENT_CHARS`자를 넘거나 명부·당직표·카탈로그가 1,000행을 넘으면 422입니다.
    명부·당직표·카탈로그·정책은 진단에만 쓰고 보관하지 않으며, 결과는
    `JETTAE_FREE_SCAN_RETENTION_DAYS`일 뒤 지워집니다.
    """
    if not identity.allows(Role.EDITOR):
        limit_free_scan_attempts(request, client)
    max_chars = settings.max_document_chars
    with free_scan(identity, quota, service.clock(), client=client, slots=slots):
        upload = await load_upload(
            settings.max_upload_mb,
            runbooks=files,
            runbook_field="files",
            roster=roster,
            oncall=oncall,
            catalog=catalog,
            policy=policy,
            max_chars=max_chars,
        )
        check_total_chars(upload.documents, max_chars)
        check_rows(upload)
        outcome = await service.free_scan(
            documents=upload.documents,
            roster=upload.roster,
            oncall=upload.oncall,
            catalog=upload.catalog,
            policy=upload.policy,
            actor=identity.user,
        )
    return FreeScanResult(
        workspace_id=outcome.workspace_id,
        access_key=outcome.access_key,
        readiness_overall=outcome.report.readiness.overall,
        findings=len(outcome.report.findings),
    )
