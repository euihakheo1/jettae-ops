"""Workspaces and their inputs (워크스페이스·입력 업로드).

Uploads are taken as **bytes** end to end: runbooks go through :func:`jettae.ingest.load_bytes`
and the structured sources through the ``load_*`` loaders with ``bytes``. Those loaders read a
*string* that looks like ``roster.csv`` as a file path, so passing anything a client typed as a
string could make the server read its own files; bytes can only ever be content.

A client-supplied file name is reduced to its last path component before it is used (as the
document's display path, PII-masked by the loader) — never to open anything. Characters that could
fake a line break, reorder what is shown (bidi overrides) or drive a terminal become ``_``: the
name is stored and printed in every report, the web app and the CLI.

:func:`load_upload` reads and parses one multipart upload within the size budget; the anonymous
free scan (:mod:`jettae.api.routers.free_scans`) takes its files through the same function.

In ``JETTAE_AUTH_MODE=public`` an anonymous visitor's workspace list holds only the demo workspace;
the workspaces their free scans created are opened one at a time with the access key. Two names are
reserved: the demo's (the demo endpoint reloads the example inputs into it) and the free scans'
(the retention purge deletes those workspaces after ``JETTAE_FREE_SCAN_RETENTION_DAYS``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from jettae.api.deps import ServiceDep, SettingsDep, latest_report_or_none
from jettae.api.errors import ApiProblem, ErrorCode, error_responses
from jettae.api.schemas import CreateWorkspaceRequest, WorkspaceSummary, workspace_summary
from jettae.api.security import EditorDep, IdentityDep, Role
from jettae.domain import CatalogService, Document, OnCallShift, Policy, RosterEntry
from jettae.ingest import load_bytes, load_catalog, load_oncall, load_policy, load_roster
from jettae.orchestrator import DEMO_WORKSPACE_NAME, FREE_SCAN_WORKSPACE_NAME, InputSummary
from jettae.store.base import validate_workspace_name

__all__ = ["TAG", "TAG_DESCRIPTION", "LoadedUpload", "load_upload", "router"]

TAG: Final = "워크스페이스"
TAG_DESCRIPTION: Final = (
    "워크스페이스 생성·조회와 입력(런북·명부·당직표·카탈로그·정책·리허설 시나리오) 업로드"
)

router = APIRouter(prefix="/workspaces", tags=[TAG])

_MB: Final = 1024 * 1024
_SEPARATORS: Final = re.compile(r"[\\/]")
#: control, format (incl. bidi overrides), surrogate and line/paragraph-separator characters
_UNSAFE_NAME_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
#: the usual file-system limit; anything longer is not a file name a person chose
_MAX_FILENAME: Final = 255


def _upload_name(filename: str | None) -> str:
    """The last path component of a client-supplied file name, safe to store and display."""
    last = _SEPARATORS.split(filename or "")[-1]
    return "".join(
        "_" if unicodedata.category(char) in _UNSAFE_NAME_CATEGORIES else char for char in last
    ).strip()


@router.post(
    "",
    status_code=201,
    operation_id="createWorkspace",
    summary="워크스페이스 만들기",
    responses=error_responses(401, 403, 409, 422),
)
def create_workspace(
    body: CreateWorkspaceRequest, service: ServiceDep, identity: EditorDep
) -> WorkspaceSummary:
    """새 워크스페이스를 만듭니다(편집자 이상). id는 이름과 시각에서 결정적으로 정해집니다.

    예제 워크스페이스와 무료 검진의 이름은 예약되어 있습니다(예제는 `POST /api/v1/demo/dmc-shop`으로 엽니다).
    """
    name = validate_workspace_name(body.name)
    # The demo endpoint (open to viewers and anonymous callers) finds its workspace by this name
    # and reloads the example inputs into it: a team workspace with that name would be overwritten.
    if name == DEMO_WORKSPACE_NAME:
        raise ApiProblem(
            422,
            f"'{DEMO_WORKSPACE_NAME}'은(는) 예제 워크스페이스용으로 예약된 이름입니다. 다른 이름을 쓰세요"
            "(예제는 '예제로 시작' — POST /api/v1/demo/dmc-shop — 으로 엽니다).",
            code=ErrorCode.INVALID_INPUT,
        )
    # The retention purge recognises free-scan workspaces by this name and deletes them.
    if name == FREE_SCAN_WORKSPACE_NAME:
        raise ApiProblem(
            422,
            f"'{FREE_SCAN_WORKSPACE_NAME}'은(는) 무료 검진용으로 예약된 이름입니다(보관 기간이 지나면 "
            "자동으로 지워집니다). 다른 이름을 쓰세요.",
            code=ErrorCode.INVALID_INPUT,
        )
    workspace = service.create_workspace(body.name, actor=identity.user)
    return workspace_summary(workspace, None, demo_name=DEMO_WORKSPACE_NAME)


@router.get(
    "",
    operation_id="listWorkspaces",
    summary="워크스페이스 목록",
    responses=error_responses(401, 422),
)
def list_workspaces(
    service: ServiceDep, settings: SettingsDep, identity: IdentityDep
) -> list[WorkspaceSummary]:
    """만든 순서대로 워크스페이스와 최신 진단의 준비도 점수·끊김 수를 돌려줍니다.

    공개 모드(`JETTAE_AUTH_MODE=public`)의 익명 방문자에게는 예제 워크스페이스만 보입니다.
    """
    workspaces = service.list_workspaces()
    if settings.auth_mode == "public" and identity.anonymous:
        workspaces = [workspace for workspace in workspaces if workspace.name == DEMO_WORKSPACE_NAME]
    return [
        workspace_summary(
            workspace, latest_report_or_none(service, workspace.id), demo_name=DEMO_WORKSPACE_NAME
        )
        for workspace in workspaces
    ]


@router.get(
    "/{ws}",
    operation_id="getWorkspace",
    summary="워크스페이스 조회",
    responses=error_responses(401, 404, 422),
)
def get_workspace(ws: str, service: ServiceDep, identity: IdentityDep) -> WorkspaceSummary:
    workspace = service.get_workspace(ws)
    return workspace_summary(workspace, latest_report_or_none(service, ws), demo_name=DEMO_WORKSPACE_NAME)


@router.get(
    "/{ws}/inputs",
    operation_id="getInputs",
    summary="현재 입력 요약",
    responses=error_responses(401, 404, 422),
)
def get_inputs(ws: str, service: ServiceDep, identity: IdentityDep) -> InputSummary:
    """다음 진단에 쓰일 입력(런북 목록, 명부·당직표·카탈로그 건수, 격리된 지시문 수)을 요약합니다."""
    return service.input_summary(ws)


# --------------------------------------------------------------------------- upload


class _Budget:
    """Bytes still allowed in this upload (``JETTAE_MAX_UPLOAD_MB`` covers all files together)."""

    def __init__(self, max_mb: int) -> None:
        self.max_mb = max_mb
        self.remaining = max_mb * _MB

    async def read(self, upload: UploadFile, field: str) -> tuple[str, bytes]:
        """``(file name, bytes)`` of one part; 413 as soon as the budget is exceeded."""
        name = _upload_name(upload.filename)
        if not name:
            raise ApiProblem(422, f"{field}: 파일 이름이 없습니다.", code=ErrorCode.INVALID_INPUT)
        if len(name) > _MAX_FILENAME:
            raise ApiProblem(
                422,
                f"{field}: 파일 이름이 너무 깁니다(최대 {_MAX_FILENAME}자).",
                code=ErrorCode.INVALID_INPUT,
            )
        data = await upload.read(self.remaining + 1)
        if len(data) > self.remaining:
            raise ApiProblem(
                413,
                f"올린 파일이 너무 큽니다({name}). "
                f"한 번에 모두 합쳐 최대 {self.max_mb}MB까지 올릴 수 있습니다.",
                code=ErrorCode.PAYLOAD_TOO_LARGE,
            )
        self.remaining -= len(data)
        return name, data


def _documents(files: list[tuple[str, bytes]], max_bytes: int, max_chars: int | None) -> tuple[Document, ...]:
    """Load runbooks; the same document uploaded twice (same title and text) is kept once."""
    loaded: dict[str, Document] = {}
    for name, data in files:
        doc = load_bytes(name, data, max_bytes=max_bytes, max_chars=max_chars)
        loaded.setdefault(doc.doc_id, doc)
    return tuple(loaded.values())


def _structured(
    roster: bytes | None, oncall: bytes | None, catalog: bytes | None, policy: bytes | None
) -> tuple[
    tuple[RosterEntry, ...] | None,
    tuple[OnCallShift, ...] | None,
    tuple[CatalogService, ...] | None,
    Policy | None,
]:
    return (
        load_roster(roster) if roster is not None else None,
        load_oncall(oncall) if oncall is not None else None,
        load_catalog(catalog) if catalog is not None else None,
        load_policy(policy) if policy is not None else None,
    )


@dataclass(frozen=True)
class LoadedUpload:
    """One multipart upload, read within the size budget and parsed. ``None`` = not uploaded."""

    documents: tuple[Document, ...]
    roster: tuple[RosterEntry, ...] | None
    oncall: tuple[OnCallShift, ...] | None
    catalog: tuple[CatalogService, ...] | None
    policy: Policy | None
    #: raw bytes of a drill file (resolved by the next diagnosis, never parsed here)
    scenarios: bytes | None


async def load_upload(
    max_upload_mb: int,
    *,
    runbooks: Sequence[UploadFile],
    runbook_field: str = "runbooks",
    roster: UploadFile | None = None,
    oncall: UploadFile | None = None,
    catalog: UploadFile | None = None,
    policy: UploadFile | None = None,
    scenarios: UploadFile | None = None,
    max_chars: int | None = None,
) -> LoadedUpload:
    """Read every part within one ``JETTAE_MAX_UPLOAD_MB`` budget (413 as soon as it is exceeded),
    then load the runbooks and the structured sources (422 with the loader's Korean message).

    ``runbook_field`` names the runbooks' form field in error messages; ``max_chars`` caps the
    converted text of each runbook (``JETTAE_MAX_DOCUMENT_CHARS``, None = no cap, 422 when over).
    """
    budget = _Budget(max_upload_mb)
    runbook_files = [await budget.read(upload, runbook_field) for upload in runbooks]
    sources: dict[str, bytes | None] = {}
    for field, upload in (
        ("roster", roster),
        ("oncall", oncall),
        ("catalog", catalog),
        ("policy", policy),
        ("scenarios", scenarios),
    ):
        sources[field] = (await budget.read(upload, field))[1] if upload is not None else None
    documents = await run_in_threadpool(_documents, runbook_files, max_upload_mb * _MB, max_chars)
    roster_rows, shifts, services, policy_value = await run_in_threadpool(
        _structured, sources["roster"], sources["oncall"], sources["catalog"], sources["policy"]
    )
    return LoadedUpload(
        documents=documents,
        roster=roster_rows,
        oncall=shifts,
        catalog=services,
        policy=policy_value,
        scenarios=sources["scenarios"],
    )


@router.post(
    "/{ws}/inputs",
    operation_id="uploadInputs",
    summary="입력 올리기",
    responses=error_responses(401, 403, 404, 413, 422),
)
async def upload_inputs(
    ws: str,
    *,
    service: ServiceDep,
    settings: SettingsDep,
    identity: EditorDep,
    runbooks: Annotated[
        list[UploadFile] | None,
        File(description="런북 파일(.md .txt .pdf .docx .html) — 올리면 런북 묶음 전체를 바꿉니다"),
    ] = None,
    roster: Annotated[UploadFile | None, File(description="인사 명부(CSV/YAML)")] = None,
    oncall: Annotated[UploadFile | None, File(description="당직표(CSV/YAML)")] = None,
    catalog: Annotated[
        UploadFile | None, File(description="서비스 카탈로그(YAML/JSON: Backstage·compose·Jaeger)")
    ] = None,
    policy: Annotated[UploadFile | None, File(description="정책(YAML)")] = None,
    scenarios: Annotated[
        UploadFile | None, File(description="사람이 쓴 리허설 시나리오(scenarios.yaml)")
    ] = None,
) -> InputSummary:
    """다음 진단에 쓸 입력을 올립니다(편집자 이상, multipart/form-data).

    런북을 올리면 지금의 런북 묶음을 **바꾸고**, 명부·당직표·카탈로그·정책은 올린 것만 바꿉니다.
    모든 파일을 합쳐 `JETTAE_MAX_UPLOAD_MB`까지 받으며, 넘으면 413입니다.
    """
    # Uploads need an editor today; should that ever be relaxed, the public text cap follows.
    max_chars = None if identity.allows(Role.EDITOR) else settings.max_document_chars
    await run_in_threadpool(service.get_workspace, ws)
    if not (runbooks or roster or oncall or catalog or policy or scenarios):
        raise ApiProblem(
            422,
            "올린 파일이 없습니다. "
            "런북(runbooks)이나 명부·당직표·카탈로그·정책 파일을 하나 이상 올려 주세요.",
            code=ErrorCode.INVALID_INPUT,
        )
    upload = await load_upload(
        settings.max_upload_mb,
        runbooks=runbooks or [],
        roster=roster,
        oncall=oncall,
        catalog=catalog,
        policy=policy,
        scenarios=scenarios,
        max_chars=max_chars,
    )
    return await run_in_threadpool(
        lambda: service.upload_inputs(
            ws,
            documents=upload.documents,
            roster=upload.roster,
            oncall=upload.oncall,
            catalog=upload.catalog,
            policy=upload.policy,
            scenarios=upload.scenarios,
            actor=identity.user,
        )
    )
