"""The latest diagnosis and its evidence report downloads (증적 리포트).

Downloads carry an RFC 6266 ``Content-Disposition`` with both an ASCII ``filename`` (old clients)
and an RFC 5987 ``filename*`` in UTF-8, so the saved file keeps the Korean workspace name.

``report.html`` is served *inline* (the web app previews it, a person may open it in a tab) under a
sandboxing CSP: the report is built from runbook text, which is untrusted (P5), so even if a
renderer bug ever let markup through, nothing in it could run scripts or reach the network.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Final, Literal
from urllib.parse import quote

from fastapi import APIRouter, Query, Response

from jettae.api.deps import ServiceDep
from jettae.api.errors import ApiProblem, ErrorCode, error_responses
from jettae.api.security import IdentityDep
from jettae.domain import DiagnosisReport
from jettae.orchestrator import JettaeService

__all__ = ["MEDIA_TYPES", "TAG", "TAG_DESCRIPTION", "content_disposition", "download_names", "router"]

TAG: Final = "리포트"
TAG_DESCRIPTION: Final = "최신 진단 결과와 증적 리포트(Markdown·HTML·SARIF) 내려받기"

router = APIRouter(prefix="/workspaces", tags=[TAG])

ReportKind = Literal["md", "html", "sarif"]

MEDIA_TYPES: Final[dict[ReportKind, str]] = {
    "md": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "sarif": "application/sarif+json",
}
_REPORT_CSP: Final = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src 'self' data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'self'; sandbox"
)
#: characters Windows/macOS refuse in file names
_UNSAFE_FILENAME: Final = re.compile(r'[\\/:*?"<>|]')
_ASCII_SAFE: Final = re.compile(r"[^A-Za-z0-9._-]+")
#: Hangul fillers draw nothing although they are letters (category Lo)
_HANGUL_FILLERS: Final = frozenset("\u115f\u1160\u3164\uffa0")
_MAX_NAME: Final = 120


def _invisible(char: str) -> bool:
    return char in _HANGUL_FILLERS or unicodedata.category(char)[0] in ("C", "Z")


def download_names(workspace_id: str, workspace_name: str, version: int, kind: ReportKind) -> tuple[str, str]:
    """``(ascii fallback, Korean display name)`` of a report download (both safe as file names).

    Invisible characters (controls, format characters, separators, Hangul fillers) become spaces
    and runs of spaces collapse, so a saved file never gets a name that looks blank or spans lines.
    """
    normalized = unicodedata.normalize("NFC", workspace_name)
    visible = "".join(" " if _invisible(char) else char for char in normalized)
    label = " ".join(_UNSAFE_FILENAME.sub("_", visible).split())[:_MAX_NAME].strip(" .") or "워크스페이스"
    ascii_id = _ASCII_SAFE.sub("_", workspace_id).strip("._") or "workspace"
    return (
        f"jettae-report-{ascii_id}-v{version}.{kind}",
        f"제때옵스-증적리포트-{label}-v{version}.{kind}",
    )


def content_disposition(
    disposition: Literal["attachment", "inline"], ascii_name: str, display_name: str
) -> str:
    """RFC 6266 header value with an RFC 5987 ``filename*`` (UTF-8, percent-encoded)."""
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(display_name, safe='')}"


def _download(
    service: JettaeService, ws: str, kind: ReportKind, body: str, disposition: Literal["attachment", "inline"]
) -> Response:
    workspace = service.get_workspace(ws)
    report = service.latest_report(ws)
    ascii_name, display_name = download_names(workspace.id, workspace.name, report.graph_version, kind)
    headers = {"Content-Disposition": content_disposition(disposition, ascii_name, display_name)}
    if kind == "html":
        headers["Content-Security-Policy"] = _REPORT_CSP
        headers["X-Frame-Options"] = "SAMEORIGIN"
    return Response(content=body.encode("utf-8"), media_type=MEDIA_TYPES[kind], headers=headers)


@router.get(
    "/{ws}/report",
    operation_id="getReport",
    summary="최신 진단 결과",
    responses=error_responses(401, 404, 422),
)
def get_report(
    ws: str,
    service: ServiceDep,
    identity: IdentityDep,
    version: int | None = Query(default=None, ge=1, description="이 그래프 버전의 결과만(최신이 아니면 404)"),
) -> DiagnosisReport:
    """최신 진단 결과(끊김·점수·리허설·수정안·근거율·LLM 사용량)를 돌려줍니다.

    `version`을 주면 최신 결과가 그 버전을 판정한 것일 때만 돌려줍니다 — 지난 버전의 결과는
    보관하지 않으므로, 다른 버전의 결과를 대신 보여 주지 않고 404로 알립니다.
    """
    report = service.latest_report(ws)
    if version is not None and version != report.graph_version:
        raise ApiProblem(
            404,
            f"그래프 버전 {version}의 진단 결과는 조회할 수 없습니다. "
            f"지금은 최신 결과(버전 {report.graph_version})만 제공합니다.",
            code=ErrorCode.REPORT_NOT_FOUND,
        )
    return report


_DOWNLOAD_RESPONSES: Final = error_responses(401, 404, 422)


@router.get(
    "/{ws}/report.md",
    operation_id="getReportMarkdown",
    summary="증적 리포트(Markdown) 내려받기",
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}}, "description": "증적 리포트"}, **_DOWNLOAD_RESPONSES},
)
def get_report_markdown(ws: str, service: ServiceDep, identity: IdentityDep) -> Response:
    """ISMS-P 증적 초안(Markdown)을 첨부 파일로 내려줍니다."""
    return _download(service, ws, "md", service.evidence_markdown(ws), "attachment")


@router.get(
    "/{ws}/report.html",
    operation_id="getReportHtml",
    summary="증적 리포트(HTML)",
    response_class=Response,
    responses={
        200: {"content": {"text/html": {}}, "description": "인쇄용 증적 리포트"},
        **_DOWNLOAD_RESPONSES,
    },
)
def get_report_html(ws: str, service: ServiceDep, identity: IdentityDep) -> Response:
    """인쇄용 증적 리포트(HTML, 스크립트 없음)를 바로 볼 수 있게 내려줍니다."""
    return _download(service, ws, "html", service.evidence_html(ws), "inline")


@router.get(
    "/{ws}/report.sarif",
    operation_id="getReportSarif",
    summary="SARIF 내려받기",
    response_class=Response,
    responses={
        200: {"content": {"application/sarif+json": {}}, "description": "SARIF 2.1.0"},
        **_DOWNLOAD_RESPONSES,
    },
)
def get_report_sarif(ws: str, service: ServiceDep, identity: IdentityDep) -> Response:
    """끊김을 SARIF 2.1.0으로 내려줍니다(GitHub 코드 스캔·CI 연동용)."""
    body = json.dumps(service.sarif(ws), ensure_ascii=False, indent=2)
    return _download(service, ws, "sarif", body, "attachment")
