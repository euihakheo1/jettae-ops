"""Ports (hexagonal architecture). Adapters live in ``jettae.store`` / ``jettae.llm``.

Keeping these protocols in the pure domain lets ``improve`` and ``orchestrator`` depend on
abstractions while SQLite/Postgres or a hosted DB are swapped without touching business logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jettae.domain.findings import Finding, ReadinessReport
from jettae.domain.graph import OpsGraph
from jettae.domain.patch import GraphVersion, Patch
from jettae.domain.rehearsal import RehearsalResult, Scenario
from jettae.domain.report import DiagnosisReport


class VersionConflictError(RuntimeError):
    """Raised when a patch targets a base version that is no longer the head."""


class NotFoundError(KeyError):
    """A requested workspace/version/patch/graph does not exist (subclass of KeyError)."""


@runtime_checkable
class GraphRepository(Protocol):
    """Append-only, content-addressed graph versions per workspace."""

    def head(self, workspace_id: str) -> GraphVersion | None: ...

    def get_version(self, workspace_id: str, number: int) -> GraphVersion: ...

    def list_versions(self, workspace_id: str) -> list[GraphVersion]: ...

    def load_graph(self, graph_hash: str) -> OpsGraph: ...

    def append_version(self, version: GraphVersion, graph: OpsGraph, *, expected_head: int | None) -> None:
        """Atomically append; raise VersionConflictError if head != expected_head."""
        ...


@runtime_checkable
class PatchRepository(Protocol):
    def save_patch(self, patch: Patch) -> None: ...

    def get_patch(self, patch_id: str) -> Patch: ...

    def list_patches(self, workspace_id: str) -> list[Patch]: ...


@runtime_checkable
class RehearsalRepository(Protocol):
    def save_scenario(self, workspace_id: str, scenario: Scenario) -> None: ...

    def list_scenarios(self, workspace_id: str) -> list[Scenario]: ...

    def save_result(self, workspace_id: str, result: RehearsalResult) -> str:
        """Persist and return a rehearsal id."""
        ...

    def list_results(self, workspace_id: str, graph_hash: str | None = None) -> list[RehearsalResult]: ...


@runtime_checkable
class ReportRepository(Protocol):
    def save_report(self, report: DiagnosisReport) -> None: ...

    def latest_report(self, workspace_id: str) -> DiagnosisReport | None: ...


@runtime_checkable
class AuditLog(Protocol):
    def record(self, workspace_id: str, actor: str, action: str, payload: dict[str, object]) -> None: ...


__all__ = [
    "AuditLog",
    "Finding",
    "GraphRepository",
    "NotFoundError",
    "PatchRepository",
    "ReadinessReport",
    "RehearsalRepository",
    "ReportRepository",
    "VersionConflictError",
]
