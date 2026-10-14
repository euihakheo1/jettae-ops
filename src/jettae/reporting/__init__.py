"""Reporting: present one diagnosis as evidence (Markdown, HTML), as alerts (SARIF) and as a
terminal summary.

Every format reads the same :class:`~jettae.domain.DiagnosisReport` and judges nothing (P1): the
Markdown and HTML evidence reports share one view model (:func:`build_evidence`) so they cannot
disagree, and all output is deterministic (P6) with untrusted text escaped for its format (P5).
"""

from jettae.reporting.evidence import EVIDENCE_TITLE, EvidenceView, build_evidence
from jettae.reporting.html import render_html, render_view_html
from jettae.reporting.markdown import render_evidence_markdown, render_view_markdown
from jettae.reporting.sarif import to_sarif
from jettae.reporting.text import render_cli_summary

__all__ = [
    "EVIDENCE_TITLE",
    "EvidenceView",
    "build_evidence",
    "render_cli_summary",
    "render_evidence_markdown",
    "render_html",
    "render_view_html",
    "render_view_markdown",
    "to_sarif",
]
