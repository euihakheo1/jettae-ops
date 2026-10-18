"""Structural readings of service-like names: a component *in a state*, and a sentence.

Two generic facts about how runbooks title themselves and their sections, independent of any
organisation or document:

* A title or heading that says something *about* a component — "<component> is down",
  "<component> outage", "Partial outage of <component>" — names that component in a state. It
  is not a second component: when the document already names the component, the phrase is read
  as that name (:func:`state_base`).
* A heading that is a sentence ("<something> is not responding") states a situation; it is not
  the name of a service (:func:`is_sentence`).
* A name that is a named component plus a deployment noun ("<component> clusters", "<component>
  nodes") is that component, deployed — the same name for the purpose of merging a service with
  its system twin (:func:`deployment_base`).

The vocabulary is English on purpose. The Korean rule parser reads Korean headings its own way
and its output is sealed; the assembler applies these readings only to documents the LLM
extractor read (:mod:`jettae.extract.assemble`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

__all__ = ["deployment_base", "is_sentence", "is_state_phrase", "state_base", "words"]

#: Words that by themselves say a component is failing (the core of a state phrase).
_STATE_CORE: Final = frozenset(
    {
        "down",
        "outage",
        "outages",
        "unavailable",
        "unavailability",
        "degraded",
        "degradation",
        "failing",
        "failed",
        "failure",
        "failures",
        "broken",
        "unreachable",
        "offline",
        "unresponsive",
        "disruption",
        "disruptions",
        "disrupted",
        "interruption",
        "interrupted",
        "crashed",
        "crashing",
        "overloaded",
        "saturated",
        "stalled",
        "incident",
        "incidents",
    }
)
#: Words a state phrase may carry around its core ("is completely down", "partial outage of the").
_STATE_FILLER: Final = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "gone",
        "goes",
        "going",
        "currently",
        "completely",
        "partially",
        "fully",
        "totally",
        "partial",
        "complete",
        "total",
        "full",
        "major",
        "minor",
        "widespread",
        "of",
        "the",
        "a",
        "an",
        "in",
        "on",
        "for",
        "with",
        "-",
        "–",
        "—",
    }
)
#: Negations that turn an ordinary word into a failure state ("not responding", "no longer up").
_NEGATIONS: Final = frozenset({"not", "isn't", "aren't", "wasn't", "weren't", "no", "longer"})
_NEGATABLE: Final = frozenset(
    {"working", "responding", "available", "reachable", "running", "serving", "accessible", "healthy", "up"}
)
#: Finite verbs and auxiliaries: a heading containing one is a sentence, not a name.
_FINITE_VERBS: Final = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "has",
        "have",
        "had",
        "hasn't",
        "haven't",
        "hadn't",
        "does",
        "do",
        "did",
        "doesn't",
        "don't",
        "didn't",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "will",
        "won't",
        "would",
        "wouldn't",
        "should",
        "shouldn't",
        "must",
        "seems",
        "appears",
    }
)
#: Nouns that name the deployment of a component (its cluster, nodes, fleet) rather than another one.
_DEPLOYMENT_UNITS: Final = frozenset(
    {
        "cluster",
        "clusters",
        "node",
        "nodes",
        "instance",
        "instances",
        "server",
        "servers",
        "fleet",
        "fleets",
        "replica",
        "replicas",
        "host",
        "hosts",
        "pod",
        "pods",
    }
)
#: Characters stripped from both ends of a word before it is compared.
_EDGE_PUNCT: Final = "\"'`*_()[]{}<>.,;:!?“”‘’«»"


def _token(word: str) -> str:
    return word.strip(_EDGE_PUNCT).replace("’", "'").casefold()


def words(text: str) -> list[str]:
    """Comparison words of ``text``: split on whitespace, edge punctuation removed, casefolded."""
    return [t for t in (_token(w) for w in text.split()) if t]


def is_state_phrase(tokens: Sequence[str]) -> bool:
    """Whether ``tokens`` (see :func:`words`) only say that something is failing.

    Every word must be a state word or filler, and at least one must carry the state: a core word
    ("down", "outage") or a negated ordinary one ("not responding"). "is working" is no state.
    """
    if not tokens:
        return False
    stated = False
    for index, token in enumerate(tokens):
        negated = token in _NEGATABLE and index > 0 and tokens[index - 1] in _NEGATIONS
        if token in _STATE_CORE or negated:
            stated = True
        elif token not in _STATE_FILLER and token not in _NEGATIONS:
            return False
    return stated


def _base(raw: Sequence[str]) -> str:
    return " ".join(raw).strip(_EDGE_PUNCT + " -–—")


def state_base(surface: str, is_known: Callable[[str], bool]) -> str | None:
    """The already-named component ``surface`` describes in a state, or ``None``.

    ``"<name> is down"`` / ``"<name> outage"`` (state phrase after the name) and ``"Outage of
    <name>"`` (before it) give ``<name>`` when ``is_known(<name>)`` — the document (or the catalog)
    names it elsewhere. The longest known name wins. A surface that is only a state phrase, or
    whose remainder is anything but a state phrase ("<name> outage runbook"), is left alone.
    """
    raw = surface.split()
    if len(raw) < 2:
        return None
    tokens = [_token(w) for w in raw]
    for cut in range(len(raw) - 1, 0, -1):
        rest = [t for t in tokens[cut:] if t]
        if is_state_phrase(rest):
            base = _base(raw[:cut])
            if base and is_known(base):
                return base
    for cut in range(1, len(raw)):
        lead = [t for t in tokens[:cut] if t]
        if is_state_phrase(lead):
            base = _base(raw[cut:])
            if base and is_known(base):
                return base
    return None


def deployment_base(surface: str, is_known: Callable[[str], bool]) -> str | None:
    """The named component whose deployment ``surface`` names, or ``None``.

    ``"<name> clusters"`` / ``"<name> nodes"`` / ``"<name> fleet"`` give ``<name>`` when
    ``is_known(<name>)``: the cluster of a component is that component, deployed. Only trailing
    deployment nouns count ("<name> cluster manager" names something else).
    """
    raw = surface.split()
    for cut in range(len(raw) - 1, 0, -1):
        rest = [t for t in (_token(w) for w in raw[cut:]) if t]
        if rest and all(t in _DEPLOYMENT_UNITS for t in rest):
            base = _base(raw[:cut])
            if base and is_known(base):
                return base
    return None


def is_sentence(text: str) -> bool:
    """Whether ``text`` is a clause rather than a name: it contains a finite verb or auxiliary
    ("<x> is down", "what to do when <x> has failed"). Names do not conjugate."""
    return any(t in _FINITE_VERBS for t in words(text))
