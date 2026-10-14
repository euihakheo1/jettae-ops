"""Seal the evaluation set: per-case SHA-256 hashes and a Merkle root in a manifest.

Why seal: a test set that can be edited after seeing the results proves nothing. The manifest
commits to every case byte-for-byte (canonical JSON), so anyone can re-generate the corpus and
check it is the one that was evaluated (P6). A separate root over the held-out cases lets the
held-out commitment be published while those cases stay unopened.

The Merkle tree follows RFC 6962 §2.1 (domain-separated leaf/node hashes, split at the largest
power of two), computed over the *sorted* case hashes so the root does not depend on case order.
No timestamps: the manifest of the same corpus is byte-identical on every run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from jettae.evalkit.corpus import GENERATOR_VERSION, EvalCase
from jettae.evalkit.mutate import ALL_OPERATORS, HELD_OUT_OPERATORS

DEFAULT_CREATED_BY = "jettae.evalkit.seal"


class SealManifest(BaseModel):
    """The published commitment. Unknown keys are rejected: a seal must not silently ignore input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_by: str
    generator_version: str
    n_cases: int
    operators_active: tuple[str, ...]
    operators_held_out: tuple[str, ...]
    #: sorted SHA-256 hex digests of the cases' canonical JSON
    case_hashes: tuple[str, ...]
    merkle_root: str
    held_out_merkle_root: str


def case_hash(case: EvalCase) -> str:
    return hashlib.sha256(case.canonical_json().encode("utf-8")).hexdigest()


def merkle_root(leaf_hashes: Sequence[str]) -> str:
    """RFC 6962 Merkle tree hash over hex leaf digests (in the given order)."""
    return _mth([bytes.fromhex(h) for h in leaf_hashes]).hex()


def _mth(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return hashlib.sha256(b"\x00" + leaves[0]).digest()
    split = 1 << ((len(leaves) - 1).bit_length() - 1)  # largest power of two < n
    return hashlib.sha256(b"\x01" + _mth(leaves[:split]) + _mth(leaves[split:])).digest()


def build_manifest(
    cases: Iterable[EvalCase],
    generator_version: str = GENERATOR_VERSION,
    *,
    created_by: str = DEFAULT_CREATED_BY,
) -> SealManifest:
    """Seal ``cases``. Every case must carry ``generator_version``: the version is part of each
    case's canonical JSON, so the Merkle root commits to it and a relabelled manifest fails."""
    hashed = [(case_hash(c), c) for c in cases]
    if stray := sorted({c.generator_version for _, c in hashed} - {generator_version}):
        raise ValueError(f"생성기 버전이 다른 사례가 섞여 있습니다: {stray} (봉인 버전 {generator_version})")
    hashes = sorted(h for h, _ in hashed)
    held_out = sorted(h for h, c in hashed if c.held_out)
    present = {c.operator for _, c in hashed if c.operator is not None}
    return SealManifest(
        created_by=created_by,
        generator_version=generator_version,
        n_cases=len(hashed),
        operators_active=tuple(
            op.value for op in ALL_OPERATORS if op in present and op not in HELD_OUT_OPERATORS
        ),
        operators_held_out=tuple(op.value for op in HELD_OUT_OPERATORS if op in present),
        case_hashes=tuple(hashes),
        merkle_root=merkle_root(hashes),
        held_out_merkle_root=merkle_root(held_out),
    )


def verify(cases: Iterable[EvalCase], manifest: SealManifest) -> bool:
    """True iff ``cases`` are exactly the sealed corpus (any edit, addition or removal fails)."""
    case_list = list(cases)
    if any(c.generator_version != manifest.generator_version for c in case_list):
        return False
    rebuilt = build_manifest(case_list, manifest.generator_version, created_by=manifest.created_by)
    return rebuilt == manifest


def write_manifest(manifest: SealManifest, path: str | Path) -> None:
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
    Path(path).write_bytes((payload + "\n").encode("utf-8"))


def read_manifest(path: str | Path) -> SealManifest:
    return SealManifest.model_validate_json(Path(path).read_bytes().decode("utf-8"))
