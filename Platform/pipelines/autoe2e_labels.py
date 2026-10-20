"""Pure contracts for the per-unit reasoning label store.

Raw bytes live in raw snapshots (``autoe2e_snapshot``). Labels are produced
by billed teacher calls, so each unit's label set is stored once under
``<dataset>/labels/<label_id>/`` with a write-once record at
``<dataset>/_stages/labels/<label_id>.json``. The label identity excludes
parser, shard, and geometry versions, so a pack contract bump never re-labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.dataset_publication import (
    canonical_json_bytes,
    sha256_bytes,
)


LABEL_RECORD_SCHEMA = "autoe2e_unit_labels_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class LabelIdentity:
    teacher: str
    prompt_version: str
    label_stride: int
    label_policy_version: str
    uid_schema_version: str


def label_identity(
    dataset: AutoE2EDataset,
    revision: str,
    unit_id: str,
    labels: LabelIdentity,
) -> dict[str, Any]:
    if not revision:
        raise ValueError("revision is required")
    if not _UNIT_ID_RE.fullmatch(unit_id):
        raise ValueError(f"invalid unit id {unit_id!r}")
    if not labels.teacher or not labels.prompt_version:
        raise ValueError("teacher and prompt_version are required")
    if labels.label_stride <= 0:
        raise ValueError("label_stride must be positive")
    if not labels.label_policy_version or not labels.uid_schema_version:
        raise ValueError("label policy and uid schema versions are required")
    return {
        "dataset": dataset.value,
        "revision": revision,
        "unit_id": unit_id,
        "teacher": labels.teacher,
        "prompt_version": labels.prompt_version,
        "label_stride": labels.label_stride,
        "label_policy_version": labels.label_policy_version,
        "uid_schema_version": labels.uid_schema_version,
    }


def label_id(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(identity)))


def label_prefix(dataset: AutoE2EDataset, digest: str) -> str:
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"invalid label id {digest!r}")
    return f"{dataset.value}/labels/{digest}"


def label_record_key(dataset: AutoE2EDataset, digest: str) -> str:
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"invalid label id {digest!r}")
    return f"{dataset.value}/_stages/labels/{digest}.json"


def listing_digest(objects: Iterable[Mapping[str, Any]]) -> tuple[str, int, int]:
    """Digest a listing of ``relative``, ``size``, and ``etag``, order-free.

    Returns ``(digest, object_count, total_bytes)``.
    """
    rows = sorted(
        (str(item["relative"]), int(item["size"]), str(item["etag"]).strip('"'))
        for item in objects
    )
    if not rows:
        raise ValueError("label artifact is empty")
    names = [row[0] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("label artifact lists a relative key twice")
    digest = sha256_bytes(canonical_json_bytes([list(row) for row in rows]))
    return digest, len(rows), sum(row[1] for row in rows)


def label_record(
    identity: Mapping[str, Any],
    *,
    bucket: str,
    objects: Iterable[Mapping[str, Any]],
    label_count: int,
) -> dict[str, Any]:
    digest = label_id(identity)
    dataset = AutoE2EDataset(identity["dataset"])
    listing, count, total = listing_digest(objects)
    if label_count < 0:
        raise ValueError("label_count must not be negative")
    return {
        "schema_version": LABEL_RECORD_SCHEMA,
        "label_id": digest,
        "identity": dict(identity),
        "artifact_uri": f"s3://{bucket}/{label_prefix(dataset, digest)}",
        "object_count": count,
        "total_bytes": total,
        "listing_sha256": listing,
        "label_count": int(label_count),
    }


def validate_label_record(
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    bucket: str,
    objects: Iterable[Mapping[str, Any]],
) -> None:
    """Reject a record for another identity or whose files changed."""
    if record.get("schema_version") != LABEL_RECORD_SCHEMA:
        raise ValueError("unsupported label record schema")
    if record.get("identity") != dict(identity):
        raise ValueError("label record identity differs from the requested one")
    expected = label_record(
        identity,
        bucket=bucket,
        objects=objects,
        label_count=int(record.get("label_count", -1)),
    )
    if dict(record) != expected:
        raise ValueError(
            f"label artifact changed since it was recorded: {record.get('artifact_uri')}"
        )


def label_set_sha256(rows: Iterable[tuple[str, str, str]]) -> str:
    """Digest of sorted ``(unit_id, label_id, record_sha256)`` rows."""
    ordered = sorted(rows)
    if len({row[0] for row in ordered}) != len(ordered):
        raise ValueError("label set lists a unit twice")
    return sha256_bytes(canonical_json_bytes([list(row) for row in ordered]))


def validate_label_meta(meta: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Check a ``generate_reasoning_labels`` ``meta.json`` against an identity.

    The meta does not record ``label_stride`` or the uid schema, so those come
    from the producing execution's inputs.
    """
    expected = {
        "source_revision": identity["revision"],
        "teacher": identity["teacher"],
        "prompt_version": identity["prompt_version"],
        "label_policy_version": identity["label_policy_version"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"label meta {key}={meta.get(key)!r} differs from {value!r}"
            )
