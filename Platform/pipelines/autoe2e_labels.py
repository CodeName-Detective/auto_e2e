"""Pure contracts for the per-scene ingest and label stage store.

Raw ingest and reasoning labels do not depend on the pack contract, so their
identities exclude parser, shard, and geometry versions. A pack contract bump
therefore never repeats a download or a teacher call.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.dataset_publication import (
    canonical_json_bytes,
    sha256_bytes,
)


STAGE_RECORD_SCHEMA = "autoe2e_scene_stage_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_S3_PREFIX_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.-]+/.+[^/]$")


class SceneStage(enum.Enum):
    RAW = "raw"
    LABELS = "labels"


@dataclass(frozen=True)
class LabelIdentity:
    teacher: str
    prompt_version: str
    label_stride: int
    label_policy_version: str
    uid_schema_version: str


def _require_scene_id(scene_id: str) -> None:
    if not _SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"invalid scene id {scene_id!r}")


def raw_identity(
    dataset: AutoE2EDataset,
    source_revision: str,
    scene_id: str,
) -> dict[str, Any]:
    if not source_revision:
        raise ValueError("source_revision is required")
    _require_scene_id(scene_id)
    return {
        "stage": SceneStage.RAW.value,
        "dataset": dataset.value,
        "source_revision": source_revision,
        "scene_id": scene_id,
    }


def label_identity(
    dataset: AutoE2EDataset,
    source_revision: str,
    scene_id: str,
    labels: LabelIdentity,
) -> dict[str, Any]:
    if not labels.teacher or not labels.prompt_version:
        raise ValueError("teacher and prompt_version are required")
    if labels.label_stride <= 0:
        raise ValueError("label_stride must be positive")
    if not labels.label_policy_version or not labels.uid_schema_version:
        raise ValueError("label policy and uid schema versions are required")
    return {
        "stage": SceneStage.LABELS.value,
        "raw": raw_identity(dataset, source_revision, scene_id),
        "teacher": labels.teacher,
        "prompt_version": labels.prompt_version,
        "label_stride": labels.label_stride,
        "label_policy_version": labels.label_policy_version,
        "uid_schema_version": labels.uid_schema_version,
    }


def stage_id(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(identity)))


def stage_record_key(dataset: AutoE2EDataset, stage: SceneStage, digest: str) -> str:
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"invalid stage id {digest!r}")
    return f"{dataset.value}/_stages/{stage.value}/{digest}.json"


def listing_digest(objects: Iterable[Mapping[str, Any]]) -> tuple[str, int, int]:
    """Digest an artifact listing of ``relative``, ``size``, and ``etag``.

    Returns ``(digest, object_count, total_bytes)``. Order does not matter.
    """
    rows = sorted(
        (str(item["relative"]), int(item["size"]), str(item["etag"]).strip('"'))
        for item in objects
    )
    if not rows:
        raise ValueError("stage artifact is empty")
    names = [row[0] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("stage artifact lists a relative key twice")
    digest = sha256_bytes(canonical_json_bytes([list(row) for row in rows]))
    return digest, len(rows), sum(row[1] for row in rows)


def stage_record(
    identity: Mapping[str, Any],
    *,
    artifact_uri: str,
    objects: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _S3_PREFIX_RE.fullmatch(artifact_uri):
        raise ValueError(f"stage artifact is not an S3 prefix: {artifact_uri!r}")
    digest, count, total = listing_digest(objects)
    return {
        "schema_version": STAGE_RECORD_SCHEMA,
        "stage_id": stage_id(identity),
        "identity": dict(identity),
        "artifact_uri": artifact_uri,
        "object_count": count,
        "total_bytes": total,
        "listing_sha256": digest,
    }


def validate_stage_record(
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    objects: Iterable[Mapping[str, Any]],
) -> None:
    """Reject a record that is for another identity or whose artifact changed."""
    if record.get("schema_version") != STAGE_RECORD_SCHEMA:
        raise ValueError("unsupported stage record schema")
    if record.get("identity") != dict(identity):
        raise ValueError("stage record identity differs from the requested one")
    if record.get("stage_id") != stage_id(identity):
        raise ValueError("stage record id does not match its identity")
    expected = stage_record(
        identity,
        artifact_uri=str(record.get("artifact_uri")),
        objects=objects,
    )
    if dict(record) != expected:
        raise ValueError(
            f"stage artifact changed since it was recorded: {record.get('artifact_uri')}"
        )


def validate_label_meta(meta: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Check a generate_reasoning_labels ``meta.json`` against a label identity.

    The meta does not record ``label_stride`` or the uid schema, so those come
    from the caller's provenance (the producing execution's inputs).
    """
    raw = identity["raw"]
    expected = {
        "source_revision": raw["source_revision"],
        "teacher": identity["teacher"],
        "prompt_version": identity["prompt_version"],
        "label_policy_version": identity["label_policy_version"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"label meta {key}={meta.get(key)!r} differs from {value!r}"
            )
