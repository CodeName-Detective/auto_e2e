"""Lineage records: a hash chain from raw bytes to every evaluation.

Three write-once records, each canonical JSON identified by its SHA-256:

- dataset lineage: one per published dataset, with a ``units.jsonl`` line
  per unit naming its raw tar and label record;
- checkpoint lineage: one per distinct checkpoint, naming its dataset
  lineage and its parent checkpoint lineage;
- evaluation record: one per checkpoint, suite, and evaluation dataset.

A record refers to another only by URI plus SHA-256, so changing any byte
upstream breaks every digest downstream.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes


DATASET_LINEAGE_SCHEMA = "autoe2e_dataset_lineage_v1"
CHECKPOINT_LINEAGE_SCHEMA = "autoe2e_checkpoint_lineage_v1"
EVALUATION_SCHEMA = "autoe2e_evaluation_v1"
UNKNOWN = {"status": "unknown"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNIT_FIELDS = (
    "unit_id",
    "partition_id",
    "raw_tar_key",
    "raw_tar_sha256",
    "raw_index_sha256",
    "label_id",
    "label_record_sha256",
    "label_origin",
    "sample_count",
    "label_count",
)


def record_bytes(record: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(record))


def record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_bytes(record_bytes(record))


def _sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{what} must be a SHA-256 hex digest, got {value!r}")
    return value


@dataclass(frozen=True)
class UnitLineage:
    unit_id: str
    partition_id: str
    raw_tar_key: str
    raw_tar_sha256: str
    raw_index_sha256: str
    label_id: Optional[str]
    label_record_sha256: Optional[str]
    label_origin: Optional[str]  # "adopted", "created", or None without labels
    sample_count: int
    label_count: int


def units_jsonl(units: Sequence[UnitLineage]) -> bytes:
    """Serialize unit rows as canonical JSON Lines sorted by unit id.

    JSON Lines, not Parquet, so the bytes never depend on a library version
    and a write-once rerun reproduces them exactly.
    """
    rows = sorted(units, key=lambda unit: unit.unit_id)
    ids = [unit.unit_id for unit in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("units must be non-empty and unique")
    for unit in rows:
        _sha(unit.raw_tar_sha256, f"{unit.unit_id} raw_tar_sha256")
        _sha(unit.raw_index_sha256, f"{unit.unit_id} raw_index_sha256")
        labelled = unit.label_id is not None
        if labelled != (unit.label_record_sha256 is not None) or labelled != (
            unit.label_origin is not None
        ):
            raise ValueError(f"{unit.unit_id}: label fields must be all set or all null")
        if labelled:
            _sha(unit.label_id, f"{unit.unit_id} label_id")
            _sha(unit.label_record_sha256, f"{unit.unit_id} label_record_sha256")
            if unit.label_origin not in {"adopted", "created"}:
                raise ValueError(f"{unit.unit_id}: invalid label_origin")
        if unit.sample_count < 0 or unit.label_count < 0:
            raise ValueError(f"{unit.unit_id}: counts must not be negative")
    return b"".join(canonical_json_bytes(asdict(unit)) + b"\n" for unit in rows)


def read_units_jsonl(payload: bytes) -> list[UnitLineage]:
    units = []
    for line in payload.splitlines():
        row = json.loads(line)
        if tuple(sorted(row)) != tuple(sorted(_UNIT_FIELDS)):
            raise ValueError(f"unit row fields {sorted(row)} != {sorted(_UNIT_FIELDS)}")
        units.append(UnitLineage(**row))
    if units_jsonl(units) != payload:
        raise ValueError("units.jsonl is not canonical")
    return units


def dataset_lineage(
    *,
    dataset_id: str,
    spec: Mapping[str, Any],
    published: Mapping[str, str],
    raw: Mapping[str, Any],
    labels: Optional[Mapping[str, Any]],
    units: Sequence[UnitLineage],
    units_uri: str,
    views: Mapping[str, Any],
    console: Optional[Mapping[str, Any]],
    pack: Mapping[str, Any],
    build: Mapping[str, Any],
    packed_reasoning_labels: Optional[int] = None,
) -> tuple[dict[str, Any], bytes]:
    """Return ``(dataset.json record, units.jsonl bytes)``."""
    from Platform.pipelines.autoe2e_labels import label_set_sha256

    _sha(dataset_id, "dataset_id")
    if sha256_bytes(canonical_json_bytes(dict(spec))) != dataset_id:
        raise ValueError("dataset_id is not the digest of the given spec")
    _sha(raw["snapshot_manifest_sha256"], "raw snapshot_manifest_sha256")
    payload = units_jsonl(units)
    _sha(views["sha256"], "views sha256")
    partition_ids = set(views["partition_ids"])
    # nuPlan archives feed several sensor-group partitions; they say "*".
    if not {u.partition_id for u in units if u.partition_id != "*"} <= partition_ids:
        raise ValueError("units reference partitions that are not published")
    if console is not None:
        _sha(console["manifest_sha256"], "console manifest_sha256")
    labelled = [u for u in units if u.label_id is not None]
    if labels is None:
        if labelled:
            raise ValueError("units carry labels but the dataset has none")
        label_summary = None
    else:
        if len(labelled) != len(units):
            raise ValueError("a labelled dataset needs a label set for every unit")
        label_summary = {
            **dict(labels),
            "label_set_sha256": label_set_sha256(
                (u.unit_id, u.label_id, u.label_record_sha256) for u in labelled
            ),
            "adopted": sum(u.label_origin == "adopted" for u in labelled),
            "created": sum(u.label_origin == "created" for u in labelled),
        }
    record = {
        "schema_version": DATASET_LINEAGE_SCHEMA,
        "dataset_id": dataset_id,
        "spec": dict(spec),
        "published": dict(published),
        "raw": {**dict(raw), "units": len(units)},
        "labels": label_summary,
        "units": {"uri": units_uri, "sha256": sha256_bytes(payload), "rows": len(units)},
        "views": {"key": views["key"], "sha256": views["sha256"], "partitions": len(partition_ids)},
        "console": None if console is None else dict(console),
        "pack": dict(pack),
        "build": dict(build),
        # reasoning_labels counts teacher records; packed_reasoning_labels counts
        # the samples that actually carry one after the pack JOIN, so a parser
        # change that drops labelled frames shows up as a gap between the two.
        "totals": {
            "samples": sum(u.sample_count for u in units),
            "reasoning_labels": sum(u.label_count for u in units),
            "packed_reasoning_labels": packed_reasoning_labels,
        },
    }
    return record, payload


def checkpoint_lineage(
    *,
    checkpoint: Mapping[str, Any],
    parent: Optional[Mapping[str, Any]],
    dataset: Optional[Mapping[str, Any]],
    split: Optional[Mapping[str, Any]],
    training: Optional[Mapping[str, Any]],
    selection: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a checkpoint lineage record.

    ``dataset``, ``split``, and ``training`` are None only for an imported
    checkpoint, and are then written as ``{"status": "unknown"}``.
    """
    _sha(checkpoint["sha256"], "checkpoint sha256")
    if checkpoint.get("role") not in {"best", "final", "imported"}:
        raise ValueError(f"invalid checkpoint role {checkpoint.get('role')!r}")
    imported = checkpoint["role"] == "imported"
    known = [dataset, split, training]
    if imported and any(value is not None for value in known):
        raise ValueError("an imported checkpoint has unknown dataset, split, and training")
    if not imported and any(value is None for value in known):
        raise ValueError("a trained checkpoint needs dataset, split, and training")
    if imported and parent is not None:
        raise ValueError("an imported checkpoint has no known parent")
    if parent is not None:
        _sha(parent["checkpoint_sha256"], "parent checkpoint_sha256")
        _sha(parent["lineage_sha256"], "parent lineage_sha256")
        stage_index = int(parent["stage_index"]) + 1
    else:
        stage_index = 0 if imported else 1
    if dataset is not None:
        _sha(dataset["dataset_id"], "dataset_id")
        _sha(dataset["lineage_sha256"], "dataset lineage_sha256")
    return {
        "schema_version": CHECKPOINT_LINEAGE_SCHEMA,
        "checkpoint": dict(checkpoint),
        "stage_index": stage_index,
        "parent": None if parent is None else dict(parent),
        "dataset": UNKNOWN if dataset is None else dict(dataset),
        "split": UNKNOWN if split is None else dict(split),
        "training": UNKNOWN if training is None else dict(training),
        "selection": None if selection is None else dict(selection),
    }


def evaluation_record(
    *,
    checkpoint_sha256: str,
    checkpoint_lineage_sha256: str,
    suite: str,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    metrics: Mapping[str, float],
    per_sample: Optional[Mapping[str, Any]],
    code: Mapping[str, str],
    flyte_execution_id: str,
    finished_at: str,
    model_reexecuted: bool = True,
) -> dict[str, Any]:
    """Build an evaluation record.

    ``model_reexecuted`` is False only for metrics the training run measured
    on its own validation split; those carry no per-sample output.
    """
    _sha(checkpoint_sha256, "checkpoint_sha256")
    _sha(checkpoint_lineage_sha256, "checkpoint_lineage_sha256")
    _sha(dataset["dataset_id"], "evaluation dataset_id")
    if model_reexecuted:
        if per_sample is None:
            raise ValueError("a re-executed evaluation must pin its per-sample output")
        _sha(per_sample["sha256"], "per_sample sha256")
    elif per_sample is not None:
        raise ValueError("training-time validation has no per-sample output")
    if not metrics:
        raise ValueError("an evaluation needs at least one metric")
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"metric {name} is not a finite number")
    return {
        "schema_version": EVALUATION_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_lineage_sha256": checkpoint_lineage_sha256,
        "suite": suite,
        "dataset": dict(dataset),
        "protocol": dict(protocol),
        "model_reexecuted": bool(model_reexecuted),
        "metrics": {name: float(value) for name, value in sorted(metrics.items())},
        "per_sample": None if per_sample is None else dict(per_sample),
        "code": dict(code),
        "flyte_execution_id": flyte_execution_id,
        "finished_at": finished_at,
    }


def parse_record(payload: bytes, *, expected_sha256: str, schema: str) -> dict[str, Any]:
    """Parse a record only if its bytes are canonical and match the digest."""
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{schema} bytes differ from the recorded SHA-256")
    record = json.loads(payload)
    if record.get("schema_version") != schema:
        raise ValueError(f"expected {schema}, got {record.get('schema_version')!r}")
    if record_bytes(record) != payload:
        raise ValueError(f"{schema} is not canonical JSON")
    return record
