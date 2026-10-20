"""Dataset adapters: the only place where wf_autoe2e knows a dataset.

An adapter turns a pinned raw snapshot manifest into a ``SourcePlan`` (units
and pack partitions) and names the published dataset. The workflow, label
store, publication, lineage, training, and MLflow code never branch on the
dataset; they ask the adapter.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, DatasetSpec
from Platform.pipelines.autoe2e_snapshot import iter_units, parse_manifest


KITSCENES_UPSTREAM = "KIT-MRT/KITScenes-Multimodal"
L2D_UPSTREAM = "yaak-ai/L2D"
NUPLAN_SNAPSHOT_SCHEMA = "nuplan_raw_snapshot_v1"
_KITSCENES_SPLIT_NAMES = {"train": "kitscenes", "val": "kitscenes-val", "test": "kitscenes-test"}
_NUPLAN_PACK_PARAMS = {"total_scenario_limit", "samples_per_shard", "max_rejection_fraction"}
_NUPLAN_PACK_DEFAULTS = {
    "total_scenario_limit": "131072",
    "samples_per_shard": "100",
    "max_rejection_fraction": "0.25",
}
# Must equal the cache_version of pack_nuplan_snapshot_reactive_dataset; a test
# pins the two together so a nuPlan pack change always changes dataset_id.
NUPLAN_PACK_TASK_VERSION = "nuplan-snapshot-pack-v13-manifest-v10"


def pack_contracts() -> dict[str, str]:
    """Pack contracts of the running code, as written into pack manifests."""
    from data_processing.contract_versions import contract_versions

    return dict(contract_versions())


def current_pack_version(dataset: AutoE2EDataset) -> str:
    """The only pack_version a DatasetSpec may carry with this code.

    Derived from the contract constants, so bumping any contract changes
    dataset_id and a stale publication is never reused.
    """
    contracts = ".".join(f"{key}={value}" for key, value in sorted(pack_contracts().items()))
    if dataset is AutoE2EDataset.NUPLAN:
        return f"{contracts}+{NUPLAN_PACK_TASK_VERSION}"
    return contracts


def _check_pack_version(spec: DatasetSpec) -> None:
    expected = current_pack_version(spec.dataset)
    if spec.pack_version != expected:
        raise ValueError(
            f"pack_version {spec.pack_version!r} is not this code's {expected!r}"
        )


@dataclass
class UnitRef:
    unit_id: str
    tar_key: str
    tar_sha256: str
    index_key: str
    index_sha256: str


@dataclass
class SourcePlan:
    """Everything the build needs from a snapshot, as plain Flyte data."""

    dataset: AutoE2EDataset
    upstream: str
    revision: str
    split: str
    published_name: str
    snapshot_manifest_sha256: str
    units: List[UnitRef] = field(default_factory=list)
    partitions: List[List[str]] = field(default_factory=list)
    labels: bool = False
    pack_params: Dict[str, str] = field(default_factory=dict)


def _int_param(params: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    raw = params.get(key, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"pack_params[{key!r}] must be an integer, got {raw!r}") from error
    if value < minimum:
        raise ValueError(f"pack_params[{key!r}] must be at least {minimum}")
    return value


def _reject_unknown(params: Mapping[str, str], allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unsupported pack_params for this dataset: {unknown}")


def _unit_plan(spec: DatasetSpec, payload: bytes, *, upstream: str) -> tuple[dict, list[UnitRef]]:
    manifest = parse_manifest(payload, expected_sha256=spec.source_manifest.sha256)
    if manifest["dataset"] != spec.dataset.value:
        raise ValueError(
            f"snapshot is for {manifest['dataset']!r}, spec asks for {spec.dataset.value!r}"
        )
    if manifest["upstream"] != upstream:
        raise ValueError(f"snapshot upstream {manifest['upstream']!r} is not {upstream!r}")
    units = [
        UnitRef(
            unit_id=unit["unit_id"],
            tar_key=unit["tar_key"],
            tar_sha256=unit["tar_sha256"],
            index_key=unit["index_key"],
            index_sha256=unit["index_sha256"],
        )
        for unit in iter_units(manifest, spec.unit_limit)
    ]
    if spec.unit_limit and len(units) < spec.unit_limit:
        raise ValueError(f"snapshot has {len(units)} units, unit_limit is {spec.unit_limit}")
    return manifest, units


def read_kitscenes(spec: DatasetSpec, payload: bytes) -> SourcePlan:
    _check_pack_version(spec)
    _reject_unknown(spec.pack_params, set())
    manifest, units = _unit_plan(spec, payload, upstream=KITSCENES_UPSTREAM)
    split = str(manifest["provenance"].get("split", ""))
    if split not in _KITSCENES_SPLIT_NAMES:
        raise ValueError(f"KITScenes snapshot split {split!r} is not train, val, or test")
    if spec.reasoning is not None and split != "train":
        raise ValueError("KITScenes evaluation splits are not reasoning-labelled")
    return SourcePlan(
        dataset=spec.dataset,
        upstream=KITSCENES_UPSTREAM,
        revision=manifest["revision"],
        split=split,
        published_name=_KITSCENES_SPLIT_NAMES[split],
        snapshot_manifest_sha256=spec.source_manifest.sha256,
        units=units,
        partitions=[[unit.unit_id] for unit in units],
        labels=spec.reasoning is not None,
    )


def read_l2d(spec: DatasetSpec, payload: bytes) -> SourcePlan:
    _check_pack_version(spec)
    _reject_unknown(spec.pack_params, {"partition_size", "osm_graph_snapshot"})
    if "osm_graph_snapshot" not in spec.pack_params:
        raise ValueError(
            "L2D Reactive packs need OSM navigation (14 map channels); "
            "set pack_params['osm_graph_snapshot']"
        )
    raise NotImplementedError(
        "the L2D packer does not receive osm_graph_snapshot_path from any task yet, "
        "so it would emit 3 map channels that Reactive training rejects"
    )


def read_nuplan(spec: DatasetSpec, payload: bytes) -> SourcePlan:
    """Plan a nuPlan build from the existing ``nuplan_raw_snapshot_v1`` manifest.

    Partitions are the 43 train sensor groups; they need an S3 read of the
    sensor inventory, so they are resolved by the build task, not here.
    """
    import json

    from Platform.pipelines.dataset_publication import sha256_bytes

    _check_pack_version(spec)
    _reject_unknown(spec.pack_params, _NUPLAN_PACK_PARAMS)
    if spec.pack_params != normalized_nuplan_pack_params(spec.pack_params):
        raise ValueError(
            "nuPlan pack_params must list every parameter explicitly: "
            f"{normalized_nuplan_pack_params(spec.pack_params)}"
        )
    if sha256_bytes(payload) != spec.source_manifest.sha256:
        raise ValueError("snapshot manifest bytes differ from the pinned SHA-256")
    manifest = json.loads(payload)
    if manifest.get("schema_version") != NUPLAN_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported nuPlan snapshot schema")
    if spec.reasoning is not None:
        raise ValueError("nuPlan has no reasoning labels")
    if spec.unit_limit:
        raise ValueError("nuPlan partitions are sensor groups; unit_limit is not supported")
    _int_param(spec.pack_params, "total_scenario_limit", 131_072, minimum=1)
    _int_param(spec.pack_params, "samples_per_shard", 100, minimum=1)
    fraction = float(spec.pack_params["max_rejection_fraction"])
    if not 0.0 <= fraction < 1.0:
        raise ValueError("max_rejection_fraction must be in [0, 1)")
    archives = manifest.get("archives") or []
    units = [
        UnitRef(
            unit_id=str(archive["archive_id"]),
            tar_key=str(archive["object_uri"]).split("/", 3)[3],
            tar_sha256=str(archive["sha256"]),
            index_key="",
            index_sha256="",
        )
        for archive in sorted(archives, key=lambda item: str(item["archive_id"]))
    ]
    return SourcePlan(
        dataset=spec.dataset,
        upstream=str(manifest["dataset"]),
        revision=str(manifest["dataset_revision"]),
        split="train",
        published_name="nuplan",
        snapshot_manifest_sha256=spec.source_manifest.sha256,
        units=units,
        partitions=[],
        labels=False,
        pack_params=dict(spec.pack_params),
    )


def normalized_nuplan_pack_params(params: Mapping[str, str]) -> dict[str, str]:
    """nuPlan pack_params with every default filled, so equal meaning is equal id."""
    return {**_NUPLAN_PACK_DEFAULTS, **dict(params)}


_READERS = {
    AutoE2EDataset.KITSCENES: read_kitscenes,
    AutoE2EDataset.L2D: read_l2d,
    AutoE2EDataset.NUPLAN: read_nuplan,
}


def read_source(spec: DatasetSpec, payload: bytes) -> SourcePlan:
    return _READERS[spec.dataset](spec, payload)


def extract_unit_tar(stream: io.BufferedIOBase, unit_id: str, root: Path) -> list[str]:
    """Extract one unit tar under ``root``, stripping the ``<unit_id>/`` prefix.

    Members of different units may share a path only if their bytes are
    identical (for example KITScenes ``data/sequence_archives.csv``).
    Returns the relative paths written.
    """
    import hashlib

    written = []
    prefix = f"{unit_id}/"
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for info in tar:
            if not info.isfile() or not info.name.startswith(prefix):
                raise ValueError(f"unexpected tar member {info.name!r} in unit {unit_id}")
            relative = info.name[len(prefix):]
            parts = relative.split("/")
            if not relative or any(part in {"", ".", ".."} for part in parts):
                raise ValueError(f"unsafe member path {info.name!r}")
            target = root.joinpath(*parts)
            handle = tar.extractfile(info)
            assert handle is not None
            body = handle.read()
            if target.exists():
                existing = target.read_bytes()
                if hashlib.sha256(existing).digest() != hashlib.sha256(body).digest():
                    raise ValueError(f"units disagree on shared file {relative!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.append(relative)
    return written
