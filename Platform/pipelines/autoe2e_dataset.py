"""Pure contracts for resolving a wf_autoe2e dataset by content identity.

A dataset is identified by ``dataset_id``: the SHA-256 of the canonical JSON
of the fields that decide its bytes. Publications keep the Console-facing
``<dataset>/vN.M/`` layout; ``<dataset>/by-id/<dataset_id>.json`` is written
after the manifest and maps the identity to that version.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    canonical_json_bytes,
    dataset_prefix,
    sha256_bytes,
)


DATASET_SPEC_SCHEMA = "autoe2e_dataset_spec_v1"
DATASET_INDEX_SCHEMA = "autoe2e_dataset_index_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_S3_URI_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.-]+/.+[^/]$")
_VERSION_PARTS_RE = re.compile(r"^v([0-9]+)(?:\.([0-9]+))?$")


class AutoE2EDataset(enum.Enum):
    """Published dataset ids that wf_autoe2e can build."""

    KITSCENES = "kitscenes"
    NUPLAN = "nuplan"
    L2D = "l2d"


@dataclass(frozen=True)
class ManifestRef:
    """An immutable S3 object pinned by version id and content digest."""

    uri: str
    version_id: str
    sha256: str


@dataclass(frozen=True)
class ReasoningSpec:
    teacher: str
    teacher_model: str
    prompt_version: str
    label_stride: int


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset contains. Every field enters ``dataset_id``.

    ``source_manifest`` is a raw snapshot manifest; it pins the unit list and
    the upstream revision. ``pack_params`` carries adapter-specific pack
    inputs as strings so the workflow signature is the same for every dataset.
    """

    dataset: AutoE2EDataset
    source_manifest: ManifestRef
    pack_version: str
    image_size: int
    unit_limit: int = 0
    pack_params: Dict[str, str] = field(default_factory=dict)
    reasoning: Optional[ReasoningSpec] = None


def validate_manifest_ref(ref: ManifestRef, name: str) -> None:
    if not _S3_URI_RE.fullmatch(ref.uri):
        raise ValueError(f"{name} uri is not an S3 object: {ref.uri!r}")
    if not ref.version_id:
        raise ValueError(f"{name} version_id is required")
    if not _SHA256_RE.fullmatch(ref.sha256):
        raise ValueError(f"{name} sha256 must be 64 lowercase hex digits")


def validate_dataset_spec(spec: DatasetSpec) -> None:
    if not isinstance(spec.dataset, AutoE2EDataset):
        raise ValueError(f"unsupported dataset {spec.dataset!r}")
    validate_manifest_ref(spec.source_manifest, "source_manifest")
    if spec.unit_limit < 0:
        raise ValueError("unit_limit must be zero or positive")
    for key, value in spec.pack_params.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ValueError(f"pack_params must map non-empty strings: {key!r}")
    if not spec.pack_version:
        raise ValueError("pack_version is required")
    if spec.image_size <= 0:
        raise ValueError("image_size must be positive")
    if spec.reasoning is not None:
        if not (
            spec.reasoning.teacher
            and spec.reasoning.teacher_model
            and spec.reasoning.prompt_version
        ):
            raise ValueError("reasoning teacher, teacher_model, and prompt_version are required")
        if spec.reasoning.label_stride <= 0:
            raise ValueError("reasoning label_stride must be positive")


def canonical_dataset_spec(spec: DatasetSpec) -> dict[str, Any]:
    """Return the JSON object whose digest is the dataset identity."""
    validate_dataset_spec(spec)
    payload = asdict(spec)
    payload["dataset"] = spec.dataset.value
    payload["pack_params"] = dict(sorted(spec.pack_params.items()))
    payload["schema_version"] = DATASET_SPEC_SCHEMA
    return payload


def compute_dataset_id(spec: DatasetSpec) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_dataset_spec(spec)))


def dataset_index_key(published_name: str, dataset_id: str) -> str:
    if not _SHA256_RE.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id {dataset_id!r}")
    dataset_prefix(published_name, "v1")
    return f"{published_name}/by-id/{dataset_id}.json"


def build_reservation_key(published_name: str, dataset_id: str) -> str:
    if not _SHA256_RE.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id {dataset_id!r}")
    dataset_prefix(published_name, "v1")
    return f"{published_name}/_stages/builds/{dataset_id}.json"


def version_reservation_key(published_name: str, version: str) -> str:
    dataset_prefix(published_name, version)
    return f"{published_name}/_stages/versions/{version}.json"


def lineage_key(published_name: str, version: str) -> str:
    return f"{dataset_prefix(published_name, version)}/lineage/dataset.json"


def units_key(published_name: str, version: str) -> str:
    return f"{dataset_prefix(published_name, version)}/lineage/units.jsonl"


def _version_parts(version: str) -> tuple[int, int]:
    match = _VERSION_PARTS_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"invalid dataset version {version!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def next_dataset_version(existing_versions: Iterable[str]) -> str:
    """Allocate the next minor version after the newest well-formed one."""
    parsed = []
    for version in existing_versions:
        try:
            parsed.append(_version_parts(version))
        except ValueError:
            continue
    if not parsed:
        return "v1.0"
    major, minor = max(parsed)
    return f"v{major}.{minor + 1}"


def dataset_index_record(
    spec: DatasetSpec,
    *,
    published_name: str,
    version: str,
    manifest_key: str,
    manifest_sha256: str,
    lineage_sha256: str,
) -> dict[str, Any]:
    canonical = canonical_dataset_spec(spec)
    expected_key = f"{dataset_prefix(published_name, version)}/shards/manifest.json"
    if manifest_key != expected_key:
        raise ValueError(f"manifest key {manifest_key!r} is not {expected_key!r}")
    for name, digest in (("manifest_sha256", manifest_sha256), ("lineage_sha256", lineage_sha256)):
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{name} must be 64 lowercase hex digits")
    return {
        "schema_version": DATASET_INDEX_SCHEMA,
        "dataset_id": sha256_bytes(canonical_json_bytes(canonical)),
        "spec": canonical,
        "published_name": published_name,
        "version": version,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "lineage_key": lineage_key(published_name, version),
        "lineage_sha256": lineage_sha256,
    }


def validate_dataset_index_record(
    record: Mapping[str, Any],
    spec: DatasetSpec,
    *,
    published_name: str,
) -> None:
    """Reject an index record that does not describe exactly ``spec``."""
    if record.get("schema_version") != DATASET_INDEX_SCHEMA:
        raise ValueError("unsupported dataset index schema")
    canonical = canonical_dataset_spec(spec)
    if record.get("spec") != canonical:
        raise ValueError("dataset index spec differs from the requested spec")
    if record.get("published_name") != published_name:
        raise ValueError("dataset index is for another published dataset")
    expected = dataset_index_record(
        spec,
        published_name=published_name,
        version=str(record.get("version")),
        manifest_key=str(record.get("manifest_key")),
        manifest_sha256=str(record.get("manifest_sha256")),
        lineage_sha256=str(record.get("lineage_sha256")),
    )
    if dict(record) != expected:
        raise ValueError("dataset index record has unexpected fields")


def validate_publication(
    record: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    manifest_payload: bytes,
    expected_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the lock and manifest bytes against an index record.

    ``expected_contracts`` are the pack contracts of the running code, so a
    publication packed under older contracts is never reused. Returns the
    parsed manifest. Shard existence is checked by the caller.
    """
    if lock.get("schema_version") != PUBLICATION_SCHEMA:
        raise ValueError("unsupported publication lock schema")
    actual_sha256 = sha256_bytes(manifest_payload)
    if lock.get("manifest_sha256") != actual_sha256:
        raise ValueError("publication lock does not match the manifest bytes")
    if record["manifest_sha256"] != actual_sha256:
        raise ValueError("dataset index does not match the manifest bytes")
    manifest = json.loads(manifest_payload)
    if manifest.get("status") != "ready":
        raise ValueError("published manifest is not ready")
    if manifest.get("dataset") != record["published_name"]:
        raise ValueError("published manifest dataset differs from the index")
    if manifest.get("version") != record["version"]:
        raise ValueError("published manifest version differs from the index")
    if manifest.get("contracts") != dict(expected_contracts):
        raise ValueError("published manifest was packed under other contracts")
    if int(manifest.get("image_size", -1)) != int(record["spec"]["image_size"]):
        raise ValueError("published manifest image size differs from the spec")
    entries = manifest.get("shard_entries") or []
    if not entries or len(entries) != manifest.get("shard_count"):
        raise ValueError("published manifest shard entries are incomplete")
    for entry in entries:
        if not entry.get("etag"):
            raise ValueError(f"shard entry {entry.get('name')!r} has no ETag")
    return manifest
