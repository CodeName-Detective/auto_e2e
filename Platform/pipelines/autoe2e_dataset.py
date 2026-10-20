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
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

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


@dataclass(frozen=True)
class ManifestRef:
    """An immutable S3 object pinned by version id and content digest."""

    uri: str
    version_id: str
    sha256: str


@dataclass(frozen=True)
class ReasoningSpec:
    teacher: str
    prompt_version: str
    label_stride: int


@dataclass(frozen=True)
class DatasetSpec:
    dataset: AutoE2EDataset
    source_revision: str
    split_manifest: ManifestRef
    pack_version: str
    image_size: int
    reasoning: Optional[ReasoningSpec] = None


def validate_dataset_spec(spec: DatasetSpec) -> None:
    if not isinstance(spec.dataset, AutoE2EDataset):
        raise ValueError(f"unsupported dataset {spec.dataset!r}")
    if not spec.source_revision:
        raise ValueError("source_revision is required")
    ref = spec.split_manifest
    if not _S3_URI_RE.fullmatch(ref.uri):
        raise ValueError(f"split_manifest uri is not an S3 object: {ref.uri!r}")
    if not ref.version_id:
        raise ValueError("split_manifest version_id is required")
    if not _SHA256_RE.fullmatch(ref.sha256):
        raise ValueError("split_manifest sha256 must be 64 lowercase hex digits")
    if not spec.pack_version:
        raise ValueError("pack_version is required")
    if spec.image_size <= 0:
        raise ValueError("image_size must be positive")
    if spec.reasoning is not None:
        if not spec.reasoning.teacher or not spec.reasoning.prompt_version:
            raise ValueError("reasoning teacher and prompt_version are required")
        if spec.reasoning.label_stride <= 0:
            raise ValueError("reasoning label_stride must be positive")


def canonical_dataset_spec(spec: DatasetSpec) -> dict[str, Any]:
    """Return the JSON object whose digest is the dataset identity."""
    validate_dataset_spec(spec)
    payload = asdict(spec)
    payload["dataset"] = spec.dataset.value
    payload["schema_version"] = DATASET_SPEC_SCHEMA
    return payload


def compute_dataset_id(spec: DatasetSpec) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_dataset_spec(spec)))


def dataset_index_key(dataset: AutoE2EDataset, dataset_id: str) -> str:
    if not _SHA256_RE.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset id {dataset_id!r}")
    return f"{dataset.value}/by-id/{dataset_id}.json"


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
    version: str,
    manifest_key: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    canonical = canonical_dataset_spec(spec)
    expected_key = (
        f"{dataset_prefix(spec.dataset.value, version)}/shards/manifest.json"
    )
    if manifest_key != expected_key:
        raise ValueError(
            f"manifest key {manifest_key!r} is not under {expected_key!r}"
        )
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be 64 lowercase hex digits")
    return {
        "schema_version": DATASET_INDEX_SCHEMA,
        "dataset_id": sha256_bytes(canonical_json_bytes(canonical)),
        "spec": canonical,
        "version": version,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
    }


def validate_dataset_index_record(
    record: Mapping[str, Any],
    spec: DatasetSpec,
) -> None:
    """Reject an index record that does not describe exactly ``spec``."""
    if record.get("schema_version") != DATASET_INDEX_SCHEMA:
        raise ValueError("unsupported dataset index schema")
    canonical = canonical_dataset_spec(spec)
    if record.get("spec") != canonical:
        raise ValueError("dataset index spec differs from the requested spec")
    if record.get("dataset_id") != sha256_bytes(canonical_json_bytes(canonical)):
        raise ValueError("dataset index id does not match its spec")
    expected = dataset_index_record(
        spec,
        version=str(record.get("version")),
        manifest_key=str(record.get("manifest_key")),
        manifest_sha256=str(record.get("manifest_sha256")),
    )
    if dict(record) != expected:
        raise ValueError("dataset index record has unexpected fields")


def validate_publication(
    record: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    manifest_payload: bytes,
) -> dict[str, Any]:
    """Check the lock and manifest bytes against an index record.

    Returns the parsed manifest. Shard object existence is checked by the
    caller, which owns the S3 client.
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
    if manifest.get("dataset") != record["spec"]["dataset"]:
        raise ValueError("published manifest dataset differs from the index")
    if manifest.get("version") != record["version"]:
        raise ValueError("published manifest version differs from the index")
    entries = manifest.get("shard_entries") or []
    if not entries or len(entries) != manifest.get("shard_count"):
        raise ValueError("published manifest shard entries are incomplete")
    return manifest
