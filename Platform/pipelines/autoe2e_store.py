"""S3 state behind wf_autoe2e: dataset reuse, version reservation, labels.

Everything here is write-once (``IfNoneMatch="*"``) and verified on read, so
a rerun either reproduces identical bytes or fails loudly; nothing is
silently rebuilt, re-labelled, or reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    build_reservation_key,
    compute_dataset_id,
    dataset_index_key,
    dataset_index_record,
    lineage_key,
    next_dataset_version,
    validate_dataset_index_record,
    validate_views,
    version_reservation_key,
    views_manifest_key,
)
from Platform.pipelines.autoe2e_labels import (
    label_attempt_prefix,
    label_claim,
    label_claim_key,
    label_id,
    label_record,
    label_record_key,
    validate_label_record,
)
from Platform.pipelines.autoe2e_lineage import (
    DATASET_LINEAGE_SCHEMA,
    parse_record,
    record_bytes,
    record_sha256,
)
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes
from Platform.pipelines.dataset_publication_tasks import (
    _get_bytes,
    _is_not_found,
    _is_precondition_failed,
    _list_objects,
    _put_immutable,
    _split_s3_uri,
)

_MAX_METADATA_BYTES = 64 * 1024**2
_MAX_VERSION_ATTEMPTS = 64


@dataclass
class DatasetHandle:
    """The resolved dataset for one execution.

    ``published`` is False when the build stage must run; ``version`` is
    then the version reserved for this ``dataset_id``. ``partition_uris`` are
    the per-partition directories Ray Train reads.
    """

    dataset_id: str
    published_name: str
    version: str
    published: bool
    # The execution that reserved the version; stable across reruns, so the
    # write-once lineage names the same build every time.
    reserved_by: str = ""
    views_sha256: str = ""
    lineage_key: str = ""
    lineage_sha256: str = ""
    samples: int = 0
    partition_uris: List[str] = field(default_factory=list)


def get_optional(s3, bucket: str, key: str) -> Optional[bytes]:
    from botocore.exceptions import ClientError

    try:
        return _get_bytes(s3, bucket, key, max_bytes=_MAX_METADATA_BYTES)
    except ClientError as exc:
        if _is_not_found(exc):
            return None
        raise


def put_if_absent(s3, bucket: str, key: str, payload: bytes) -> bool:
    """Write once; return False if another writer already created the key."""
    from botocore.exceptions import ClientError

    try:
        s3.put_object(
            Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*",
            ContentType="application/json",
        )
        return True
    except ClientError as exc:
        if _is_precondition_failed(exc):
            return False
        raise


def _version_dirs(s3, bucket: str, published_name: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{published_name}/"
    versions = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            versions.append(common["Prefix"][len(prefix):].rstrip("/"))
    reserved_prefix = f"{published_name}/_stages/versions/"
    for page in paginator.paginate(Bucket=bucket, Prefix=reserved_prefix):
        for item in page.get("Contents", []):
            name = item["Key"][len(reserved_prefix):]
            if name.endswith(".json"):
                versions.append(name[: -len(".json")])
    return versions


def reserve_version(s3, *, bucket: str, published_name: str, dataset_id: str, execution_id: str) -> tuple[str, str]:
    """Return the one version this dataset_id builds under, reserving it once.

    A rerun of the same spec gets the same version back, so an aborted build
    finishes into its version instead of allocating a new one. Two different
    specs racing for a number each get their own, because version
    reservations are write-once too.
    """
    build_key = build_reservation_key(published_name, dataset_id)
    existing = get_optional(s3, bucket, build_key)
    if existing is not None:
        record = json.loads(existing)
        return str(record["version"]), str(record["execution_id"])
    for _ in range(_MAX_VERSION_ATTEMPTS):
        version = next_dataset_version(_version_dirs(s3, bucket, published_name))
        claim = canonical_json_bytes({"dataset_id": dataset_id, "execution_id": execution_id})
        if not put_if_absent(s3, bucket, version_reservation_key(published_name, version), claim):
            continue
        reservation = canonical_json_bytes({"version": version, "execution_id": execution_id})
        if put_if_absent(s3, bucket, build_key, reservation):
            return version, execution_id
        winner = json.loads(get_optional(s3, bucket, build_key) or b"{}")
        return str(winner["version"]), str(winner["execution_id"])
    raise RuntimeError(f"could not reserve a version for {published_name} after {_MAX_VERSION_ATTEMPTS} tries")


def read_dataset_lineage(s3, *, bucket: str, key: str, sha256: str) -> dict[str, Any]:
    payload = _get_bytes(s3, bucket, key, max_bytes=_MAX_METADATA_BYTES)
    return parse_record(payload, expected_sha256=sha256, schema=DATASET_LINEAGE_SCHEMA)


def resolve_published_dataset(
    s3,
    *,
    bucket: str,
    spec: DatasetSpec,
    published_name: str,
    expected_contracts: Mapping[str, Any],
    execution_id: str,
) -> DatasetHandle:
    """Return the verified publication for ``spec``, or reserve its version."""
    dataset_id = compute_dataset_id(spec)
    index_payload = get_optional(s3, bucket, dataset_index_key(published_name, dataset_id))
    if index_payload is None:
        version, reserved_by = reserve_version(
            s3, bucket=bucket, published_name=published_name,
            dataset_id=dataset_id, execution_id=execution_id,
        )
        return DatasetHandle(
            dataset_id=dataset_id,
            published_name=published_name,
            version=version,
            published=False,
            reserved_by=reserved_by,
        )
    record = json.loads(index_payload)
    validate_dataset_index_record(record, spec, published_name=published_name)
    views = validate_views(
        record,
        views_payload=_get_bytes(s3, bucket, record["views_key"], max_bytes=_MAX_METADATA_BYTES),
        expected_contracts=expected_contracts,
    )
    lineage = read_dataset_lineage(
        s3, bucket=bucket, key=record["lineage_key"], sha256=record["lineage_sha256"],
    )
    if lineage["dataset_id"] != dataset_id or lineage["views"]["sha256"] != record["views_sha256"]:
        raise ValueError("dataset lineage does not describe the indexed publication")
    return DatasetHandle(
        dataset_id=dataset_id,
        published_name=published_name,
        version=record["version"],
        published=True,
        reserved_by=str(lineage["build"].get("flyte_execution_id", "")),
        views_sha256=record["views_sha256"],
        lineage_key=record["lineage_key"],
        lineage_sha256=record["lineage_sha256"],
        samples=int(views["samples"]),
        partition_uris=verify_partition_views(s3, bucket=bucket, views=views),
    )


def verify_partition_views(s3, *, bucket: str, views: Mapping[str, Any]) -> list[str]:
    """Check each training partition manifest and its shards against the views."""
    uris = []
    for partition in views["partitions"]:
        prefix = partition["prefix"]
        payload = _get_bytes(s3, bucket, f"{prefix}/manifest.json", max_bytes=_MAX_METADATA_BYTES)
        if sha256_bytes(payload) != partition["manifest_sha256"]:
            raise ValueError(f"partition manifest changed: s3://{bucket}/{prefix}")
        manifest = json.loads(payload)
        for name in manifest.get("shard_names") or []:
            s3.head_object(Bucket=bucket, Key=f"{prefix}/{name}")
        uris.append(f"s3://{bucket}/{prefix}")
    return uris


def write_views_manifest(s3, *, bucket: str, views: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(dict(views))
    _put_immutable(
        s3, bucket=bucket, key=views_manifest_key(views["published_name"], views["version"]),
        payload=payload, content_type="application/json",
    )
    return sha256_bytes(payload)


def write_dataset_lineage(
    s3,
    *,
    bucket: str,
    published_name: str,
    version: str,
    record: Mapping[str, Any],
    units_payload: bytes,
    units_key_name: str,
) -> tuple[str, str]:
    """Write units then dataset lineage; return ``(key, sha256)``."""
    _put_immutable(s3, bucket=bucket, key=units_key_name, payload=units_payload, content_type="application/x-ndjson")
    key = lineage_key(published_name, version)
    _put_immutable(s3, bucket=bucket, key=key, payload=record_bytes(record), content_type="application/json")
    return key, record_sha256(record)


def write_dataset_index(
    s3,
    *,
    bucket: str,
    spec: DatasetSpec,
    published_name: str,
    version: str,
    views_sha256: str,
    lineage_sha256: str,
) -> str:
    """Write the by-id pointer last: it is the commit of a dataset build."""
    record = dataset_index_record(
        spec,
        published_name=published_name,
        version=version,
        views_sha256=views_sha256,
        lineage_sha256=lineage_sha256,
    )
    key = dataset_index_key(published_name, record["dataset_id"])
    _put_immutable(
        s3, bucket=bucket, key=key,
        payload=canonical_json_bytes(record, pretty=True), content_type="application/json",
    )
    return key


def artifact_objects(s3, artifact_uri: str) -> list[dict]:
    bucket, prefix = _split_s3_uri(artifact_uri)
    return _list_objects(s3, bucket, prefix)


def lookup_label_set(s3, *, bucket: str, dataset: AutoE2EDataset, identity: Mapping[str, Any]) -> Optional[dict]:
    """Return the verified committed label record for one unit, or None.

    A record whose files changed raises, so damaged labels are never
    silently replaced by new teacher calls.
    """
    payload = get_optional(s3, bucket, label_record_key(dataset, label_id(identity)))
    if payload is None:
        return None
    record = json.loads(payload)
    validate_label_record(
        record, identity, bucket=bucket,
        objects=artifact_objects(s3, record["artifact_uri"]),
    )
    return record


def claim_label_set(
    s3,
    *,
    bucket: str,
    dataset: AutoE2EDataset,
    identity: Mapping[str, Any],
    execution_id: str,
    claimed_at: str,
) -> None:
    """Become the only execution allowed to call the teacher for this unit.

    A retry of the same execution may proceed; any other execution fails,
    so two runs never pay for the same labels.
    """
    digest = label_id(identity)
    key = label_claim_key(dataset, digest)
    claim = canonical_json_bytes(label_claim(identity, execution_id=execution_id, claimed_at=claimed_at))
    if put_if_absent(s3, bucket, key, claim):
        return
    holder = json.loads(get_optional(s3, bucket, key) or b"{}")
    if holder.get("execution_id") != execution_id:
        raise RuntimeError(
            f"label set {digest} is being produced by execution {holder.get('execution_id')}; "
            "rerun after it commits, or delete its claim if that execution is dead"
        )


def commit_label_set(
    s3,
    *,
    bucket: str,
    dataset: AutoE2EDataset,
    identity: Mapping[str, Any],
    attempt_id: str,
    label_count: int,
) -> dict:
    """Commit one attempt's files as the unit's labels (the write-once record)."""
    digest = label_id(identity)
    record = label_record(
        identity,
        bucket=bucket,
        attempt_id=attempt_id,
        objects=artifact_objects(s3, f"s3://{bucket}/{label_attempt_prefix(dataset, digest, attempt_id)}"),
        label_count=label_count,
    )
    _put_immutable(
        s3, bucket=bucket, key=label_record_key(dataset, digest),
        payload=canonical_json_bytes(record, pretty=True), content_type="application/json",
    )
    return record
