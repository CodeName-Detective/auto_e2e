"""Flyte tasks behind wf_autoe2e."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from flytekit import Resources, task

from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    compute_dataset_id,
    dataset_index_key,
    dataset_index_record,
    next_dataset_version,
    validate_dataset_index_record,
    validate_publication,
)
from Platform.pipelines.autoe2e_stages import (
    SceneStage,
    stage_id,
    stage_record,
    stage_record_key,
    validate_stage_record,
)
from Platform.pipelines.dataset_publication import (
    DatasetPublication,
    canonical_json_bytes,
)
from Platform.pipelines.dataset_publication_tasks import (
    DATA_PREP_IMAGE,
    _get_bytes,
    _is_not_found,
    _list_objects,
    _put_immutable,
    _split_s3_uri,
)


_MAX_METADATA_BYTES = 64 * 1024**2


@dataclass
class DatasetHandle:
    """The resolved dataset for one execution.

    ``published`` is False when no valid publication exists; ``version`` is
    then the version the build stage must publish under.
    """

    dataset_id: str
    version: str
    published: bool
    manifest_key: str = ""
    manifest_sha256: str = ""
    shard_uris: List[str] = field(default_factory=list)


def _get_optional(s3, bucket: str, key: str) -> bytes | None:
    from botocore.exceptions import ClientError

    try:
        return _get_bytes(s3, bucket, key, max_bytes=_MAX_METADATA_BYTES)
    except ClientError as exc:
        if _is_not_found(exc):
            return None
        raise


def _version_dirs(s3, bucket: str, dataset: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{dataset}/"
    versions = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            versions.append(common["Prefix"][len(prefix):].rstrip("/"))
    return versions


def _check_shards(s3, bucket: str, manifest: dict) -> list[str]:
    uris = []
    for entry in manifest["shard_entries"]:
        key = str(entry["key"])
        head = s3.head_object(Bucket=bucket, Key=key)
        if int(head["ContentLength"]) != int(entry["byte_size"]):
            raise ValueError(f"published shard size differs: s3://{bucket}/{key}")
        etag = str(head.get("ETag", "")).strip('"')
        if entry.get("etag") and etag != entry["etag"]:
            raise ValueError(f"published shard ETag differs: s3://{bucket}/{key}")
        uris.append(f"s3://{bucket}/{key}")
    return uris


def resolve_published_dataset(s3, *, bucket: str, spec: DatasetSpec) -> DatasetHandle:
    """Return the valid publication for ``spec``, or the version to build."""
    dataset_id = compute_dataset_id(spec)
    index_payload = _get_optional(
        s3, bucket, dataset_index_key(spec.dataset, dataset_id)
    )
    if index_payload is None:
        return DatasetHandle(
            dataset_id=dataset_id,
            version=next_dataset_version(
                _version_dirs(s3, bucket, spec.dataset.value)
            ),
            published=False,
        )

    record = json.loads(index_payload)
    validate_dataset_index_record(record, spec)
    manifest_key = record["manifest_key"]
    lock_key = manifest_key.rsplit("/", 1)[0] + "/.publication-lock.json"
    lock = json.loads(_get_bytes(s3, bucket, lock_key, max_bytes=_MAX_METADATA_BYTES))
    manifest_payload = _get_bytes(
        s3, bucket, manifest_key, max_bytes=_MAX_METADATA_BYTES
    )
    manifest = validate_publication(
        record, lock=lock, manifest_payload=manifest_payload
    )
    return DatasetHandle(
        dataset_id=dataset_id,
        version=record["version"],
        published=True,
        manifest_key=manifest_key,
        manifest_sha256=record["manifest_sha256"],
        shard_uris=_check_shards(s3, bucket, manifest),
    )


def write_dataset_index(
    s3,
    *,
    bucket: str,
    spec: DatasetSpec,
    version: str,
    publication: DatasetPublication,
) -> str:
    """Write the by-id pointer once the manifest gate exists."""
    record = dataset_index_record(
        spec,
        version=version,
        manifest_key=publication.manifest_key,
        manifest_sha256=publication.manifest_sha256,
    )
    key = dataset_index_key(spec.dataset, record["dataset_id"])
    _put_immutable(
        s3,
        bucket=bucket,
        key=key,
        payload=canonical_json_bytes(record, pretty=True),
        content_type="application/json",
    )
    return key


def _artifact_objects(s3, artifact_uri: str) -> list[dict]:
    bucket, prefix = _split_s3_uri(artifact_uri)
    return _list_objects(s3, bucket, prefix)


def lookup_scene_stage(
    s3,
    *,
    bucket: str,
    dataset: AutoE2EDataset,
    stage: SceneStage,
    identity: dict,
) -> str | None:
    """Return the recorded artifact URI for a scene stage, or None.

    A record whose artifact listing changed raises instead of returning None,
    so a damaged artifact is never silently replaced by a new download or a
    new teacher call.
    """
    payload = _get_optional(
        s3, bucket, stage_record_key(dataset, stage, stage_id(identity))
    )
    if payload is None:
        return None
    record = json.loads(payload)
    validate_stage_record(
        record,
        identity,
        objects=_artifact_objects(s3, record["artifact_uri"]),
    )
    return record["artifact_uri"]


def record_scene_stage(
    s3,
    *,
    bucket: str,
    dataset: AutoE2EDataset,
    stage: SceneStage,
    identity: dict,
    artifact_uri: str,
) -> str:
    """Write the write-once record for an artifact a build just produced."""
    record = stage_record(
        identity,
        artifact_uri=artifact_uri.rstrip("/"),
        objects=_artifact_objects(s3, artifact_uri),
    )
    key = stage_record_key(dataset, stage, record["stage_id"])
    _put_immutable(
        s3,
        bucket=bucket,
        key=key,
        payload=canonical_json_bytes(record, pretty=True),
        content_type="application/json",
    )
    return key


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
    retries=2,
)
def resolve_dataset(
    spec: DatasetSpec,
    datasets_bucket: str,
    aws_region: str = "us-west-2",
) -> DatasetHandle:
    import boto3

    if not datasets_bucket:
        raise ValueError("datasets_bucket is required")
    s3 = boto3.client("s3", region_name=aws_region)
    return resolve_published_dataset(s3, bucket=datasets_bucket, spec=spec)


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
    retries=2,
)
def record_dataset_index(
    spec: DatasetSpec,
    version: str,
    manifest_key: str,
    manifest_sha256: str,
    datasets_bucket: str,
    aws_region: str = "us-west-2",
) -> DatasetHandle:
    """Index a finished publication, then re-resolve it through validation."""
    import boto3

    s3 = boto3.client("s3", region_name=aws_region)
    write_dataset_index(
        s3,
        bucket=datasets_bucket,
        spec=spec,
        version=version,
        publication=DatasetPublication(manifest_key, manifest_sha256),
    )
    return resolve_published_dataset(s3, bucket=datasets_bucket, spec=spec)
