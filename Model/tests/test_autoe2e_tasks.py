"""wf_autoe2e dataset resolution against an in-memory S3."""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError

from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    compute_dataset_id,
)
from Platform.pipelines.autoe2e_tasks import (
    resolve_published_dataset,
    write_dataset_index,
)
from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    DatasetPublication,
    canonical_json_bytes,
    sha256_bytes,
)

BUCKET = "datasets"


def _error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Op",
    )


class _Paginator:
    def __init__(self, s3):
        self.s3 = s3

    def paginate(self, *, Bucket, Prefix, Delimiter):
        prefixes = sorted({
            Prefix + key[len(Prefix):].split(Delimiter, 1)[0] + Delimiter
            for bucket, key in self.s3.objects
            if bucket == Bucket
            and key.startswith(Prefix)
            and Delimiter in key[len(Prefix):]
        })
        yield {"CommonPrefixes": [{"Prefix": prefix} for prefix in prefixes]}


class _S3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict] = {}

    def put(self, key: str, payload: bytes) -> None:
        self.objects[(BUCKET, key)] = payload
        self.metadata[(BUCKET, key)] = {"sha256": sha256_bytes(payload)}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)

    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _error("NoSuchKey", 404)
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _error("404", 404)
        payload = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(payload),
            "ETag": f'"{sha256_bytes(payload)[:32]}"',
            "Metadata": self.metadata[(Bucket, Key)],
        }

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, Metadata, **_):
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            raise _error("PreconditionFailed", 412)
        self.objects[(Bucket, Key)] = Body
        self.metadata[(Bucket, Key)] = Metadata


def _spec(**overrides) -> DatasetSpec:
    values = dict(
        dataset=AutoE2EDataset.KITSCENES,
        source_revision="rev",
        split_manifest=ManifestRef(
            uri="s3://checkpoints/recovery-manifests/a.json",
            version_id="v1",
            sha256="a" * 64,
        ),
        pack_version="v3.3",
        image_size=256,
    )
    values.update(overrides)
    return DatasetSpec(**values)


def _publish(s3: _S3, version: str = "v3.4") -> DatasetPublication:
    shard_key = f"kitscenes/{version}/shards/train-000000.tar"
    shard = b"tar-bytes"
    s3.put(shard_key, shard)
    manifest = {
        "status": "ready",
        "dataset": "kitscenes",
        "version": version,
        "shard_count": 1,
        "shard_entries": [{
            "name": "train-000000.tar",
            "key": shard_key,
            "byte_size": len(shard),
            "etag": sha256_bytes(shard)[:32],
        }],
    }
    payload = canonical_json_bytes(manifest, pretty=True)
    digest = sha256_bytes(payload)
    s3.put(
        f"kitscenes/{version}/shards/.publication-lock.json",
        canonical_json_bytes(
            {"schema_version": PUBLICATION_SCHEMA, "manifest_sha256": digest},
            pretty=True,
        ),
    )
    manifest_key = f"kitscenes/{version}/shards/manifest.json"
    s3.put(manifest_key, payload)
    return DatasetPublication(manifest_key, digest)


def test_unpublished_spec_allocates_the_next_version():
    s3 = _S3()
    s3.put("kitscenes/v3.3/shards/manifest.json", b"{}")
    s3.put("kitscenes/v2.2/shards/manifest.json", b"{}")

    handle = resolve_published_dataset(s3, bucket=BUCKET, spec=_spec())

    assert handle.published is False
    assert handle.version == "v3.4"
    assert handle.dataset_id == compute_dataset_id(_spec())
    assert handle.shard_uris == []


def test_indexed_publication_is_reused():
    s3 = _S3()
    spec = _spec()
    publication = _publish(s3)
    write_dataset_index(
        s3, bucket=BUCKET, spec=spec, version="v3.4", publication=publication
    )

    handle = resolve_published_dataset(s3, bucket=BUCKET, spec=spec)

    assert handle.published is True
    assert handle.version == "v3.4"
    assert handle.manifest_sha256 == publication.manifest_sha256
    assert handle.shard_uris == [
        f"s3://{BUCKET}/kitscenes/v3.4/shards/train-000000.tar"
    ]


def test_a_different_spec_does_not_reuse_the_publication():
    s3 = _S3()
    write_dataset_index(
        s3, bucket=BUCKET, spec=_spec(), version="v3.4", publication=_publish(s3)
    )

    handle = resolve_published_dataset(
        s3, bucket=BUCKET, spec=_spec(image_size=512)
    )

    assert handle.published is False
    assert handle.version == "v3.5"


def test_missing_shard_fails_resolution():
    s3 = _S3()
    spec = _spec()
    write_dataset_index(
        s3, bucket=BUCKET, spec=spec, version="v3.4", publication=_publish(s3)
    )
    del s3.objects[(BUCKET, "kitscenes/v3.4/shards/train-000000.tar")]

    with pytest.raises(ClientError):
        resolve_published_dataset(s3, bucket=BUCKET, spec=spec)


def test_truncated_shard_fails_resolution():
    s3 = _S3()
    spec = _spec()
    write_dataset_index(
        s3, bucket=BUCKET, spec=spec, version="v3.4", publication=_publish(s3)
    )
    s3.objects[(BUCKET, "kitscenes/v3.4/shards/train-000000.tar")] = b"tar"

    with pytest.raises(ValueError, match="size differs"):
        resolve_published_dataset(s3, bucket=BUCKET, spec=spec)


def test_index_is_write_once():
    s3 = _S3()
    spec = _spec()
    publication = _publish(s3)
    write_dataset_index(
        s3, bucket=BUCKET, spec=spec, version="v3.4", publication=publication
    )
    write_dataset_index(
        s3, bucket=BUCKET, spec=spec, version="v3.4", publication=publication
    )
    other = _publish(s3, version="v3.5")

    with pytest.raises(RuntimeError, match="different content"):
        write_dataset_index(
            s3, bucket=BUCKET, spec=spec, version="v3.5", publication=other
        )
