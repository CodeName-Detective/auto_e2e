"""wf_autoe2e dataset resolution against an in-memory S3."""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError

from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    ReasoningSpec,
    compute_dataset_id,
)
from Platform.pipelines.autoe2e_stages import (
    LabelIdentity,
    SceneStage,
    label_identity,
    raw_identity,
)
from Platform.pipelines.autoe2e_tasks import (
    DatasetHandle,
    lookup_scene_stage,
    record_scene_stage,
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

    def paginate(self, *, Bucket, Prefix, Delimiter=None):
        keys = sorted(
            key for bucket, key in self.s3.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        if Delimiter is None:
            yield {"Contents": [
                {
                    "Key": key,
                    "Size": len(self.s3.objects[(Bucket, key)]),
                    "ETag": f'"{sha256_bytes(self.s3.objects[(Bucket, key)])[:32]}"',
                }
                for key in keys
            ]}
            return
        prefixes = sorted({
            Prefix + key[len(Prefix):].split(Delimiter, 1)[0] + Delimiter
            for key in keys
            if Delimiter in key[len(Prefix):]
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


@pytest.mark.parametrize(
    ("value", "python_type"),
    [
        (_spec(), DatasetSpec),
        (
            _spec(
                dataset=AutoE2EDataset.NUPLAN,
                reasoning=ReasoningSpec(teacher="t", prompt_version="p", label_stride=10),
            ),
            DatasetSpec,
        ),
        (
            DatasetHandle(
                dataset_id="a" * 64,
                version="v3.4",
                published=True,
                manifest_key="kitscenes/v3.4/shards/manifest.json",
                manifest_sha256="b" * 64,
                shard_uris=["s3://datasets/kitscenes/v3.4/shards/a.tar"],
            ),
            DatasetHandle,
        ),
    ],
)
def test_task_interface_types_round_trip_through_flyte(value, python_type):
    from flytekit.core.context_manager import FlyteContextManager
    from flytekit.core.type_engine import TypeEngine

    ctx = FlyteContextManager.current_context()
    literal_type = TypeEngine.to_literal_type(python_type)
    literal = TypeEngine.to_literal(ctx, value, python_type, literal_type)
    assert TypeEngine.to_python_value(ctx, literal, python_type) == value


def _raw_artifact(s3: _S3, prefix: str = "sv/run/abc") -> str:
    s3.put(f"{prefix}/data/train/scene-a/x.mcap", b"frames")
    s3.put(f"{prefix}/data/sequence_archives.csv", b"csv")
    return f"s3://{BUCKET}/{prefix}"


def test_unrecorded_scene_stage_is_absent():
    identity = raw_identity(AutoE2EDataset.KITSCENES, "rev", "scene-a")
    assert lookup_scene_stage(
        _S3(),
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.RAW,
        identity=identity,
    ) is None


def test_recorded_scene_stage_is_reused():
    s3 = _S3()
    uri = _raw_artifact(s3)
    identity = raw_identity(AutoE2EDataset.KITSCENES, "rev", "scene-a")
    key = record_scene_stage(
        s3,
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.RAW,
        identity=identity,
        artifact_uri=uri + "/",
    )
    assert key.startswith("kitscenes/_stages/raw/")
    assert lookup_scene_stage(
        s3,
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.RAW,
        identity=identity,
    ) == uri


def test_changed_scene_artifact_fails_instead_of_rebuilding():
    s3 = _S3()
    uri = _raw_artifact(s3)
    identity = raw_identity(AutoE2EDataset.KITSCENES, "rev", "scene-a")
    record_scene_stage(
        s3,
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.RAW,
        identity=identity,
        artifact_uri=uri,
    )
    del s3.objects[(BUCKET, "sv/run/abc/data/sequence_archives.csv")]
    with pytest.raises(ValueError, match="changed"):
        lookup_scene_stage(
            s3,
            bucket=BUCKET,
            dataset=AutoE2EDataset.KITSCENES,
            stage=SceneStage.RAW,
            identity=identity,
        )


def test_label_record_is_separate_from_raw_record():
    s3 = _S3()
    uri = _raw_artifact(s3)
    labels = LabelIdentity("openai_compatible", "p", 10, "v2", "v1")
    record_scene_stage(
        s3,
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.RAW,
        identity=raw_identity(AutoE2EDataset.KITSCENES, "rev", "scene-a"),
        artifact_uri=uri,
    )
    assert lookup_scene_stage(
        s3,
        bucket=BUCKET,
        dataset=AutoE2EDataset.KITSCENES,
        stage=SceneStage.LABELS,
        identity=label_identity(AutoE2EDataset.KITSCENES, "rev", "scene-a", labels),
    ) is None
