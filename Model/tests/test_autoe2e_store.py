"""Write-once S3 store for wf_autoe2e, against an in-memory S3."""

from __future__ import annotations

import io

import pytest
from botocore.exceptions import ClientError

from Platform.pipelines.autoe2e_adapters import current_pack_version, pack_contracts
from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    ReasoningSpec,
    canonical_dataset_spec,
    compute_dataset_id,
    partition_view_prefix,
    units_key,
    views_manifest,
    views_manifest_key,
)
from Platform.pipelines.autoe2e_labels import (
    LabelIdentity,
    label_attempt_prefix,
    label_id,
    label_identity,
)
from Platform.pipelines.autoe2e_lineage import UnitLineage, dataset_lineage
from Platform.pipelines.autoe2e_store import (
    DatasetHandle,
    claim_label_set,
    commit_label_set,
    lookup_label_set,
    reserve_version,
    resolve_published_dataset,
    write_dataset_index,
    write_dataset_lineage,
    write_views_manifest,
)
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes

BUCKET = "datasets"
KIT = AutoE2EDataset.KITSCENES


def _error(code: str, status: int) -> ClientError:
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "Op")


class _Paginator:
    def __init__(self, s3):
        self.s3 = s3

    def paginate(self, *, Bucket, Prefix, Delimiter=None):
        keys = sorted(k for b, k in self.s3.objects if b == Bucket and k.startswith(Prefix))
        if Delimiter is None:
            yield {"Contents": [
                {"Key": k, "Size": len(self.s3.objects[(Bucket, k)]),
                 "ETag": f'"{sha256_bytes(self.s3.objects[(Bucket, k)])[:32]}"'}
                for k in keys
            ]}
            return
        prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter, 1)[0] + Delimiter
                           for k in keys if Delimiter in k[len(Prefix):]})
        yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}


class S3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict] = {}

    def put(self, key: str, payload: bytes) -> None:
        self.objects[(BUCKET, key)] = payload
        self.metadata[(BUCKET, key)] = {"sha256": sha256_bytes(payload)}

    def get_paginator(self, name):
        return _Paginator(self)

    def get_object(self, *, Bucket, Key, **_):
        if (Bucket, Key) not in self.objects:
            raise _error("NoSuchKey", 404)
        payload = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(payload), "ContentLength": len(payload)}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _error("404", 404)
        payload = self.objects[(Bucket, Key)]
        return {"ContentLength": len(payload), "ETag": f'"{sha256_bytes(payload)[:32]}"',
                "Metadata": self.metadata[(Bucket, Key)]}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, Metadata=None, **_):
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            raise _error("PreconditionFailed", 412)
        self.objects[(Bucket, Key)] = Body
        self.metadata[(Bucket, Key)] = Metadata or {"sha256": sha256_bytes(Body)}


def _spec(**overrides) -> DatasetSpec:
    values = dict(
        dataset=KIT,
        source_manifest=ManifestRef("s3://datasets/kitscenes/raw-snapshots/r/s/manifest.json", "v1", "a" * 64),
        pack_version=current_pack_version(KIT),
        image_size=256,
        reasoning=ReasoningSpec("openai_compatible", "nvidia/Cosmos3-Nano", "p", 10),
    )
    values.update(overrides)
    return DatasetSpec(**values)


def _resolve(s3, spec=None, contracts=None, execution="exec-1"):
    return resolve_published_dataset(
        s3, bucket=BUCKET, spec=spec or _spec(), published_name="kitscenes",
        expected_contracts=contracts if contracts is not None else pack_contracts(),
        execution_id=execution,
    )


def _publish(s3, spec, version, *, contracts=None):
    """Simulate a finished build: partition view, views manifest, lineage, index."""
    dataset_id = compute_dataset_id(spec)
    prefix = partition_view_prefix("kitscenes", version, "p-a")
    manifest = canonical_json_bytes({"partition_id": "p-a", "shard_names": ["a.tar"], "total_samples": 5})
    s3.put(f"{prefix}/manifest.json", manifest)
    s3.put(f"{prefix}/a.tar", b"tar")
    views = views_manifest(
        dataset_id=dataset_id, published_name="kitscenes", version=version,
        contracts=contracts or pack_contracts(), image_size=spec.image_size,
        partitions=[{"partition_id": "p-a", "prefix": prefix, "manifest_sha256": sha256_bytes(manifest), "sample_count": 5}],
    )
    views_sha = write_views_manifest(s3, bucket=BUCKET, views=views)
    record, units = dataset_lineage(
        dataset_id=dataset_id, spec=canonical_dataset_spec(spec),
        published={"name": "kitscenes", "version": version},
        raw={"snapshot_manifest_sha256": "a" * 64}, labels=None,
        units=[UnitLineage("scene-a", "p-a", "k", "sha256:" + "1" * 64, "2" * 64, None, None, None, 5, 0)],
        units_uri="s3://x", views={"key": views_manifest_key("kitscenes", version), "sha256": views_sha, "partition_ids": ["p-a"]},
        console=None, pack={"contracts": pack_contracts()}, build={"flyte_execution_id": "exec-1"},
    )
    _, lineage_sha = write_dataset_lineage(
        s3, bucket=BUCKET, published_name="kitscenes", version=version, record=record,
        units_payload=units, units_key_name=units_key("kitscenes", version),
    )
    write_dataset_index(s3, bucket=BUCKET, spec=spec, published_name="kitscenes",
                        version=version, views_sha256=views_sha, lineage_sha256=lineage_sha)


def test_unpublished_spec_reserves_the_next_version_once():
    s3 = S3()
    s3.put("kitscenes/v3.5/shards/manifest.json", b"{}")
    first = _resolve(s3)
    assert (first.published, first.version, first.reserved_by) == (False, "v3.6", "exec-1")
    again = _resolve(s3, execution="exec-2")
    assert (again.version, again.reserved_by) == ("v3.6", "exec-1")


def test_different_specs_never_share_a_reserved_version():
    s3 = S3()
    a = _resolve(s3).version
    b = _resolve(s3, spec=_spec(image_size=512)).version
    assert a != b


def test_reservation_race_on_a_version_number_moves_on():
    s3 = S3()
    s3.put("kitscenes/_stages/versions/v1.0.json", b"{}")
    assert reserve_version(s3, bucket=BUCKET, published_name="kitscenes", dataset_id="d" * 64, execution_id="e") == ("v1.1", "e")


def test_committed_build_is_reused_with_partition_views():
    s3 = S3()
    spec = _spec()
    version = _resolve(s3).version
    _publish(s3, spec, version)
    handle = _resolve(s3)
    assert handle.published and handle.version == version and handle.samples == 5
    assert handle.partition_uris == [f"s3://{BUCKET}/{partition_view_prefix('kitscenes', version, 'p-a')}"]


def test_publication_under_other_contracts_is_never_reused():
    s3 = S3()
    spec = _spec()
    _publish(s3, spec, "v1.0", contracts={**pack_contracts(), "parser_version": "v0"})
    with pytest.raises(ValueError, match="other contracts"):
        _resolve(s3)


def test_changed_partition_manifest_fails_reuse():
    s3 = S3()
    spec = _spec()
    _publish(s3, spec, "v1.0")
    s3.objects[(BUCKET, f"{partition_view_prefix('kitscenes', 'v1.0', 'p-a')}/manifest.json")] = b"{}"
    with pytest.raises(ValueError, match="changed"):
        _resolve(s3)


def test_missing_partition_shard_fails_reuse():
    s3 = S3()
    spec = _spec()
    _publish(s3, spec, "v1.0")
    del s3.objects[(BUCKET, f"{partition_view_prefix('kitscenes', 'v1.0', 'p-a')}/a.tar")]
    with pytest.raises(ClientError):
        _resolve(s3)


def test_index_for_another_published_name_does_not_match():
    s3 = S3()
    spec = _spec()
    _publish(s3, spec, "v1.0")
    handle = resolve_published_dataset(
        s3, bucket=BUCKET, spec=spec, published_name="kitscenes-val",
        expected_contracts=pack_contracts(), execution_id="e",
    )
    assert handle.published is False and handle.published_name == "kitscenes-val"


LABELS = LabelIdentity("openai_compatible", "nvidia/Cosmos3-Nano", "p", 10, "v2", "v1")


def _identity():
    return label_identity(KIT, "rev", "scene-a", LABELS)


def _write_attempt(s3, attempt="exec-1-0"):
    prefix = label_attempt_prefix(KIT, label_id(_identity()), attempt)
    s3.put(f"{prefix}/meta.json", b'{"num_records": 1}')
    s3.put(f"{prefix}/records.jsonl", b'{"uid": "a"}\n')


def test_label_set_is_claimed_written_committed_and_reused():
    s3 = S3()
    assert lookup_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity()) is None
    claim_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), execution_id="exec-1", claimed_at="t")
    _write_attempt(s3)
    committed = commit_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), attempt_id="exec-1-0", label_count=1)
    assert lookup_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity()) == committed


def test_another_execution_cannot_label_a_claimed_unit():
    s3 = S3()
    claim_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), execution_id="exec-1", claimed_at="t")
    claim_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), execution_id="exec-1", claimed_at="t2")
    with pytest.raises(RuntimeError, match="being produced"):
        claim_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), execution_id="exec-2", claimed_at="t")


def test_uncommitted_attempts_are_never_read_and_commit_is_once():
    s3 = S3()
    _write_attempt(s3, "exec-1-0")
    _write_attempt(s3, "exec-1-1")
    commit_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), attempt_id="exec-1-1", label_count=1)
    with pytest.raises(RuntimeError, match="different content"):
        commit_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), attempt_id="exec-1-0", label_count=1)
    found = lookup_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity())
    assert found["attempt_id"] == "exec-1-1"


def test_changed_committed_label_file_fails_instead_of_relabeling():
    s3 = S3()
    _write_attempt(s3)
    commit_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity(), attempt_id="exec-1-0", label_count=1)
    key = f"{label_attempt_prefix(KIT, label_id(_identity()), 'exec-1-0')}/records.jsonl"
    s3.objects[(BUCKET, key)] = b"tampered"
    with pytest.raises(ValueError, match="changed"):
        lookup_label_set(s3, bucket=BUCKET, dataset=KIT, identity=_identity())


def test_dataset_handle_round_trips_through_flyte():
    from flytekit.core.context_manager import FlyteContextManager
    from flytekit.core.type_engine import TypeEngine

    ctx = FlyteContextManager.current_context()
    for value, python_type in [
        (DatasetHandle("a" * 64, "kitscenes", "v3.6", True, "exec-1", "b" * 64, "k", "c" * 64, 5, ["s3://x"]), DatasetHandle),
        (_spec(), DatasetSpec),
    ]:
        literal = TypeEngine.to_literal(ctx, value, python_type, TypeEngine.to_literal_type(python_type))
        assert TypeEngine.to_python_value(ctx, literal, python_type) == value
