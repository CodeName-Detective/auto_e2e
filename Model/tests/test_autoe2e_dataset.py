"""wf_autoe2e dataset identity and publication validation tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from Platform.pipelines.autoe2e_dataset import (
    AutoE2EDataset,
    DatasetSpec,
    ManifestRef,
    ReasoningSpec,
    compute_dataset_id,
    dataset_index_key,
    dataset_index_record,
    next_dataset_version,
    validate_dataset_index_record,
    validate_publication,
)
from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    canonical_json_bytes,
    sha256_bytes,
)


def _spec(**overrides) -> DatasetSpec:
    values = dict(
        dataset=AutoE2EDataset.KITSCENES,
        source_manifest=ManifestRef(
            uri="s3://datasets/kitscenes/raw-snapshots/rev/snap/manifest.json",
            version_id="zpW2",
            sha256="a" * 64,
        ),
        pack_version="v3.3",
        image_size=256,
        reasoning=ReasoningSpec(
            teacher="openai_compatible",
            prompt_version="action_relevant_reasoning_v3_temporal_front256",
            label_stride=10,
        ),
    )
    values.update(overrides)
    return DatasetSpec(**values)


def _published(spec: DatasetSpec, version: str = "v3.4"):
    manifest = {
        "status": "ready",
        "dataset": spec.dataset.value,
        "version": version,
        "shard_count": 1,
        "shard_entries": [{"name": "a.tar", "key": "k", "byte_size": 1}],
    }
    payload = canonical_json_bytes(manifest, pretty=True)
    digest = sha256_bytes(payload)
    record = dataset_index_record(
        spec,
        version=version,
        manifest_key=f"kitscenes/{version}/shards/manifest.json",
        manifest_sha256=digest,
    )
    lock = {"schema_version": PUBLICATION_SCHEMA, "manifest_sha256": digest}
    return record, lock, payload


def test_dataset_id_is_stable_and_hex():
    assert compute_dataset_id(_spec()) == compute_dataset_id(_spec())
    assert len(compute_dataset_id(_spec())) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"unit_limit": 10},
        {"pack_params": {"partition_size": "4"}},
        {"pack_version": "v3.4"},
        {"image_size": 512},
        {"reasoning": None},
        {"dataset": AutoE2EDataset.NUPLAN},
    ],
)
def test_dataset_id_changes_with_every_identity_field(overrides):
    assert compute_dataset_id(_spec(**overrides)) != compute_dataset_id(_spec())


def test_dataset_id_changes_with_source_manifest_digest():
    ref = dataclasses.replace(_spec().source_manifest, sha256="b" * 64)
    assert compute_dataset_id(_spec(source_manifest=ref)) != compute_dataset_id(
        _spec()
    )


def test_pack_params_order_does_not_change_the_id():
    a = _spec(pack_params={"a": "1", "b": "2"})
    b = _spec(pack_params={"b": "2", "a": "1"})
    assert compute_dataset_id(a) == compute_dataset_id(b)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unit_limit": -1},
        {"pack_params": {"": "x"}},
        {"pack_params": {"k": 1}},
        {"pack_version": ""},
        {"image_size": 0},
        {"source_manifest": ManifestRef(uri="https://x/a.json", version_id="v", sha256="a" * 64)},
        {"source_manifest": ManifestRef(uri="s3://b/a.json", version_id="", sha256="a" * 64)},
        {"source_manifest": ManifestRef(uri="s3://b/a.json", version_id="v", sha256="A" * 64)},
        {"reasoning": ReasoningSpec(teacher="t", prompt_version="p", label_stride=0)},
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValueError):
        compute_dataset_id(_spec(**overrides))


def test_index_key_lives_outside_version_directories():
    dataset_id = compute_dataset_id(_spec())
    assert dataset_index_key(AutoE2EDataset.KITSCENES, dataset_id) == (
        f"kitscenes/by-id/{dataset_id}.json"
    )
    with pytest.raises(ValueError):
        dataset_index_key(AutoE2EDataset.KITSCENES, "not-a-digest")


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ([], "v1.0"),
        (["v3.3", "v2.2", "v3.0"], "v3.4"),
        (["v9", "v10"], "v10.1"),
        (["by-id", "v2.1", "tmp"], "v2.2"),
    ],
)
def test_next_dataset_version(existing, expected):
    assert next_dataset_version(existing) == expected


def test_index_record_round_trips():
    spec = _spec()
    record, _, _ = _published(spec)
    validate_dataset_index_record(json.loads(json.dumps(record)), spec)
    assert record["dataset_id"] == compute_dataset_id(spec)


def test_index_record_rejects_a_different_spec():
    record, _, _ = _published(_spec())
    with pytest.raises(ValueError, match="spec differs"):
        validate_dataset_index_record(record, _spec(image_size=512))


def test_index_record_rejects_extra_fields_and_wrong_key():
    spec = _spec()
    record, _, _ = _published(spec)
    with pytest.raises(ValueError):
        validate_dataset_index_record({**record, "extra": 1}, spec)
    with pytest.raises(ValueError):
        dataset_index_record(
            spec,
            version="v3.4",
            manifest_key="kitscenes/v3.3/shards/manifest.json",
            manifest_sha256="a" * 64,
        )


def test_publication_validates():
    record, lock, payload = _published(_spec())
    manifest = validate_publication(record, lock=lock, manifest_payload=payload)
    assert manifest["version"] == "v3.4"


def test_publication_rejects_tampered_manifest_bytes():
    record, lock, payload = _published(_spec())
    with pytest.raises(ValueError, match="lock does not match"):
        validate_publication(record, lock=lock, manifest_payload=payload + b" ")


def test_publication_rejects_lock_from_another_schema():
    record, lock, payload = _published(_spec())
    with pytest.raises(ValueError, match="lock schema"):
        validate_publication(
            record,
            lock={**lock, "schema_version": "v1"},
            manifest_payload=payload,
        )


def test_publication_rejects_index_pointing_at_other_bytes():
    record, lock, payload = _published(_spec())
    with pytest.raises(ValueError, match="index does not match"):
        validate_publication(
            {**record, "manifest_sha256": "b" * 64},
            lock=lock,
            manifest_payload=payload,
        )
