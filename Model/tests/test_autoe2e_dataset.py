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
    partition_view_prefix,
    validate_dataset_index_record,
    validate_views,
    views_manifest,
)
from Platform.pipelines.dataset_publication import (
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
            teacher_model="nvidia/Cosmos3-Nano",
            prompt_version="action_relevant_reasoning_v3_temporal_front256",
            label_stride=10,
        ),
    )
    values.update(overrides)
    return DatasetSpec(**values)


CONTRACTS = {"parser_version": "v3", "shard_schema_version": "v13"}


def _published(spec: DatasetSpec, version: str = "v3.4", contracts=CONTRACTS):
    views = views_manifest(
        dataset_id=compute_dataset_id(spec), published_name="kitscenes", version=version,
        contracts=contracts, image_size=spec.image_size,
        partitions=[{"partition_id": "p-a", "prefix": partition_view_prefix("kitscenes", version, "p-a"),
                     "manifest_sha256": "b" * 64, "sample_count": 5}],
    )
    payload = canonical_json_bytes(views)
    record = dataset_index_record(
        spec, published_name="kitscenes", version=version,
        views_sha256=sha256_bytes(payload), lineage_sha256="c" * 64,
    )
    return record, payload


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
        {"reasoning": ReasoningSpec(teacher="t", teacher_model="m", prompt_version="p", label_stride=0)},
        {"reasoning": ReasoningSpec(teacher="t", teacher_model="", prompt_version="p", label_stride=1)},
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValueError):
        compute_dataset_id(_spec(**overrides))


def test_index_key_lives_outside_version_directories():
    dataset_id = compute_dataset_id(_spec())
    assert dataset_index_key("kitscenes-val", dataset_id) == f"kitscenes-val/by-id/{dataset_id}.json"
    with pytest.raises(ValueError):
        dataset_index_key("kitscenes", "not-a-digest")


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ([], "v1.0"),
        (["v3.3", "v2.2", "v3.0"], "v3.4"),
        (["v9", "v10"], "v10.1"),
        (["by-id", "v2.1", "tmp", "_stages"], "v2.2"),
    ],
)
def test_next_dataset_version(existing, expected):
    assert next_dataset_version(existing) == expected


def test_index_record_round_trips_and_pins_lineage():
    spec = _spec()
    record, _ = _published(spec)
    validate_dataset_index_record(json.loads(json.dumps(record)), spec, published_name="kitscenes")
    assert record["dataset_id"] == compute_dataset_id(spec)
    assert record["lineage_key"] == "kitscenes/v3.4/lineage/dataset.json"
    assert record["views_key"] == "kitscenes/v3.4/partitions/manifest.json"


def test_index_record_rejects_another_spec_or_published_name():
    record, _ = _published(_spec())
    with pytest.raises(ValueError, match="spec differs"):
        validate_dataset_index_record(record, _spec(image_size=512), published_name="kitscenes")
    with pytest.raises(ValueError, match="another published"):
        validate_dataset_index_record(record, _spec(), published_name="kitscenes-val")
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_dataset_index_record({**record, "extra": 1}, _spec(), published_name="kitscenes")


def test_views_validate_against_index_and_running_contracts():
    record, payload = _published(_spec())
    views = validate_views(record, views_payload=payload, expected_contracts=CONTRACTS)
    assert views["samples"] == 5
    with pytest.raises(ValueError, match="other contracts"):
        validate_views(record, views_payload=payload, expected_contracts={**CONTRACTS, "parser_version": "v4"})
    with pytest.raises(ValueError, match="differs from the index"):
        validate_views(record, views_payload=payload + b" ", expected_contracts=CONTRACTS)


def test_views_reject_partitions_outside_the_publication():
    with pytest.raises(ValueError, match="outside"):
        views_manifest(
            dataset_id="d" * 64, published_name="kitscenes", version="v1.0", contracts={}, image_size=1,
            partitions=[{"partition_id": "p", "prefix": "elsewhere/p", "manifest_sha256": "b" * 64, "sample_count": 1}],
        )
