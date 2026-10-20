"""Per-unit label store identity and record validation tests."""

from __future__ import annotations

import dataclasses

import pytest

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.autoe2e_labels import (
    LabelIdentity,
    label_id,
    label_identity,
    label_prefix,
    label_record,
    label_record_key,
    label_set_sha256,
    listing_digest,
    validate_label_meta,
    validate_label_record,
)

KIT = AutoE2EDataset.KITSCENES
REV = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
UNIT = "008fba36-5e82-e02b-8edf-a55f5271758d"
LABELS = LabelIdentity(
    teacher="openai_compatible",
    teacher_model="nvidia/Cosmos3-Nano",
    prompt_version="action_relevant_reasoning_v3_temporal_front256",
    label_stride=10,
    label_policy_version="v2",
    uid_schema_version="v1",
)
OBJECTS = [
    {"relative": "records.jsonl", "size": 10, "etag": '"e1"'},
    {"relative": "meta.json", "size": 3, "etag": "e2"},
]


def _identity(**overrides):
    return label_identity(KIT, REV, UNIT, dataclasses.replace(LABELS, **overrides))


def test_identity_has_no_pack_contract_fields():
    flat = repr(_identity())
    for forbidden in ("parser", "shard", "geometry", "pack"):
        assert forbidden not in flat


@pytest.mark.parametrize(
    "change",
    [
        {"teacher": "mock"},
        {"teacher_model": "nvidia/Cosmos3-Super"},
        {"prompt_version": "p2"},
        {"label_stride": 5},
        {"label_policy_version": "v3"},
        {"uid_schema_version": "v2"},
    ],
)
def test_label_id_changes_with_every_label_setting(change):
    assert label_id(_identity(**change)) != label_id(_identity())


def test_label_id_separates_units_and_revisions():
    base = label_id(_identity())
    assert label_id(label_identity(KIT, REV, "other", LABELS)) != base
    assert label_id(label_identity(KIT, "rev2", UNIT, LABELS)) != base
    assert label_id(label_identity(AutoE2EDataset.L2D, REV, UNIT, LABELS)) != base


@pytest.mark.parametrize("unit", ["", "../x", "a/b", "-lead"])
def test_unsafe_unit_ids_are_rejected(unit):
    with pytest.raises(ValueError):
        label_identity(KIT, REV, unit, LABELS)


def test_store_layout():
    digest = label_id(_identity())
    assert label_prefix(KIT, digest) == f"kitscenes/labels/{digest}"
    assert label_record_key(KIT, digest) == f"kitscenes/_stages/labels/{digest}.json"


def test_listing_digest_is_order_and_quote_insensitive():
    reordered = [{**OBJECTS[1], "etag": '"e2"'}, {**OBJECTS[0], "etag": "e1"}]
    assert listing_digest(OBJECTS) == listing_digest(reordered)
    assert listing_digest(OBJECTS)[1:] == (2, 13)


def test_listing_digest_rejects_empty_and_duplicates():
    with pytest.raises(ValueError, match="empty"):
        listing_digest([])
    with pytest.raises(ValueError, match="twice"):
        listing_digest([OBJECTS[0], OBJECTS[0]])


def test_record_points_into_the_label_store():
    record = label_record(_identity(), bucket="datasets", attempt_id="exec-0", objects=OBJECTS, label_count=1)
    assert record["artifact_uri"] == (
        f"s3://datasets/kitscenes/labels/{label_id(_identity())}/exec-0"
    )


def test_record_validates_against_the_same_listing():
    record = label_record(_identity(), bucket="datasets", attempt_id="exec-0", objects=OBJECTS, label_count=1)
    validate_label_record(
        record, _identity(), bucket="datasets", objects=list(reversed(OBJECTS))
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda objects: objects[:1],
        lambda objects: [{**objects[0], "size": 11}, objects[1]],
        lambda objects: [{**objects[0], "etag": "changed"}, objects[1]],
        lambda objects: objects + [{"relative": "new", "size": 1, "etag": "e"}],
    ],
)
def test_record_rejects_changed_files(mutate):
    record = label_record(_identity(), bucket="datasets", attempt_id="exec-0", objects=OBJECTS, label_count=1)
    with pytest.raises(ValueError, match="changed"):
        validate_label_record(record, _identity(), bucket="datasets", objects=mutate(OBJECTS))


def test_record_rejects_another_identity_or_bucket():
    record = label_record(_identity(), bucket="datasets", attempt_id="exec-0", objects=OBJECTS, label_count=1)
    with pytest.raises(ValueError, match="identity differs"):
        validate_label_record(record, _identity(teacher="mock"), bucket="datasets", objects=OBJECTS)
    with pytest.raises(ValueError, match="changed"):
        validate_label_record(record, _identity(), bucket="other", objects=OBJECTS)


def test_label_set_digest_is_order_free_and_rejects_duplicates():
    rows = [("b", "lb", "rb"), ("a", "la", "ra")]
    assert label_set_sha256(rows) == label_set_sha256(list(reversed(rows)))
    with pytest.raises(ValueError, match="twice"):
        label_set_sha256([("a", "x", "y"), ("a", "z", "w")])


def test_label_meta_from_the_july_full_run_matches():
    meta = {
        "dataset": "KIT-MRT/KITScenes-Multimodal",
        "split": "train",
        "teacher": "openai_compatible",
        "source_revision": REV,
        "prompt_version": "action_relevant_reasoning_v3_temporal_front256",
        "num_records": 1,
        "label_policy_version": "v2",
    }
    validate_label_meta(meta, _identity())
    with pytest.raises(ValueError, match="split"):
        validate_label_meta({**meta, "split": "val"}, _identity())
    with pytest.raises(ValueError, match="prompt_version"):
        validate_label_meta({**meta, "prompt_version": "other"}, _identity())
