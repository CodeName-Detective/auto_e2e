"""Scene stage store identity and record validation tests."""

from __future__ import annotations

import dataclasses

import pytest

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.autoe2e_stages import (
    LabelIdentity,
    SceneStage,
    label_identity,
    listing_digest,
    raw_identity,
    stage_id,
    stage_record,
    stage_record_key,
    validate_label_meta,
    validate_stage_record,
)

KIT = AutoE2EDataset.KITSCENES
REV = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
SCENE = "008fba36-5e82-e02b-8edf-a55f5271758d"
LABELS = LabelIdentity(
    teacher="openai_compatible",
    prompt_version="action_relevant_reasoning_v3_temporal_front256",
    label_stride=10,
    label_policy_version="v2",
    uid_schema_version="v1",
)
OBJECTS = [
    {"relative": "data/train/a.mcap", "size": 10, "etag": '"e1"'},
    {"relative": "data/sequence_archives.csv", "size": 3, "etag": "e2"},
]
URI = "s3://artifacts/sv/run/abc"


def test_raw_identity_has_no_pack_contract_fields():
    identity = raw_identity(KIT, REV, SCENE)
    assert set(identity) == {"stage", "dataset", "source_revision", "scene_id"}


def test_label_identity_has_no_pack_contract_fields():
    identity = label_identity(KIT, REV, SCENE, LABELS)
    flat = repr(identity)
    for forbidden in ("parser", "shard", "geometry", "pack"):
        assert forbidden not in flat


def test_stage_ids_separate_scenes_revisions_and_label_settings():
    base = stage_id(label_identity(KIT, REV, SCENE, LABELS))
    assert stage_id(label_identity(KIT, REV, "other-scene", LABELS)) != base
    assert stage_id(label_identity(KIT, "rev2", SCENE, LABELS)) != base
    for change in (
        {"teacher": "mock"},
        {"prompt_version": "p2"},
        {"label_stride": 5},
        {"label_policy_version": "v3"},
        {"uid_schema_version": "v2"},
    ):
        changed = dataclasses.replace(LABELS, **change)
        assert stage_id(label_identity(KIT, REV, SCENE, changed)) != base
    assert stage_id(raw_identity(KIT, REV, SCENE)) != base


@pytest.mark.parametrize("scene_id", ["", "../x", "a/b", "-lead"])
def test_unsafe_scene_ids_are_rejected(scene_id):
    with pytest.raises(ValueError):
        raw_identity(KIT, REV, scene_id)


def test_record_key_layout():
    digest = stage_id(raw_identity(KIT, REV, SCENE))
    assert stage_record_key(KIT, SceneStage.RAW, digest) == (
        f"kitscenes/_stages/raw/{digest}.json"
    )


def test_listing_digest_is_order_and_quote_insensitive():
    reordered = [
        {**OBJECTS[1], "etag": '"e2"'},
        {**OBJECTS[0], "etag": "e1"},
    ]
    assert listing_digest(OBJECTS) == listing_digest(reordered)
    digest, count, total = listing_digest(OBJECTS)
    assert (count, total) == (2, 13)


def test_listing_digest_rejects_empty_and_duplicate_listings():
    with pytest.raises(ValueError, match="empty"):
        listing_digest([])
    with pytest.raises(ValueError, match="twice"):
        listing_digest([OBJECTS[0], OBJECTS[0]])


def test_record_validates_against_the_same_listing():
    identity = raw_identity(KIT, REV, SCENE)
    record = stage_record(identity, artifact_uri=URI, objects=OBJECTS)
    validate_stage_record(record, identity, objects=list(reversed(OBJECTS)))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda objects: objects[:1],
        lambda objects: [{**objects[0], "size": 11}, objects[1]],
        lambda objects: [{**objects[0], "etag": "changed"}, objects[1]],
        lambda objects: objects + [{"relative": "new", "size": 1, "etag": "e"}],
    ],
)
def test_record_rejects_a_changed_artifact(mutate):
    identity = raw_identity(KIT, REV, SCENE)
    record = stage_record(identity, artifact_uri=URI, objects=OBJECTS)
    with pytest.raises(ValueError, match="changed"):
        validate_stage_record(record, identity, objects=mutate(OBJECTS))


def test_record_rejects_another_identity():
    record = stage_record(raw_identity(KIT, REV, SCENE), artifact_uri=URI, objects=OBJECTS)
    with pytest.raises(ValueError, match="identity differs"):
        validate_stage_record(
            record, raw_identity(KIT, REV, "other-scene"), objects=OBJECTS
        )


def test_record_requires_an_s3_prefix():
    with pytest.raises(ValueError):
        stage_record(raw_identity(KIT, REV, SCENE), artifact_uri="/tmp/x", objects=OBJECTS)


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
    validate_label_meta(meta, label_identity(KIT, REV, SCENE, LABELS))
    with pytest.raises(ValueError, match="prompt_version"):
        validate_label_meta(
            {**meta, "prompt_version": "other"},
            label_identity(KIT, REV, SCENE, LABELS),
        )
