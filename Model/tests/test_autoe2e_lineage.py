"""Lineage record contract tests."""

from __future__ import annotations


import pytest

from Platform.pipelines.autoe2e_lineage import (
    CHECKPOINT_LINEAGE_SCHEMA,
    DATASET_LINEAGE_SCHEMA,
    EVALUATION_SCHEMA,
    UnitLineage,
    checkpoint_lineage,
    dataset_lineage,
    evaluation_record,
    parse_record,
    read_units_jsonl,
    record_bytes,
    record_sha256,
    units_jsonl,
)

H = lambda c: c * 64  # noqa: E731


def _unit(unit_id="a", **overrides):
    values = dict(
        unit_id=unit_id,
        partition_id=f"p-{unit_id}",
        raw_tar_key=f"kitscenes/raw-snapshots/r/s/units/{unit_id}.tar",
        raw_tar_sha256=H("1"),
        raw_index_sha256=H("2"),
        label_id=H("3"),
        label_record_sha256=H("4"),
        label_origin="reused",
        sample_count=10,
        label_count=2,
    )
    values.update(overrides)
    return UnitLineage(**values)


SPEC = {"dataset": "kitscenes", "schema_version": "autoe2e_dataset_spec_v1"}


def _spec_id(spec=SPEC):
    from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes

    return sha256_bytes(canonical_json_bytes(spec))


def _dataset(units=None, labels={"teacher": "openai_compatible"}):
    return dataset_lineage(
        dataset_id=_spec_id(),
        spec=SPEC,
        published={"name": "kitscenes", "version": "v3.6"},
        raw={"upstream": "KIT-MRT/KITScenes-Multimodal", "revision": "r", "snapshot_manifest_sha256": H("f")},
        labels=labels,
        units=units or [_unit("b"), _unit("a", label_origin="created")],
        units_uri="s3://datasets/kitscenes/v3.6/lineage/units.jsonl",
        views={"key": "kitscenes/v3.6/partitions/manifest.json", "sha256": H("7"), "partition_ids": ["p-a", "p-b"]}, console=None,
        pack={"contracts": {"shard": "v13"}, "image_digest": "sha256:x", "git_commit": "c"},
        build={"flyte_execution_id": "x"},
    )


def test_dataset_id_must_be_the_spec_digest():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="digest of the given spec"):
        dataset_lineage(
            dataset_id=H("d"), spec=SPEC,
            published={"name": "kitscenes", "version": "v3.6"},
            raw={"snapshot_manifest_sha256": H("f")}, labels=None,
            units=[_unit("a", label_id=None, label_record_sha256=None, label_origin=None)],
            units_uri="s3://x", views={"key": "kitscenes/v3.6/partitions/manifest.json", "sha256": H("7"), "partition_ids": ["p-a", "p-b"]}, console=None, pack={}, build={},
        )


def test_units_jsonl_is_canonical():
    payload = units_jsonl([_unit("a")])
    with pytest.raises(ValueError, match="canonical"):
        read_units_jsonl(payload.replace(b'{"', b'{ "', 1))


def test_units_jsonl_round_trips_sorted_and_deterministic():
    units = [_unit("b"), _unit("a")]
    payload = units_jsonl(units)
    assert payload == units_jsonl(list(reversed(units)))
    assert [u.unit_id for u in read_units_jsonl(payload)] == ["a", "b"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw_tar_sha256": "x"},
        {"label_id": None},
        {"label_origin": "maybe"},
        {"sample_count": -1},
    ],
)
def test_invalid_unit_rows_are_rejected(overrides):
    with pytest.raises(ValueError):
        units_jsonl([_unit(**overrides)])


def test_duplicate_units_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        units_jsonl([_unit("a"), _unit("a")])


def test_dataset_lineage_summarizes_labels_and_pins_units():
    record, payload = _dataset()
    assert record["schema_version"] == DATASET_LINEAGE_SCHEMA
    assert record["labels"]["reused"] == 1 and record["labels"]["created"] == 1
    assert record["units"]["rows"] == 2
    assert record["totals"] == {"samples": 20, "reasoning_labels": 4}
    from Platform.pipelines.dataset_publication import sha256_bytes

    assert record["units"]["sha256"] == sha256_bytes(payload)


def test_label_set_digest_ignores_label_origin():
    a, _ = _dataset([_unit("a", label_origin="created")])
    b, _ = _dataset([_unit("a", label_origin="reused")])
    assert a["labels"]["label_set_sha256"] == b["labels"]["label_set_sha256"]


def test_unlabelled_dataset_rejects_label_rows():
    with pytest.raises(ValueError, match="has none"):
        _dataset(labels=None)
    record, _ = _dataset(
        [_unit("a", label_id=None, label_record_sha256=None, label_origin=None)],
        labels=None,
    )
    assert record["labels"] is None


def test_labelled_dataset_needs_labels_for_every_unit():
    with pytest.raises(ValueError, match="every unit"):
        _dataset([_unit("a"), _unit("b", label_id=None, label_record_sha256=None, label_origin=None)])


def _ckpt(**overrides):
    values = dict(
        checkpoint={"sha256": H("c"), "uri": "s3://x/best.pt", "role": "best", "epoch": 6},
        parent={"model_id": "m-1", "model_version": "p/1", "checkpoint_sha256": H("9"), "lineage_sha256": H("8"), "stage_index": 1},
        dataset={"dataset_id": H("d"), "lineage_uri": "s3://x/dataset.json", "lineage_sha256": H("e")},
        split={"policy": "kitscenes_train_dev_v4", "validation_units_sha256": H("7"), "validation_units": 40},
        training={"flyte_execution_id": "x"},
        selection={"metric": "eval/internal_val/ade_6p4s", "value": 2.6},
    )
    values.update(overrides)
    return checkpoint_lineage(**values)


def test_checkpoint_stage_index_follows_the_parent():
    assert _ckpt()["stage_index"] == 2
    assert _ckpt(parent=None)["stage_index"] == 1


def test_imported_checkpoint_records_unknowns_not_guesses():
    record = _ckpt(
        checkpoint={"sha256": H("c"), "uri": "s3://x.pt", "role": "imported"},
        parent=None, dataset=None, split=None, training=None, selection=None,
    )
    assert record["stage_index"] == 0
    assert record["dataset"] == record["split"] == record["training"] == {"status": "unknown"}


def test_imported_checkpoint_cannot_claim_a_parent():
    with pytest.raises(ValueError, match="no known parent"):
        _ckpt(
            checkpoint={"sha256": H("c"), "uri": "s3://x.pt", "role": "imported"},
            dataset=None, split=None, training=None, selection=None,
        )


def test_trained_checkpoint_cannot_omit_its_dataset():
    with pytest.raises(ValueError, match="needs dataset"):
        _ckpt(dataset=None)
    with pytest.raises(ValueError, match="imported"):
        _ckpt(checkpoint={"sha256": H("c"), "uri": "u", "role": "imported"}, parent=None)


def _eval(**overrides):
    values = dict(
        checkpoint_sha256=H("c"),
        checkpoint_lineage_sha256=H("8"),
        suite="kitscenes_benchmark",
        dataset={"dataset_id": H("d"), "lineage_sha256": H("e"), "published": "kitscenes-val/v3.5"},
        protocol={"name": "kitscenes_paper_approximation", "version": "v1"},
        metrics={"ade_3s": 0.71, "fde_3s": 1.9},
        per_sample={"uri": "s3://x.parquet", "sha256": H("6"), "rows": 200},
        code={"image_digest": "sha256:x", "git_commit": "c"},
        flyte_execution_id="x",
        finished_at="2026-10-21T00:00:00Z",
    )
    values.update(overrides)
    return evaluation_record(**values)


def test_evaluation_record_always_claims_model_reexecution():
    assert _eval()["model_reexecuted"] is True
    assert _eval()["schema_version"] == EVALUATION_SCHEMA


@pytest.mark.parametrize("metrics", [{}, {"ade": float("nan")}, {"ade": float("inf")}, {"ade": True}, {"ade": "1"}])
def test_evaluation_rejects_missing_or_non_numeric_metrics(metrics):
    with pytest.raises(ValueError):
        _eval(metrics=metrics)


def test_parse_record_checks_digest_schema_and_canonical_form():
    record = _ckpt()
    payload = record_bytes(record)
    assert parse_record(payload, expected_sha256=record_sha256(record), schema=CHECKPOINT_LINEAGE_SCHEMA) == record
    with pytest.raises(ValueError, match="differ"):
        parse_record(payload + b" ", expected_sha256=record_sha256(record), schema=CHECKPOINT_LINEAGE_SCHEMA)
    with pytest.raises(ValueError, match="expected"):
        parse_record(payload, expected_sha256=record_sha256(record), schema=EVALUATION_SCHEMA)
    import json
    from Platform.pipelines.dataset_publication import sha256_bytes

    pretty = json.dumps(record, indent=1).encode()
    with pytest.raises(ValueError, match="canonical"):
        parse_record(pretty, expected_sha256=sha256_bytes(pretty), schema=CHECKPOINT_LINEAGE_SCHEMA)


def test_units_must_belong_to_published_partitions():
    with pytest.raises(ValueError, match="not published"):
        _dataset([_unit("a", partition_id="p-zzz")])
    record, _ = _dataset([_unit("a", partition_id="*")])
    assert record["views"]["partitions"] == 2
