"""MLflow placement logic for wf_autoe2e, against an in-memory client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Platform.pipelines.autoe2e_mlflow import (
    DESCRIPTION_TAG,
    REGISTERED_MODEL,
    description,
    ensure_logged_model,
    ensure_version,
    flatten,
    lineage_artifacts,
    log_evaluation,
    mlflow_dataset,
    model_params,
)

H = lambda c: c * 64  # noqa: E731


class _Client:
    def __init__(self):
        self.models = {}
        self.versions = []
        self.artifacts = {}
        self.inputs = []
        self.metrics = []
        self.registered = set()

    def search_logged_models(self, experiment_ids, filter_string):
        sha = filter_string.split("'")[1]
        return [m for m in self.models.values() if m.params.get("checkpoint_sha256") == sha]

    def create_logged_model(self, experiment_id, name, source_run_id, params, tags):
        model = SimpleNamespace(model_id=f"m-{len(self.models)}", params=dict(params), tags=dict(tags), name=name, status="PENDING")
        self.models[model.model_id] = model
        return model

    def log_model_artifacts(self, model_id, root):
        from pathlib import Path

        for path in Path(root).rglob("*"):
            if path.is_file():
                self.artifacts[(model_id, str(path.relative_to(root)))] = path.read_bytes()

    def finalize_logged_model(self, model_id, status):
        self.models[model_id].status = status

    def get_logged_model(self, model_id):
        return self.models[model_id]

    def create_registered_model(self, name):
        from mlflow.exceptions import MlflowException

        if name in self.registered:
            raise MlflowException("RESOURCE_ALREADY_EXISTS: exists")
        self.registered.add(name)

    def search_model_versions(self, filter_string):
        sha = filter_string.split("tags.checkpoint_sha256 = '")[1].rstrip("'")
        return [v for v in self.versions if v.tags.get("checkpoint_sha256") == sha]

    def create_model_version(self, name, source, run_id, tags, description):
        version = SimpleNamespace(version=len(self.versions) + 1, source=source, run_id=run_id, tags=dict(tags), description=description)
        self.versions.append(version)
        return version

    def log_inputs(self, run_id, datasets=None, models=None):
        self.inputs.append((run_id, datasets, models))

    def log_batch(self, run_id, metrics=()):
        self.metrics.extend(metrics)


def _params(**overrides):
    identity = {
        "checkpoint_sha256": H("c"),
        "checkpoint_role": "best",
        "lineage_sha256": H("1"),
        "dataset_id": H("d"),
        "dataset_lineage_sha256": H("2"),
        "parent_model_id": None,
        "stage_index": 1,
    }
    identity.update(overrides)
    return model_params(
        identity=identity,
        dataset_spec={"dataset": "kitscenes", "pack_params": {}},
        training_spec={"epochs": 10, "learning_rate": 3e-05},
        provenance={"image_digest": "sha256:x", "git_commit": "c"},
    )


def _ensure(client, params=None):
    return ensure_logged_model(
        client,
        experiment="1",
        source_run_id="run-1",
        name="best",
        params=params or _params(),
        description_text="desc",
        artifacts=lineage_artifacts(checkpoint_lineage=b"{}", datasets=[(H("d"), b"{}", b"PAR1")]),
    )


def test_flatten_is_deterministic_and_keeps_json_leaves():
    assert flatten("t", {"b": 1, "a": {"x": [1, 2]}}) == {"t.a.x": "[1, 2]", "t.b": "1"}
    with pytest.raises(ValueError, match="longer"):
        flatten("big", "x" * 7000)


def test_model_params_carry_identity_and_drop_unknown_parent():
    params = _params()
    assert params["checkpoint_sha256"] == H("c")
    assert "parent_model_id" not in params
    assert params["training.learning_rate"] == "3e-05"


def test_logged_model_is_created_once_with_artifacts_and_description():
    client = _Client()
    first = _ensure(client)
    assert first.status == "READY"
    assert first.tags[DESCRIPTION_TAG] == "desc"
    assert (first.model_id, "lineage/checkpoint.json") in client.artifacts
    assert (first.model_id, f"lineage/datasets/{H('d')}/units.parquet") in client.artifacts
    assert _ensure(client).model_id == first.model_id
    assert len(client.models) == 1


def test_same_weights_with_different_lineage_fail():
    client = _Client()
    _ensure(client)
    with pytest.raises(RuntimeError, match="different params"):
        _ensure(client, _params(lineage_sha256=H("9")))


def test_version_points_at_the_logged_model_and_is_reused():
    client = _Client()
    model = _ensure(client)
    assert ensure_version(client, model=model, run_id="run-1") == "1"
    assert ensure_version(client, model=model, run_id="run-1") == "1"
    version = client.versions[0]
    assert version.source == f"models:/{model.model_id}"
    assert version.tags["checkpoint_sha256"] == H("c")
    assert REGISTERED_MODEL in client.registered


def test_version_pointing_elsewhere_is_an_error():
    client = _Client()
    model = _ensure(client)
    ensure_version(client, model=model, run_id="run-1")
    client.versions[0].source = "s3://somewhere/else.pt"
    with pytest.raises(RuntimeError, match="points at"):
        ensure_version(client, model=model, run_id="run-1")


def _lineage():
    return {
        "dataset_id": H("d"),
        "published": {"name": "kitscenes-val", "version": "v3.5"},
        "raw": {"snapshot_manifest_sha256": H("3")},
        "labels": None,
        "pack": {"contracts": {"shard": "v13"}},
        "totals": {"samples": 200},
        "units": {"rows": 40},
    }


def test_dataset_entity_carries_profile_and_digest():
    dataset = mlflow_dataset(dataset_lineage=_lineage(), dataset_lineage_sha256=H("4"), index_uri="s3://b/by-id/x.json")
    assert dataset.name == "kitscenes-val/v3.5" and dataset.digest == H("d")
    profile = json.loads(dataset.profile)
    assert profile["dataset_lineage_sha256"] == H("4") and profile["label_set_sha256"] is None
    assert json.loads(dataset.source) == {"url": "s3://b/by-id/x.json"}


def test_evaluation_metrics_attach_to_the_model_and_dataset():
    client = _Client()
    model = _ensure(client)
    dataset = mlflow_dataset(dataset_lineage=_lineage(), dataset_lineage_sha256=H("4"), index_uri="s3://b/x")
    log_evaluation(
        client, run_id="eval-1", model_id=model.model_id, dataset=dataset,
        context="eval:kitscenes_benchmark", metrics={"ade_3s": 0.7, "fde_3s": 1.9},
        suite="kitscenes_benchmark", record_payload=b"{}", record_name=f"{H('d')}.json",
    )
    assert {(m.key, m.model_id, m.dataset_digest) for m in client.metrics} == {
        ("eval/kitscenes_benchmark/ade_3s", model.model_id, H("d")),
        ("eval/kitscenes_benchmark/fde_3s", model.model_id, H("d")),
    }
    run_id, datasets, models = client.inputs[0]
    assert run_id == "eval-1" and models[0].model_id == model.model_id
    assert datasets[0].tags[0].value == "eval:kitscenes_benchmark"
    assert (model.model_id, f"evaluations/kitscenes_benchmark/{H('d')}.json") in client.artifacts


def test_description_links_parent_or_states_unknowns():
    text = description(
        checkpoint={"role": "best", "sha256": H("c")},
        lineage={"stage_index": 2, "parent": {"model_id": "m-1", "checkpoint_sha256": H("9")}, "dataset": {"dataset_id": H("d"), "lineage_uri": "s3://x"}},
        tracking_url="https://mlflow", experiment="1",
    )
    assert "/#/experiments/1/models/m-1" in text and H("d") in text
    imported = description(
        checkpoint={"role": "imported", "sha256": H("c")},
        lineage={"stage_index": 0, "parent": None, "dataset": {"status": "unknown"}},
        tracking_url="https://mlflow", experiment="1",
    )
    assert "unknown" in imported
