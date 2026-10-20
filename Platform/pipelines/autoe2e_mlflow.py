"""MLflow placement for wf_autoe2e: everything about a checkpoint on its LoggedModel.

Runs keep what MLflow and training record on their own. Each distinct
checkpoint becomes one LoggedModel whose params carry every lineage digest
(the LoggedModel page shows params, not tags), whose artifacts carry the
lineage and evaluation records, and whose metrics carry the evaluation
dataset. A registry version points at it with ``source="models:/<model_id>"``
because the OSS registry has no ``model_id`` column.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

EXPERIMENT = "auto-e2e"
REGISTERED_MODEL = "auto-e2e-driving-policy"
DESCRIPTION_TAG = "mlflow.note.content"
IDENTITY_KEYS = (
    "checkpoint_sha256",
    "checkpoint_role",
    "lineage_sha256",
    "dataset_id",
    "dataset_lineage_sha256",
    "parent_model_id",
    "stage_index",
)
_MAX_PARAM_VALUE = 6000


def mlflow_dataset(
    *,
    dataset_lineage: Mapping[str, Any],
    dataset_lineage_sha256: str,
    index_uri: str,
):
    """MLflow Dataset entity for a published dataset, identified by ``dataset_id``.

    Built as an entity rather than through ``mlflow.data`` because the
    skinny client has no ``mlflow.data`` and ``MetaDataset`` drops profiles.
    """
    from mlflow.entities import Dataset

    published = dataset_lineage["published"]
    labels = dataset_lineage.get("labels") or {}
    profile = {
        "dataset_lineage_sha256": dataset_lineage_sha256,
        "snapshot_manifest_sha256": dataset_lineage["raw"]["snapshot_manifest_sha256"],
        "label_set_sha256": labels.get("label_set_sha256"),
        "pack_contracts": dataset_lineage["pack"].get("contracts", {}),
        "samples": dataset_lineage["totals"]["samples"],
        "units": dataset_lineage["units"]["rows"],
    }
    return Dataset(
        name=f"{published['name']}/{published['version']}",
        digest=dataset_lineage["dataset_id"],
        source_type="http",
        source=json.dumps({"url": index_uri}),
        profile=json.dumps(profile, sort_keys=True),
    )


def log_dataset_input(client, *, run_id: str, dataset, context: str, model_id: Optional[str] = None) -> None:
    from mlflow.entities import DatasetInput, InputTag, LoggedModelInput

    client.log_inputs(
        run_id,
        datasets=[DatasetInput(dataset, [InputTag("mlflow.data.context", context)])],
        models=[LoggedModelInput(model_id)] if model_id else None,
    )


def experiment_id(client) -> str:
    existing = client.get_experiment_by_name(EXPERIMENT)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(EXPERIMENT)


def flatten(prefix: str, value: Any) -> dict[str, str]:
    """Flatten nested specs into MLflow params; values are canonical JSON leaves."""
    if isinstance(value, Mapping):
        out: dict[str, str] = {}
        for key in sorted(value):
            out.update(flatten(f"{prefix}.{key}" if prefix else str(key), value[key]))
        return out
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    if len(text) > _MAX_PARAM_VALUE:
        raise ValueError(f"param {prefix} is longer than {_MAX_PARAM_VALUE} characters")
    return {prefix: text}


def find_logged_model(client, experiment: str, checkpoint_sha256: str):
    """Return the one LoggedModel for a checkpoint, or None. Never two."""
    models = list(client.search_logged_models(
        experiment_ids=[experiment],
        filter_string=f"params.checkpoint_sha256 = '{checkpoint_sha256}'",
    ))
    if len(models) > 1:
        raise RuntimeError(f"{len(models)} LoggedModels claim checkpoint {checkpoint_sha256}")
    return models[0] if models else None


def description(
    *,
    checkpoint: Mapping[str, Any],
    lineage: Mapping[str, Any],
    tracking_url: str,
    experiment: str,
) -> str:
    """Generated Markdown so the model page links every upstream layer."""
    lines = [
        f"**{checkpoint['role']}** checkpoint `{checkpoint['sha256']}`, stage {lineage['stage_index']}.",
        "",
    ]
    parent = lineage.get("parent")
    if parent:
        lines.append(
            f"- Parent: [{parent['model_id']}]({tracking_url}/#/experiments/"
            f"{experiment}/models/{parent['model_id']}) (`{parent['checkpoint_sha256']}`)"
        )
    dataset = lineage.get("dataset", {})
    if dataset.get("status") == "unknown":
        lines.append("- Dataset, split, and training: unknown (imported checkpoint)")
    else:
        lines.append(f"- Dataset: `{dataset['dataset_id']}`, lineage {dataset['lineage_uri']}")
    lines.append("- Lineage and evaluation records: Artifacts tab, `lineage/` and `evaluations/`")
    return "\n".join(lines)


def ensure_logged_model(
    client,
    *,
    experiment: str,
    source_run_id: Optional[str],
    name: str,
    params: Mapping[str, str],
    description_text: str,
    artifacts: Mapping[str, bytes],
):
    """Create the LoggedModel for a checkpoint once; return the existing one after.

    An existing model must have exactly the same params, so a rerun can
    never attach different lineage to the same weights.
    """
    sha = params["checkpoint_sha256"]
    existing = find_logged_model(client, experiment, sha)
    if existing is not None:
        if dict(existing.params) != dict(params):
            raise RuntimeError(f"LoggedModel {existing.model_id} has different params for {sha}")
        return existing
    model = client.create_logged_model(
        experiment_id=experiment,
        name=name,
        source_run_id=source_run_id,
        params=dict(params),
        tags={DESCRIPTION_TAG: description_text},
    )
    with tempfile.TemporaryDirectory() as root:
        for relative, payload in artifacts.items():
            path = Path(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        client.log_model_artifacts(model.model_id, root)
    client.finalize_logged_model(model.model_id, "READY")
    return client.get_logged_model(model.model_id)


def ensure_version(client, *, model, run_id: Optional[str]) -> str:
    """Register a LoggedModel once; one version per checkpoint across executions."""
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(REGISTERED_MODEL)
    except MlflowException as error:
        if "RESOURCE_ALREADY_EXISTS" not in str(error):
            raise
    sha = model.params["checkpoint_sha256"]
    found = list(client.search_model_versions(
        f"name = '{REGISTERED_MODEL}' and tags.checkpoint_sha256 = '{sha}'"
    ))
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} versions claim checkpoint {sha}")
    source = f"models:/{model.model_id}"
    if found:
        if found[0].source != source:
            raise RuntimeError(f"version {found[0].version} for {sha} points at {found[0].source}")
        return str(found[0].version)
    tags = {key: model.params[key] for key in IDENTITY_KEYS if key in model.params}
    created = client.create_model_version(
        name=REGISTERED_MODEL,
        source=source,
        run_id=run_id,
        tags=tags,
        description=model.tags.get(DESCRIPTION_TAG, ""),
    )
    return str(created.version)


def log_evaluation(
    client,
    *,
    run_id: str,
    model_id: str,
    dataset,
    context: str,
    metrics: Mapping[str, float],
    suite: str,
    record_payload: bytes,
    record_name: str,
) -> None:
    """Attach evaluation metrics to the LoggedModel and keep the record with it."""
    import time

    from mlflow.entities import Metric

    log_dataset_input(client, run_id=run_id, dataset=dataset, context=context, model_id=model_id)
    timestamp = int(time.time() * 1000)
    client.log_batch(run_id, metrics=[
        Metric(
            key=f"eval/{suite}/{name}",
            value=float(value),
            timestamp=timestamp,
            step=0,
            model_id=model_id,
            dataset_name=dataset.name,
            dataset_digest=dataset.digest,
            run_id=run_id,
        )
        for name, value in sorted(metrics.items())
    ])
    with tempfile.TemporaryDirectory() as root:
        path = Path(root, "evaluations", suite, record_name)
        path.parent.mkdir(parents=True)
        path.write_bytes(record_payload)
        client.log_model_artifacts(model_id, root)


def model_params(
    *,
    identity: Mapping[str, Any],
    dataset_spec: Optional[Mapping[str, Any]],
    training_spec: Optional[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    params = flatten("", {k: v for k, v in identity.items() if v is not None})
    if dataset_spec is not None:
        params.update(flatten("dataset", dataset_spec))
    if training_spec is not None:
        params.update(flatten("training", training_spec))
    params.update(flatten("provenance", provenance))
    missing = [key for key in ("checkpoint_sha256", "checkpoint_role", "lineage_sha256") if key not in params]
    if missing:
        raise ValueError(f"LoggedModel params need {missing}")
    return params


def lineage_artifacts(
    *,
    checkpoint_lineage: bytes,
    datasets: Sequence[tuple[str, bytes, bytes]],
) -> dict[str, bytes]:
    """Artifact files for a LoggedModel: its lineage and each dataset lineage copy."""
    files = {"lineage/checkpoint.json": checkpoint_lineage}
    for dataset_id, dataset_json, units in datasets:
        files[f"lineage/datasets/{dataset_id}/dataset.json"] = dataset_json
        files[f"lineage/datasets/{dataset_id}/units.parquet"] = units
    return files
