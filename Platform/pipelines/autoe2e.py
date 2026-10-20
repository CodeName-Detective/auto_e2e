"""wf_autoe2e: one pipeline from a raw snapshot to traceable MLflow models.

1. plan and resolve the training dataset (and each evaluation dataset);
2. build and publish any dataset that is not published yet;
3. resolve the parent model, if any;
4. train with Ray Train on the profile's Kueue queue (skipped when epochs=0);
5. register one MLflow LoggedModel and version per distinct checkpoint;
6. evaluate every suite and attach the metrics to the LoggedModels.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from flytekit import Resources, dynamic, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from Platform.pipelines.autoe2e_adapters import SourcePlan
from Platform.pipelines.autoe2e_dataset import DatasetSpec, canonical_dataset_spec
from Platform.pipelines.autoe2e_store import DatasetHandle
from Platform.pipelines.autoe2e_tasks import build_dataset, plan_dataset, resolve_dataset
from Platform.pipelines.distributed_training import (
    MLFLOW_URI,
    RAY_1,
    RAY_2,
    RAY_8,
    RAY_REACTIVE_4,
    RAY_TASK_ENVIRONMENT,
    TRAINING_IMAGE,
    _bev_evaluation_pod_template,
)

EXPERIMENT_ENV = {"AUTO_E2E_MLFLOW_EXPERIMENT": "auto-e2e", "MLFLOW_TRACKING_URI": MLFLOW_URI}
TRAIN_ENVIRONMENT = {**RAY_TASK_ENVIRONMENT, **EXPERIMENT_ENV}
CAPACITY_BLOCK_NAME = "auto-e2e-p5en-capacity-block"


class GpuProfile(enum.Enum):
    RAY_1 = "ray_1"  # one validation GPU; smoke only
    RAY_2 = "ray_2"
    RAY_4 = "ray_4"
    RAY_8 = "ray_8"


class EvalSuite(enum.Enum):
    INTERNAL_VAL = "internal_val"
    KITSCENES_BENCHMARK = "kitscenes_benchmark"


@dataclass
class TrainingSpec:
    gpu_profile: GpuProfile = GpuProfile.RAY_8
    stage: str = "kitscenes_finetune"
    epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    per_rank_batch_size: int = 2
    seed: int = 149
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    steps_per_epoch: int = 0
    checkpoint_interval_steps: int = 256
    shuffle_buffer: int = 256
    num_loader_workers: int = 4
    backbone: str = "res_net_50"
    is_pretrained: bool = True
    freeze_bevformer: bool = True
    training_scope: str = "multitask"
    trajectory_weight: float = 1.0
    bev_weight: float = 0.0
    route_weight: float = 1.0
    corridor_pos_weight: float = 1.0
    bev_encoder_learning_rate: float = 1e-5
    bev_repeat_frequency_threshold: float = 0.05
    bev_min_positive_samples: int = 20
    bev_min_positive_cells: int = 2000
    validation_sample_limit: int = 1024
    parent_profile: str = ""


@dataclass
class EvalTarget:
    suite: EvalSuite
    dataset: Optional[DatasetSpec] = None


@dataclass
class EvaluationSpec:
    targets: List[EvalTarget] = field(default_factory=list)


@dataclass
class ModelRef:
    """Exactly one field is set: a registry version ("name/version") or an S3 checkpoint."""

    model_version: str = ""
    checkpoint_uri: str = ""


@dataclass
class RuntimeOptions:
    datasets_bucket: str = ""
    checkpoints_bucket: str = ""
    dynamo_table: str = "auto-e2e-console"
    label_concurrency: int = 5
    pack_concurrency: int = 60
    tracking_url: str = ""


@dataclass
class ParentHandle:
    present: bool = False
    checkpoint_uri: str = ""
    checkpoint_sha256: str = ""
    model_id: str = ""
    model_version: str = ""
    lineage_sha256: str = ""
    stage_index: int = 0
    epoch: int = 0


@dataclass
class TrainedCheckpoints:
    trained: bool
    run_id: str = ""
    best_uri: str = ""
    best_sha256: str = ""
    best_epoch: int = 0
    final_uri: str = ""
    final_sha256: str = ""
    final_epoch: int = 0
    best_metrics_json: str = "{}"
    final_metrics_json: str = "{}"


@dataclass
class RegisteredModel:
    role: str
    model_id: str
    version: str
    checkpoint_uri: str
    checkpoint_sha256: str
    lineage_sha256: str


@dataclass
class PipelineResult:
    dataset_id: str
    dataset_lineage_sha256: str
    models: List[RegisteredModel] = field(default_factory=list)


def _client():
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_URI))


def _s3():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))


def _execution_id() -> str:
    from flytekit import current_context

    try:
        return current_context().execution_id.name or "local"
    except Exception:
        return "local"


# --- datasets -----------------------------------------------------------------


@dynamic(container_image=TRAINING_IMAGE)
def ensure_dataset(
    spec: DatasetSpec,
    plan: SourcePlan,
    handle: DatasetHandle,
    runtime: RuntimeOptions,
) -> DatasetHandle:
    """Return a published dataset: the resolved one, or a freshly built one."""
    if handle.published:
        return handle
    return build_dataset(
        spec=spec, plan=plan, handle=handle,
        datasets_bucket=runtime.datasets_bucket, dynamo_table=runtime.dynamo_table,
        label_concurrency=runtime.label_concurrency, pack_concurrency=runtime.pack_concurrency,
    )


@dynamic(container_image=TRAINING_IMAGE)
def ensure_evaluation_datasets(evaluation: EvaluationSpec, runtime: RuntimeOptions) -> List[DatasetHandle]:
    handles = []
    for target in evaluation.targets:
        if target.dataset is None:
            continue
        plan = plan_dataset(spec=target.dataset, datasets_bucket=runtime.datasets_bucket)
        resolved = resolve_dataset(spec=target.dataset, plan=plan, datasets_bucket=runtime.datasets_bucket)
        handles.append(ensure_dataset(spec=target.dataset, plan=plan, handle=resolved, runtime=runtime))
    return handles


# --- parent ---------------------------------------------------------------------


def checkpoint_sha256(s3, uri: str) -> str:
    import hashlib

    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(8 * 1024**2), b""):
        digest.update(chunk)
    return digest.hexdigest()


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=EXPERIMENT_ENV, retries=2)
def resolve_parent(parent: Optional[ModelRef]) -> ParentHandle:
    """Pin the parent checkpoint by digest, from a registry version or a bare URI."""
    from Platform.pipelines.autoe2e_mlflow import REGISTERED_MODEL

    if parent is None or (not parent.model_version and not parent.checkpoint_uri):
        return ParentHandle()
    if parent.model_version and parent.checkpoint_uri:
        raise ValueError("ModelRef sets both model_version and checkpoint_uri")
    s3 = _s3()
    if parent.checkpoint_uri:
        return ParentHandle(
            present=True, checkpoint_uri=parent.checkpoint_uri,
            checkpoint_sha256=checkpoint_sha256(s3, parent.checkpoint_uri),
        )
    name, _, version = parent.model_version.rpartition("/")
    if name != REGISTERED_MODEL or not version:
        raise ValueError(f"model_version must be {REGISTERED_MODEL}/<version>")
    client = _client()
    registered = client.get_model_version(name, version)
    if not registered.source.startswith("models:/m-"):
        raise ValueError(f"version {version} is not a wf_autoe2e LoggedModel ({registered.source})")
    model = client.get_logged_model(registered.source.removeprefix("models:/"))
    params = model.params
    actual = checkpoint_sha256(s3, params["checkpoint_uri"])
    if actual != params["checkpoint_sha256"]:
        raise ValueError(f"parent checkpoint bytes differ from version {version}")
    return ParentHandle(
        present=True, checkpoint_uri=params["checkpoint_uri"], checkpoint_sha256=actual,
        model_id=model.model_id, model_version=parent.model_version,
        lineage_sha256=params["lineage_sha256"], stage_index=int(params["stage_index"]),
        epoch=int(params.get("checkpoint_epoch", "0")),
    )


# --- training -------------------------------------------------------------------


def active_capacity_block_end() -> str:
    """End time of the one active p5en Capacity Block, read by the task itself."""
    import boto3

    reservations = boto3.client("ec2").describe_capacity_reservations(Filters=[
        {"Name": "tag:Name", "Values": [CAPACITY_BLOCK_NAME]},
        {"Name": "state", "Values": ["active"]},
    ])["CapacityReservations"]
    blocks = [r for r in reservations if r.get("ReservationType") == "capacity-block"]
    if len(blocks) != 1:
        raise RuntimeError(f"expected one active {CAPACITY_BLOCK_NAME}, found {len(blocks)}")
    return blocks[0]["EndDate"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _train(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle, num_workers: int) -> TrainedCheckpoints:
    from Platform.pipelines.distributed_training import BEV_POS_WEIGHT_CAP, _run_reactive_stage_task

    output = _run_reactive_stage_task(
        shards=[FlyteDirectory(uri) for uri in dataset.partition_uris],
        stage=spec.stage,
        num_workers=num_workers,
        parent_checkpoint=FlyteFile(parent.checkpoint_uri) if parent.present else None,
        resume_checkpoint=None,
        backbone=spec.backbone,
        epochs=spec.epochs,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip=spec.grad_clip,
        val_fraction=spec.val_fraction,
        num_loader_workers=spec.num_loader_workers,
        per_rank_batch_size=spec.per_rank_batch_size,
        training_seed=spec.seed,
        precision=spec.precision,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        steps_per_epoch=spec.steps_per_epoch,
        checkpoint_interval_steps=spec.checkpoint_interval_steps,
        shuffle_buffer=spec.shuffle_buffer,
        is_pretrained=spec.is_pretrained,
        trajectory_weight=spec.trajectory_weight,
        bev_weight=spec.bev_weight,
        route_weight=spec.route_weight,
        corridor_pos_weight=spec.corridor_pos_weight,
        bev_pos_weight_cap=BEV_POS_WEIGHT_CAP,
        bev_repeat_frequency_threshold=spec.bev_repeat_frequency_threshold,
        bev_max_repeat=4,
        bev_min_positive_samples=spec.bev_min_positive_samples,
        bev_min_positive_cells=spec.bev_min_positive_cells,
        freeze_bevformer=spec.freeze_bevformer,
        validation_sample_limit=spec.validation_sample_limit,
        capacity_block_end_utc=active_capacity_block_end() if num_workers > 2 else "",
        training_scope=spec.training_scope,
        bev_encoder_learning_rate=spec.bev_encoder_learning_rate,
        allow_single_worker_smoke=num_workers == 1,
        parent_profile=spec.parent_profile,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        parent_checkpoint_epoch=parent.epoch,
    )
    metadata = json.loads(open(output.metadata.download()).read())
    return TrainedCheckpoints(
        trained=True,
        run_id=metadata["tracking"]["mlflow_run_id"],
        best_uri=output.checkpoint_uri,
        best_sha256=output.checkpoint_sha256,
        best_epoch=int(metadata["selected_epoch"]),
        final_uri=output.final_checkpoint_uri,
        final_sha256=output.final_checkpoint_sha256,
        final_epoch=int(metadata["final_epoch"]),
        best_metrics_json=json.dumps(metadata["metrics"], sort_keys=True, default=str),
        final_metrics_json=json.dumps(metadata["final_metrics"], sort_keys=True, default=str),
    )


_QUEUE = {
    "gpu-validation": {"kueue.x-k8s.io/queue-name": "gpu-validation", "kueue.x-k8s.io/priority-class": "research-low"},
    "p5en": {"kueue.x-k8s.io/queue-name": "p5en-capacity-block", "kueue.x-k8s.io/priority-class": "production-high"},
}


@task(task_config=RAY_1, container_image=TRAINING_IMAGE, labels=_QUEUE["gpu-validation"], environment=TRAIN_ENVIRONMENT, retries=1)
def train_ray_1(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle) -> TrainedCheckpoints:
    return _train(spec, dataset, parent, 1)


@task(task_config=RAY_2, container_image=TRAINING_IMAGE, labels=_QUEUE["gpu-validation"], environment=TRAIN_ENVIRONMENT, retries=1)
def train_ray_2(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle) -> TrainedCheckpoints:
    return _train(spec, dataset, parent, 2)


@task(task_config=RAY_REACTIVE_4, container_image=TRAINING_IMAGE, labels=_QUEUE["p5en"], environment=TRAIN_ENVIRONMENT, retries=2)
def train_ray_4(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle) -> TrainedCheckpoints:
    return _train(spec, dataset, parent, 4)


@task(task_config=RAY_8, container_image=TRAINING_IMAGE, labels=_QUEUE["p5en"], environment=TRAIN_ENVIRONMENT, retries=1)
def train_ray_8(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle) -> TrainedCheckpoints:
    return _train(spec, dataset, parent, 8)


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="1", mem="1Gi"))
def parent_as_checkpoints(parent: ParentHandle) -> TrainedCheckpoints:
    """epochs=0: evaluate the parent as both best and final, without training."""
    if not parent.present:
        raise ValueError("epochs=0 evaluates a parent model; set parent")
    return TrainedCheckpoints(
        trained=False, best_uri=parent.checkpoint_uri, best_sha256=parent.checkpoint_sha256,
        best_epoch=parent.epoch, final_uri=parent.checkpoint_uri,
        final_sha256=parent.checkpoint_sha256, final_epoch=parent.epoch,
    )


@dynamic(container_image=TRAINING_IMAGE)
def train(spec: TrainingSpec, dataset: DatasetHandle, parent: ParentHandle) -> TrainedCheckpoints:
    """Pick the Ray task for the GPU profile; the profile is the only queue choice."""
    if spec.epochs == 0:
        return parent_as_checkpoints(parent=parent)
    task_for = {
        GpuProfile.RAY_1: train_ray_1,
        GpuProfile.RAY_2: train_ray_2,
        GpuProfile.RAY_4: train_ray_4,
        GpuProfile.RAY_8: train_ray_8,
    }
    return task_for[spec.gpu_profile](spec=spec, dataset=dataset, parent=parent)


# --- registration ---------------------------------------------------------------


def store_checkpoint(s3, *, source_uri: str, sha256: str, bucket: str) -> str:
    """Copy a checkpoint to a content-addressed, write-once key and verify it."""
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    key = f"autoe2e/checkpoints/{sha256}.pt"
    target = f"s3://{bucket}/{key}"
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        source_bucket, source_key = _split_s3_uri(source_uri)
        s3.copy({"Bucket": source_bucket, "Key": source_key}, bucket, key)
    if checkpoint_sha256(s3, target) != sha256:
        raise RuntimeError(f"stored checkpoint differs from {sha256}: {target}")
    return target


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=EXPERIMENT_ENV, retries=2)
def register_models(
    dataset_spec: DatasetSpec,
    dataset: DatasetHandle,
    training: TrainingSpec,
    parent: ParentHandle,
    trained: TrainedCheckpoints,
    runtime: RuntimeOptions,
) -> List[RegisteredModel]:
    """One LoggedModel and version per distinct checkpoint, with full lineage."""
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import checkpoint_lineage, record_bytes, record_sha256
    from Platform.pipelines.autoe2e_store import read_dataset_lineage

    client = _client()
    s3 = _s3()
    experiment = am.experiment_id(client)
    if parent.present and not parent.model_id:
        # A bare checkpoint is imported first, so every parent has a model page
        # and a lineage digest for its children to point at.
        imported = _register_imported(client, s3, experiment, parent, runtime)
        parent = ParentHandle(
            present=True, checkpoint_uri=imported.checkpoint_uri,
            checkpoint_sha256=imported.checkpoint_sha256, model_id=imported.model_id,
            model_version=f"{am.REGISTERED_MODEL}/{imported.version}",
            lineage_sha256=imported.lineage_sha256, stage_index=0, epoch=parent.epoch,
        )
    if not trained.trained:
        return [RegisteredModel("parent", parent.model_id, parent.model_version.rpartition("/")[2],
                                parent.checkpoint_uri, parent.checkpoint_sha256, parent.lineage_sha256)]
    lineage = read_dataset_lineage(
        s3, bucket=runtime.datasets_bucket, key=dataset.lineage_key, sha256=dataset.lineage_sha256,
    )
    dataset_json = record_bytes(lineage)
    units_bucket_key = lineage["units"]["uri"].removeprefix(f"s3://{runtime.datasets_bucket}/")
    units_payload = s3.get_object(Bucket=runtime.datasets_bucket, Key=units_bucket_key)["Body"].read()
    roles = [("best", trained.best_uri, trained.best_sha256, trained.best_epoch, trained.best_metrics_json)]
    if trained.final_sha256 != trained.best_sha256:
        roles.append(("final", trained.final_uri, trained.final_sha256, trained.final_epoch, trained.final_metrics_json))
    results = []
    for role, uri, sha, epoch, metrics_json in roles:
        stored = store_checkpoint(s3, source_uri=uri, sha256=sha, bucket=runtime.checkpoints_bucket)
        record = checkpoint_lineage(
            checkpoint={"sha256": sha, "uri": stored, "role": role, "epoch": epoch},
            parent=None if not parent.present else {
                "model_id": parent.model_id, "model_version": parent.model_version,
                "checkpoint_sha256": parent.checkpoint_sha256,
                "lineage_sha256": parent.lineage_sha256, "stage_index": parent.stage_index,
            },
            dataset={"dataset_id": dataset.dataset_id, "lineage_uri": f"s3://{runtime.datasets_bucket}/{dataset.lineage_key}",
                     "lineage_sha256": dataset.lineage_sha256},
            split={"policy": "stage_default", "val_fraction": training.val_fraction},
            training={"spec": _spec_dict(training), "flyte_execution_id": _execution_id(),
                      "mlflow_run_id": trained.run_id, "image_digest": TRAINING_IMAGE,
                      "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
            selection={"metrics": json.loads(metrics_json)},
        )
        lineage_sha = record_sha256(record)
        params = am.model_params(
            identity={
                "checkpoint_sha256": sha, "checkpoint_role": role, "checkpoint_uri": stored,
                "checkpoint_epoch": epoch, "lineage_sha256": lineage_sha,
                "dataset_id": dataset.dataset_id, "dataset_lineage_sha256": dataset.lineage_sha256,
                "parent_model_id": parent.model_id or None, "stage_index": record["stage_index"],
            },
            dataset_spec=canonical_dataset_spec(dataset_spec),
            training_spec=_spec_dict(training),
            provenance={"flyte_execution_id": _execution_id(), "training_image": TRAINING_IMAGE,
                        "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
        )
        model = am.ensure_logged_model(
            client, experiment=experiment, source_run_id=trained.run_id, name=role, params=params,
            description_text=am.description(
                checkpoint={"role": role, "sha256": sha}, lineage=record,
                tracking_url=runtime.tracking_url, experiment=experiment,
            ),
            artifacts=am.lineage_artifacts(
                checkpoint_lineage=record_bytes(record),
                datasets=[(dataset.dataset_id, dataset_json, units_payload)],
            ),
        )
        version = am.ensure_version(client, model=model, run_id=trained.run_id)
        results.append(RegisteredModel(role, model.model_id, version, stored, sha, lineage_sha))
    training_dataset = am.mlflow_dataset(
        dataset_lineage=lineage, dataset_lineage_sha256=dataset.lineage_sha256,
        index_uri=f"s3://{runtime.datasets_bucket}/{dataset.published_name}/by-id/{dataset.dataset_id}.json",
    )
    am.log_dataset_input(client, run_id=trained.run_id, dataset=training_dataset, context="training",
                         model_id=parent.model_id or None)
    from mlflow.entities import LoggedModelOutput

    client.log_outputs(trained.run_id, [LoggedModelOutput(m.model_id, 0) for m in results])
    return results


def _spec_dict(spec: TrainingSpec) -> dict:
    payload = asdict(spec)
    payload["gpu_profile"] = spec.gpu_profile.value
    return payload


def _register_imported(client, s3, experiment, parent: ParentHandle, runtime: RuntimeOptions) -> RegisteredModel:
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import checkpoint_lineage, record_bytes, record_sha256

    stored = store_checkpoint(s3, source_uri=parent.checkpoint_uri, sha256=parent.checkpoint_sha256,
                              bucket=runtime.checkpoints_bucket)
    record = checkpoint_lineage(
        checkpoint={"sha256": parent.checkpoint_sha256, "uri": stored, "role": "imported", "epoch": parent.epoch},
        parent=None, dataset=None, split=None, training=None, selection=None,
    )
    lineage_sha = record_sha256(record)
    params = am.model_params(
        identity={"checkpoint_sha256": parent.checkpoint_sha256, "checkpoint_role": "imported",
                  "checkpoint_uri": stored, "checkpoint_epoch": parent.epoch,
                  "lineage_sha256": lineage_sha, "stage_index": 0},
        dataset_spec=None, training_spec=None,
        provenance={"imported_from": parent.checkpoint_uri, "flyte_execution_id": _execution_id()},
    )
    model = am.ensure_logged_model(
        client, experiment=experiment, source_run_id=None, name="imported", params=params,
        description_text=am.description(checkpoint={"role": "imported", "sha256": parent.checkpoint_sha256},
                                        lineage=record, tracking_url=runtime.tracking_url, experiment=experiment),
        artifacts=am.lineage_artifacts(checkpoint_lineage=record_bytes(record), datasets=[]),
    )
    version = am.ensure_version(client, model=model, run_id=None)
    return RegisteredModel("imported", model.model_id, version, stored, parent.checkpoint_sha256, lineage_sha)


# --- evaluation -----------------------------------------------------------------


def numeric_leaves(prefix: str, value) -> dict:
    out = {}
    if isinstance(value, dict):
        for key in sorted(value):
            out.update(numeric_leaves(f"{prefix}.{key}" if prefix else key, value[key]))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        import math

        if math.isfinite(value):
            out[prefix] = float(value)
    return out


def _evaluation_run(client, experiment: str, name: str) -> str:
    return client.create_run(experiment, run_name=name, tags={"flyte_execution_id": _execution_id()}).info.run_id


def _log(client, s3, runtime, *, run_id, model: RegisteredModel, dataset: DatasetHandle, suite: str,
         metrics: dict, protocol: dict, per_sample_uri: str, per_sample_sha: str) -> None:
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.autoe2e_lineage import evaluation_record, record_bytes
    from Platform.pipelines.autoe2e_store import read_dataset_lineage

    lineage = read_dataset_lineage(s3, bucket=runtime.datasets_bucket, key=dataset.lineage_key, sha256=dataset.lineage_sha256)
    record = evaluation_record(
        checkpoint_sha256=model.checkpoint_sha256, checkpoint_lineage_sha256=model.lineage_sha256,
        suite=suite,
        dataset={"dataset_id": dataset.dataset_id, "lineage_sha256": dataset.lineage_sha256,
                 "published": f"{dataset.published_name}/{dataset.version}"},
        protocol=protocol, metrics=metrics,
        per_sample={"uri": per_sample_uri, "sha256": per_sample_sha},
        code={"image_digest": TRAINING_IMAGE, "git_commit": os.environ.get("AUTO_E2E_GIT_COMMIT", "unknown")},
        flyte_execution_id=_execution_id(), finished_at=datetime.now(timezone.utc).isoformat(),
    )
    mlflow_dataset = am.mlflow_dataset(
        dataset_lineage=lineage, dataset_lineage_sha256=dataset.lineage_sha256,
        index_uri=f"s3://{runtime.datasets_bucket}/{dataset.published_name}/by-id/{dataset.dataset_id}.json",
    )
    suite_name = f"{suite}@{dataset.published_name}-{dataset.version}"
    am.log_evaluation(
        client, run_id=run_id, model_id=model.model_id, dataset=mlflow_dataset,
        context=f"eval:{suite}", metrics=metrics, suite=suite_name,
        record_payload=record_bytes(record), record_name=f"{dataset.dataset_id}.json",
    )


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="2", mem="8Gi"), environment=EXPERIMENT_ENV, retries=2)
def log_internal_validation(
    models: List[RegisteredModel],
    trained: TrainedCheckpoints,
    dataset: DatasetHandle,
    runtime: RuntimeOptions,
) -> None:
    """Attach the training run's validation metrics to the matching models."""
    from Platform.pipelines import autoe2e_mlflow as am

    if not trained.trained:
        return
    client = _client()
    s3 = _s3()
    run_id = _evaluation_run(client, am.experiment_id(client), f"internal-val-{_execution_id()}")
    for model in models:
        source = trained.best_metrics_json if model.role == "best" else trained.final_metrics_json
        metrics = numeric_leaves("", json.loads(source))
        _log(client, s3, runtime, run_id=run_id, model=model, dataset=dataset, suite="internal_val",
             metrics=metrics, protocol={"name": "ray_validation_holdout", "source": "training_metrics"},
             per_sample_uri="", per_sample_sha=model.lineage_sha256)
    client.set_terminated(run_id)


@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="6", mem="28Gi", gpu="1", ephemeral_storage="420Gi"),
    limits=Resources(cpu="6", mem="28Gi", gpu="1", ephemeral_storage="420Gi"),
    labels=_QUEUE["gpu-validation"],
    pod_template=_bev_evaluation_pod_template(),
    environment=EXPERIMENT_ENV,
    retries=1,
)
def evaluate_kitscenes_benchmark(
    models: List[RegisteredModel],
    dataset: DatasetHandle,
    runtime: RuntimeOptions,
) -> None:
    """Re-run every model on a KITScenes val or test publication."""
    from Platform.pipelines import autoe2e_mlflow as am
    from Platform.pipelines.distributed_training import evaluate_reactive_kitscenes_checkpoint
    from Platform.pipelines.dataset_publication import sha256_bytes

    split = {"kitscenes-val": "val", "kitscenes-test": "test"}.get(dataset.published_name)
    if split is None:
        raise ValueError(f"{dataset.published_name} is not a KITScenes benchmark publication")
    client = _client()
    s3 = _s3()
    run_id = _evaluation_run(client, am.experiment_id(client), f"kitscenes-benchmark-{_execution_id()}")
    for model in models:
        output = evaluate_reactive_kitscenes_checkpoint.task_function(
            checkpoint=FlyteFile(model.checkpoint_uri),
            shards=[FlyteDirectory(uri) for uri in dataset.partition_uris],
            source_split=split,
        )
        report_bytes = open(output.report.download(), "rb").read()
        if sha256_bytes(report_bytes) != output.report_sha256 or output.checkpoint_sha256 != model.checkpoint_sha256:
            raise RuntimeError("benchmark report does not match its checkpoint")
        report = json.loads(report_bytes)
        key = f"autoe2e/evaluations/{model.checkpoint_sha256}/kitscenes_benchmark/{dataset.dataset_id}.json"
        s3.put_object(Bucket=runtime.checkpoints_bucket, Key=key, Body=report_bytes)
        _log(client, s3, runtime, run_id=run_id, model=model, dataset=dataset, suite="kitscenes_benchmark",
             metrics=numeric_leaves("", report["metrics"]),
             protocol={"name": report["schema_version"], "evaluation_role": report["evaluation_role"]},
             per_sample_uri=f"s3://{runtime.checkpoints_bucket}/{key}", per_sample_sha=output.report_sha256)
    client.set_terminated(run_id)


@dynamic(container_image=TRAINING_IMAGE)
def evaluate(
    models: List[RegisteredModel],
    evaluation: EvaluationSpec,
    evaluation_datasets: List[DatasetHandle],
    trained: TrainedCheckpoints,
    dataset: DatasetHandle,
    runtime: RuntimeOptions,
) -> None:
    datasets = iter(evaluation_datasets)
    for target in evaluation.targets:
        if target.suite is EvalSuite.INTERNAL_VAL:
            if target.dataset is not None:
                raise ValueError("internal_val uses the training dataset; leave its dataset empty")
            log_internal_validation(models=models, trained=trained, dataset=dataset, runtime=runtime)
        elif target.suite is EvalSuite.KITSCENES_BENCHMARK:
            if target.dataset is None:
                raise ValueError("kitscenes_benchmark needs a val or test dataset")
            evaluate_kitscenes_benchmark(models=models, dataset=next(datasets), runtime=runtime)


@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="1", mem="1Gi"))
def pipeline_result(dataset: DatasetHandle, models: List[RegisteredModel]) -> PipelineResult:
    return PipelineResult(dataset.dataset_id, dataset.lineage_sha256, models)


@workflow
def wf_autoe2e(
    dataset: DatasetSpec,
    training: TrainingSpec,
    evaluation: EvaluationSpec,
    runtime: RuntimeOptions,
    parent: Optional[ModelRef] = None,
) -> PipelineResult:
    """Build or reuse a dataset, train, register, and evaluate, all traceable in MLflow."""
    plan = plan_dataset(spec=dataset, datasets_bucket=runtime.datasets_bucket)
    resolved = resolve_dataset(spec=dataset, plan=plan, datasets_bucket=runtime.datasets_bucket)
    published = ensure_dataset(spec=dataset, plan=plan, handle=resolved, runtime=runtime)
    evaluation_datasets = ensure_evaluation_datasets(evaluation=evaluation, runtime=runtime)
    parent_handle = resolve_parent(parent=parent)
    trained = train(spec=training, dataset=published, parent=parent_handle)
    models = register_models(
        dataset_spec=dataset, dataset=published, training=training,
        parent=parent_handle, trained=trained, runtime=runtime,
    )
    done = evaluate(
        models=models, evaluation=evaluation, evaluation_datasets=evaluation_datasets,
        trained=trained, dataset=published, runtime=runtime,
    )
    result = pipeline_result(dataset=published, models=models)
    done >> result
    return result
