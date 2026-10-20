"""Wiring contracts of the two registered workflows."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("flytekitplugins.ray")

MODULES = (
    "Platform.pipelines.autoe2e",
    "Platform.pipelines.autoe2e_acquire",
    "Platform.pipelines.autoe2e_tasks",
)


def _entities(kind):
    found = {}
    for name in MODULES:
        module = importlib.import_module(name)
        for attr in vars(module).values():
            if isinstance(attr, kind) and attr.name.startswith("Platform.pipelines.autoe2e"):
                found[attr.name] = attr
    return found


def test_only_two_workflows_are_registered():
    from flytekit.core.workflow import WorkflowBase

    assert sorted(_entities(WorkflowBase)) == [
        "Platform.pipelines.autoe2e.wf_autoe2e",
        "Platform.pipelines.autoe2e_acquire.wf_acquire_raw_snapshot",
    ]


def test_only_hub_acquisition_holds_the_hf_token():
    from flytekit.core.base_task import PythonTask

    def secrets(task):
        context = task.security_context
        return (context.secrets or []) if context is not None else []

    holders = sorted(
        name for name, task in _entities(PythonTask).items()
        if any(secret.group == "hf-token" for secret in secrets(task))
    )
    assert holders == ["Platform.pipelines.autoe2e_acquire.acquire_unit_from_hub"]


@pytest.mark.parametrize(
    ("task_name", "queue", "priority", "workers", "gpus"),
    [
        ("train_ray_1", "gpu-validation", "research-low", 1, 1),
        ("train_ray_2", "gpu-validation", "research-low", 2, 1),
        ("train_ray_4", "p5en-capacity-block", "production-high", 1, 4),
        ("train_ray_8", "p5en-capacity-block", "production-high", 1, 8),
    ],
)
def test_gpu_profile_picks_the_queue(task_name, queue, priority, workers, gpus):
    import Platform.pipelines.autoe2e as wf

    task = getattr(wf, task_name)
    labels = task.metadata.labels if hasattr(task.metadata, "labels") else task._labels
    assert labels["kueue.x-k8s.io/queue-name"] == queue
    assert labels["kueue.x-k8s.io/priority-class"] == priority
    group = task.task_config.worker_node_config[0]
    assert group.replicas == workers
    assert task.environment["AUTO_E2E_MLFLOW_EXPERIMENT"] == "auto-e2e"


def test_benchmark_evaluation_runs_on_a_validation_gpu():
    import Platform.pipelines.autoe2e as wf

    task = wf.evaluate_kitscenes_benchmark
    assert task.resources.requests.gpu == "1"


def test_pack_partition_is_cached_and_label_unit_is_not():
    import Platform.pipelines.autoe2e_tasks as tasks

    assert tasks.pack_partition.metadata.cache is True
    assert tasks.label_unit.metadata.cache is False


@pytest.mark.parametrize(("labels", "console"), [(True, True), (True, False), (False, True), (False, False)])
def test_build_dataset_compiles_for_every_label_and_console_combination(labels, console):
    """map_task turns unbound defaulted inputs into required lists; compile catches it."""
    from flytekit.configuration import ImageConfig, SerializationSettings
    from flytekit.core.context_manager import ExecutionState, FlyteContextManager

    import Platform.pipelines.autoe2e_tasks as tasks
    from Platform.pipelines.autoe2e_adapters import SourcePlan, UnitRef, current_pack_version
    from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, DatasetSpec, ManifestRef, ReasoningSpec
    from Platform.pipelines.autoe2e_store import DatasetHandle

    plan = SourcePlan(
        dataset=AutoE2EDataset.KITSCENES, upstream="u", revision="r", split="train",
        published_name="kitscenes", snapshot_manifest_sha256="e" * 64,
        units=[UnitRef(f"u{i}", f"k{i}", "1" * 64, "i", "2" * 64) for i in range(2)],
        partitions=[["u0"], ["u1"]], labels=labels, console=console,
    )
    spec = DatasetSpec(
        dataset=AutoE2EDataset.KITSCENES,
        source_manifest=ManifestRef("s3://d/m.json", "null", "e" * 64),
        pack_version=current_pack_version(AutoE2EDataset.KITSCENES), image_size=512,
        reasoning=ReasoningSpec("openai_compatible", "nvidia/Cosmos3-Nano", "p", 10) if labels else None,
    )
    handle = DatasetHandle("d" * 64, "kitscenes", "v9.9", False, "exec-1")
    settings = SerializationSettings(image_config=ImageConfig.auto_default_image())
    ctx = FlyteContextManager.current_context()
    state = ctx.execution_state.with_params(mode=ExecutionState.Mode.TASK_EXECUTION)
    with FlyteContextManager.with_context(ctx.with_serialization_settings(settings).with_execution_state(state)) as compiled:
        spec_out = tasks.build_dataset.compile_into_workflow(
            compiled, tasks.build_dataset._task_function, spec=spec, plan=plan, handle=handle,
            datasets_bucket="d", dynamo_table="t", label_concurrency=2, pack_concurrency=2,
        )
    assert spec_out.nodes
