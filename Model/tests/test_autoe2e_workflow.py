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
