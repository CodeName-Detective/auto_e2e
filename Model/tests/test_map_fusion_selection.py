"""``map_fusion_mode`` must reach the model and survive into the checkpoint.

The failure this guards against is silent: if ``train_il`` dropped the fusion
argument, ``AutoE2E`` would default to ``residual`` and an attention-fusion run
would train residual without raising or logging anything.
"""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("flytekit")

from model_components.map_encoder.map_bev_fusion import MAP_FUSION_REGISTRY
from Platform.pipelines import workflows


def _train_il_source() -> ast.Module:
    return ast.parse(inspect.getsource(workflows.train_il.task_function))


def _autoe2e_call(tree: ast.Module) -> ast.Call:
    """The AutoE2E(...) construction inside train_il."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AutoE2E"
        ):
            return node
    raise AssertionError("train_il no longer constructs AutoE2E directly")


class TestEnum:
    def test_every_value_is_a_registry_key(self):
        """A selectable value that the registry rejects fails at model build."""
        assert {m.value for m in workflows.MapFusion} == set(MAP_FUSION_REGISTRY)

    def test_default_is_residual(self):
        params = inspect.signature(workflows.train_il.task_function).parameters
        assert params["map_fusion_mode"].default is workflows.MapFusion.RESIDUAL


class TestReachesTheModel:
    def test_train_il_forwards_it_to_autoe2e(self):
        call = _autoe2e_call(_train_il_source())
        assert "map_fusion_mode" in {kw.arg for kw in call.keywords}, (
            "train_il builds AutoE2E without map_fusion_mode, so every run "
            "silently trains the constructor default"
        )

class TestSurvivesTheCheckpoint:
    def test_checkpoint_config_records_it(self):
        """Evaluation rebuilds from this dict; a missing key loads mismatched weights."""
        source = inspect.getsource(workflows.train_il.task_function)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "checkpoint_config" not in targets:
                continue
            assert isinstance(node.value, ast.Dict)
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            assert "map_fusion_mode" in keys
            return
        raise AssertionError("train_il no longer builds a checkpoint_config dict")

    @pytest.mark.parametrize("mode", sorted(MAP_FUSION_REGISTRY))
    def test_model_kwargs_preserves_it(self, mode):
        config = {
            "backbone": "swin_v2_tiny",
            "map_fusion_mode": mode,
            "embed_dim": 256,
            "num_views": 6,
            "is_pretrained": False,
            # A key the constructor does not accept; _model_kwargs must
            # drop it without taking map_fusion_mode with it.
            "fusion_mode": "bev",
        }
        kwargs = workflows._model_kwargs(config)
        assert kwargs["map_fusion_mode"] == mode
        assert "fusion_mode" not in kwargs
