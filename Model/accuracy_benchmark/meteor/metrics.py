"""Metrics calculation for METEOR benchmark on KITScenes val."""
from __future__ import annotations

from typing import Any
import numpy as np


def compute_sample_errors(
    pred_paths: np.ndarray,  # [N, T, 2]
    gt_paths: np.ndarray,    # [N, T, 2]
) -> np.ndarray:             # [N, T]
    """Compute per-timestep L2 displacement errors."""
    return np.linalg.norm(pred_paths - gt_paths, axis=-1)


def compute_metrics(
    paths: np.ndarray,       # [N, K, T, 2] or [N, T, 2]
    scores: np.ndarray | None,  # [N, K] or None
    gt: np.ndarray,          # [N, T, 2]
    valid: np.ndarray,       # [N, T] bool or uint8
    horizons: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute top1, best_of_3 metrics and diagnostics.

    Follows the ADE/FDE definition in Model/training/reactive_stage_runner.py.
    """
    if horizons is None:
        horizons = {"1s": 10, "2s": 20, "3s": 30}

    if paths.ndim == 3:
        # Single path per sample [N, T, 2]
        paths = paths[:, None, :, :]  # [N, 1, T, 2]
        scores = np.ones((paths.shape[0], 1), dtype=np.float32)

    N, K, T, _ = paths.shape
    valid_bool = valid.astype(bool)

    # 1. Top-1 path evaluation
    top1_indices = np.argmax(scores, axis=-1) if (scores is not None and K > 1) else np.zeros(N, dtype=np.int64)
    top1_paths = paths[np.arange(N), top1_indices]  # [N, T, 2]
    top1_errors = compute_sample_errors(top1_paths, gt)  # [N, T]

    top1_metrics: dict[str, Any] = {"ade": {}, "fde": {}, "ade_count": {}, "fde_count": {}}
    for h_name, h_step in horizons.items():
        usable = min(h_step, T)
        h_valid = valid_bool[:, :usable]
        valid_counts = h_valid.sum(axis=1)
        eligible_ade = valid_counts > 0

        if np.any(eligible_ade):
            sample_ade = np.sum(top1_errors[:, :usable] * h_valid, axis=1) / np.maximum(valid_counts, 1)
            top1_metrics["ade"][h_name] = float(np.mean(sample_ade[eligible_ade]))
            top1_metrics["ade_count"][h_name] = int(np.sum(eligible_ade))
        else:
            top1_metrics["ade"][h_name] = float("nan")
            top1_metrics["ade_count"][h_name] = 0

        if h_step <= T:
            step_idx = h_step - 1
            eligible_fde = valid_bool[:, step_idx]
            if np.any(eligible_fde):
                top1_metrics["fde"][h_name] = float(np.mean(top1_errors[eligible_fde, step_idx]))
                top1_metrics["fde_count"][h_name] = int(np.sum(eligible_fde))
            else:
                top1_metrics["fde"][h_name] = float("nan")
                top1_metrics["fde_count"][h_name] = 0
        else:
            top1_metrics["fde"][h_name] = float("nan")
            top1_metrics["fde_count"][h_name] = 0

    # 2. Best-of-K (minADE and minFDE)
    best_of_k_metrics: dict[str, Any] = {"min_ade": {}, "min_fde": {}, "ade_count": {}, "fde_count": {}}
    # Precompute errors for all modes [N, K, T]
    all_errors = np.stack([compute_sample_errors(paths[:, k], gt) for k in range(K)], axis=1)

    for h_name, h_step in horizons.items():
        usable = min(h_step, T)
        h_valid = valid_bool[:, :usable]
        valid_counts = h_valid.sum(axis=1)
        eligible_ade = valid_counts > 0

        if np.any(eligible_ade):
            # Per mode ADE for each sample: [N, K]
            # h_valid is [N, usable], expand to [N, 1, usable]
            h_valid_exp = h_valid[:, None, :]
            sample_mode_ade = np.sum(all_errors[:, :, :usable] * h_valid_exp, axis=-1) / np.maximum(valid_counts[:, None], 1)
            # Min over modes: [N]
            sample_min_ade = np.min(sample_mode_ade, axis=-1)
            best_of_k_metrics["min_ade"][h_name] = float(np.mean(sample_min_ade[eligible_ade]))
            best_of_k_metrics["ade_count"][h_name] = int(np.sum(eligible_ade))
        else:
            best_of_k_metrics["min_ade"][h_name] = float("nan")
            best_of_k_metrics["ade_count"][h_name] = 0

        if h_step <= T:
            step_idx = h_step - 1
            eligible_fde = valid_bool[:, step_idx]
            if np.any(eligible_fde):
                # errors at step_idx for each mode: [N, K]
                mode_fde = all_errors[:, :, step_idx]
                sample_min_fde = np.min(mode_fde, axis=-1)
                best_of_k_metrics["min_fde"][h_name] = float(np.mean(sample_min_fde[eligible_fde]))
                best_of_k_metrics["fde_count"][h_name] = int(np.sum(eligible_fde))
            else:
                best_of_k_metrics["min_fde"][h_name] = float("nan")
                best_of_k_metrics["fde_count"][h_name] = 0
        else:
            best_of_k_metrics["min_fde"][h_name] = float("nan")
            best_of_k_metrics["fde_count"][h_name] = 0

    # 3. Diagnostics for top1 at 3s
    # Step 30 (index 29) endpoint and overall 3s horizon
    disp = top1_paths[:, :30] - gt[:, :30]  # [N, 30, 2]
    dx = disp[..., 0]
    dy = disp[..., 1]

    # Endpoint (at step 29)
    step29_valid = valid_bool[:, 29] if T >= 30 else np.zeros(N, dtype=bool)
    if np.any(step29_valid):
        diag_endpoint = {
            "mean_signed_lon_3s": float(np.mean(dx[step29_valid, 29])),
            "mean_abs_lon_3s": float(np.mean(np.abs(dx[step29_valid, 29]))),
            "mean_signed_lat_3s": float(np.mean(dy[step29_valid, 29])),
            "mean_abs_lat_3s": float(np.mean(np.abs(dy[step29_valid, 29]))),
        }
    else:
        diag_endpoint = {
            "mean_signed_lon_3s": float("nan"),
            "mean_abs_lon_3s": float("nan"),
            "mean_signed_lat_3s": float("nan"),
            "mean_abs_lat_3s": float("nan"),
        }

    # Cumulative over valid steps up to 30 (reactive_stage_runner style)
    valid_30 = valid_bool[:, :min(30, T)]
    if np.any(valid_30):
        diag_cumulative = {
            "cum_mean_signed_lon_3s": float(np.mean(dx[:, :min(30, T)][valid_30])),
            "cum_mean_abs_lon_3s": float(np.mean(np.abs(dx[:, :min(30, T)][valid_30]))),
            "cum_mean_signed_lat_3s": float(np.mean(dy[:, :min(30, T)][valid_30])),
            "cum_mean_abs_lat_3s": float(np.mean(np.abs(dy[:, :min(30, T)][valid_30]))),
        }
    else:
        diag_cumulative = {
            "cum_mean_signed_lon_3s": float("nan"),
            "cum_mean_abs_lon_3s": float("nan"),
            "cum_mean_signed_lat_3s": float("nan"),
            "cum_mean_abs_lat_3s": float("nan"),
        }

    return {
        "top1": top1_metrics,
        "best_of_3": best_of_k_metrics,
        "diagnostics_3s": {**diag_endpoint, **diag_cumulative},
        "sample_count": N,
    }
