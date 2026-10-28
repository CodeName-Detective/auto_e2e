"""METEOR on the 202-keyframe KITScenes val population, scored against ground truth from KIT poses.txt.

Keyframes (the rule proposed in auto_e2e issue #210): in every scene, frames 40 + 90k that still have
at least 3 s (30 frames) of future. On the 117 val scenes this gives 202 keyframes, 16 of which are
not in the AutoE2E v3.5 mirror (they have < 5 s of future). Ground truth for every keyframe is built
from the scene's poses.txt with the mirror builder's own formula (_local_future_to_ego_xy + pose_yaws),
so all 202 are scored the same way. METEOR is fed exactly as in run_meteor_kitscenes.py (build_feed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).parent))
from metrics import compute_metrics
from run_meteor_kitscenes import CAMS_ORDER, build_camera_extrinsics, build_feed, discover_default_scenes, interpolate_path

HISTORY, STRIDE, MIN_FUTURE, STEPS = 40, 90, 30, 30   # frames at 10 Hz; STEPS = 3 s scored horizon


def keyframes(poses: np.ndarray) -> list[int]:
    last = len(poses) - 1
    return [a for a in range(HISTORY, last + 1, STRIDE) if a + MIN_FUTURE <= last]


def gt_from_poses(poses: np.ndarray, a: int) -> np.ndarray:
    """Future positions 0.1 ... 3.0 s ahead in the car frame at frame a (x forward, y left)."""
    xy = poses[:, 1:3]
    yaw = Rotation.from_quat(poses[a, 4:8]).as_euler("ZYX")[0]
    d = xy[a + 1:a + 1 + STEPS] - xy[a]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", help="file with one scene id per line (default: every prepared scene)")
    parser.add_argument("--quarter-dir", required=True, help="reduced KITScenes scenes (prepare_kitscenes_images.py)")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="crop", choices=["crop", "pad", "stretch"])
    parser.add_argument("--narrow", default="front", choices=["front", "zero"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.scenes:
        scenes = [s.strip() for s in open(args.scenes) if s.strip()]
    else:
        scenes = discover_default_scenes(args.quarter_dir)

    if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21 only
        ort.preload_dlls()
    sess = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("providers:", sess.get_providers())

    ids, frames_left, v0s, gts, paths, scores = [], [], [], [], [], []
    t0 = time.time()
    for s in scenes:
        scene_dir = os.path.join(args.quarter_dir, s)
        poses = np.loadtxt(os.path.join(scene_dir, "poses.txt"))
        calib = json.load(open(os.path.join(scene_dir, "calibration/calib.json")))
        Ts = build_camera_extrinsics(calib, CAMS_ORDER)
        for a in keyframes(poses):
            v0 = float(np.linalg.norm(poses[a + 1, 1:3] - poses[a - 1, 1:3]) / (poses[a + 1, 0] - poses[a - 1, 0]))
            ego = sess.run(["ego"], build_feed(scene_dir, calib, Ts, a, v0, args.variant, args.narrow))[0][0]
            ids.append(f"kitscenes-v1-{s}-f{a:06d}")
            frames_left.append(len(poses) - 1 - a)
            v0s.append(v0)
            gts.append(gt_from_poses(poses, a))
            paths.append(ego[:36].reshape(3, 6, 2))
            scores.append(ego[36:39])
    n = len(ids)
    print(f"{n} keyframes from {len(scenes)} scenes in {time.time() - t0:.0f} s")

    gt = np.stack(gts).astype(np.float32)
    valid = np.ones((n, STEPS), dtype=bool)
    meteor_10hz = np.stack([[interpolate_path(p) for p in sample] for sample in paths])
    t = np.arange(1, STEPS + 1) * 0.1
    cv = np.zeros((n, 3, STEPS, 2), dtype=np.float32)
    cv[..., 0] = np.asarray(v0s)[:, None, None] * t
    cv_scores = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1))
    results = {
        "population": "keyframes 40 + 90k with >= 3 s future", "keyframes": n, "scenes": len(scenes),
        "ground_truth": "KIT poses.txt", "variant": args.variant, "narrow": args.narrow,
        "meteor": compute_metrics(meteor_10hz, np.stack(scores), gt, valid),
        "constant_velocity": compute_metrics(cv, cv_scores, gt, valid),
    }
    np.savez_compressed(os.path.join(args.out, "predictions_keyframes.npz"), sample_uid=np.array(ids),
                        frames_of_future=np.array(frames_left), v0=np.array(v0s, np.float32),
                        paths=np.stack(paths), scores=np.stack(scores), gt_pose=gt)
    with open(os.path.join(args.out, "metrics_keyframes.json"), "w") as f:
        json.dump(results, f, indent=2)
    for name in ("meteor", "constant_velocity"):
        top, best = results[name]["top1"], results[name]["best_of_3"]
        print(f"{name:18s} top1 ADE " + " / ".join(f"{top['ade'][h]:.3f}" for h in ("1s", "2s", "3s"))
              + "  FDE " + " / ".join(f"{top['fde'][h]:.3f}" for h in ("1s", "2s", "3s"))
              + "  | best-of-3 ADE " + " / ".join(f"{best['min_ade'][h]:.3f}" for h in ("1s", "2s", "3s")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
