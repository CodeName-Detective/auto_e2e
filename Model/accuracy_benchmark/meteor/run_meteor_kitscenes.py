"""METEOR adapter and benchmark runner for KITScenes val (117 scenes, 11,035 dense samples).

Scores the released METEOR ONNX model against the AutoE2E v3.5 mirror ground truth
(trajectory_xy_m) and against ground truth rebuilt from KIT poses.txt. Images come from
the original KITScenes-Multimodal scenes reduced by prepare_kitscenes_images.py.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))
from metrics import compute_metrics


CAMS_ORDER = [
    "camera_ring_front",        # Slot 0
    "camera_ring_front_left",   # Slot 1
    "camera_ring_front_right",  # Slot 2
    "camera_ring_rear",         # Slot 3
    "camera_ring_rear_left",    # Slot 4
    "camera_ring_rear_right",   # Slot 5
]


def discover_default_scenes(quarter_dir: str, count: Optional[int] = None) -> List[str]:
    """Find the first `count` (default: all) scene IDs in sorted order with PREP_DONE.json."""
    all_entries = sorted(os.listdir(quarter_dir))
    valid_scenes = []
    for s in all_entries:
        if s.startswith("."):
            continue
        scene_path = os.path.join(quarter_dir, s)
        if os.path.isdir(scene_path) and os.path.isfile(os.path.join(scene_path, "PREP_DONE.json")):
            valid_scenes.append(s)
    return valid_scenes[:count]


def map_scenes_to_shards(shard_dir: str) -> Dict[str, str]:
    """Map scene ID to shard tar path."""
    import glob
    tars = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    scene_to_tar = {}
    for t in tars:
        with tarfile.open(t) as tf:
            for name in tf.getnames()[:15]:
                m = re.match(r"kitscenes-v1-([0-9a-f\-]+)-f(\d+)", name)
                if m:
                    scene_to_tar[m.group(1)] = t
                    break
    return scene_to_tar


def load_scene_poses(poses_file: str) -> List[List[float]]:
    """Load TUM poses.txt."""
    with open(poses_file) as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]


def interpolate_path(path_6x2: np.ndarray) -> np.ndarray:
    """Linearly interpolate METEOR 6 waypoints (t=0.5..3.0s) to 30 steps at 10Hz (t=0.1..3.0s)."""
    t_meteor = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    pts = np.vstack([[0.0, 0.0], path_6x2])  # [7, 2]
    t_target = (np.arange(1, 31) * 0.1).astype(np.float32)  # [30]
    interp_x = np.interp(t_target, t_meteor, pts[:, 0])
    interp_y = np.interp(t_target, t_meteor, pts[:, 1])
    return np.stack([interp_x, interp_y], axis=-1).astype(np.float32)  # [30, 2]


def fit_image_and_intrinsics(
    bgr_quarter: np.ndarray,
    focal_length: float,
    pp_u: float,
    pp_v: float,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit quarter-res 876x568 image to 768x432 and adjust intrinsics K.

    Convention:
    Quarter-res:
        f_q = f / 4.0
        cx_q = (cx + 0.5) / 4.0 - 0.5
        cy_q = (cy + 0.5) / 4.0 - 0.5
    """
    f_q = focal_length / 4.0
    cx_q = (pp_u + 0.5) / 4.0 - 0.5
    cy_q = (pp_v + 0.5) / 4.0 - 0.5

    if variant == "crop":
        # Keep rows 37..529 (493 rows, centered 16:9), resize 876x493 -> 768x432
        cropped = bgr_quarter[37:530, :]
        resized = cv2.resize(cropped, (768, 432), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        sx = 768.0 / 876.0
        sy = 432.0 / 493.0
        fx = f_q * sx
        fy = f_q * sy
        cx_out = (cx_q + 0.5) * sx - 0.5
        cy_out = (cy_q - 37.0 + 0.5) * sy - 0.5

    elif variant == "pad":
        # Resize to 666x432 (s = 432/568), add 51 black columns left and right
        resized_666 = cv2.resize(bgr_quarter, (666, 432), interpolation=cv2.INTER_AREA)
        padded_bgr = np.zeros((432, 768, 3), dtype=np.uint8)
        padded_bgr[:, 51:51 + 666] = resized_666
        rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

        s = 432.0 / 568.0
        fx = f_q * s
        fy = f_q * s
        cx_out = (cx_q + 0.5) * s - 0.5 + 51.0
        cy_out = (cy_q + 0.5) * s - 0.5

    elif variant == "stretch":
        # Resize 876x568 -> 768x432
        resized = cv2.resize(bgr_quarter, (768, 432), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        sx = 768.0 / 876.0
        sy = 432.0 / 568.0
        fx = f_q * sx
        fy = f_q * sy
        cx_out = (cx_q + 0.5) * sx - 0.5
        cy_out = (cy_q + 0.5) * sy - 0.5

    else:
        raise ValueError(f"Unknown image fitting variant: {variant}")

    K = np.array([
        [fx, 0.0, cx_out],
        [0.0, fy, cy_out],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return rgb, K


# METEOR's CAM_FRONT_NARROW focal length at 768x432 (demo-scene manifest). KITScenes has no narrow
# camera, so the slot is synthesized from camera_base_front_center: crop the quarter-res image
# around its principal point to this focal length and resize to 768x432 (same optical centre and
# pose, narrower field of view). A black slot (METEOR's absent-camera convention) makes METEOR
# under-predict progress by ~10% on KITScenes (compare --narrow zero).
NARROW_SOURCE_CAM = "camera_base_front_center"
NARROW_FOCAL = 1415.0


def synthesize_narrow(bgr_quarter: np.ndarray, focal: float, pp_u: float, pp_v: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (RGB uint8 [432,768,3], K [3,3]) of a narrow view cropped from a quarter-res image."""
    f_q = focal / 4.0
    cx_q = (pp_u + 0.5) / 4.0 - 0.5
    cy_q = (pp_v + 0.5) / 4.0 - 0.5
    w, h = 768.0 * f_q / NARROW_FOCAL, 432.0 * f_q / NARROW_FOCAL
    x0, y0 = int(round(cx_q + 0.5 - w / 2)), int(round(cy_q + 0.5 - h / 2))
    wi, hi = int(round(w)), int(round(h))
    crop = bgr_quarter[y0:y0 + hi, x0:x0 + wi]
    if crop.shape[:2] != (hi, wi):
        raise ValueError(f"narrow crop {wi}x{hi} at ({x0},{y0}) leaves the {bgr_quarter.shape[1]}x{bgr_quarter.shape[0]} image")
    rgb = cv2.cvtColor(cv2.resize(crop, (768, 432), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    sx, sy = 768.0 / wi, 432.0 / hi
    K = np.array([[f_q * sx, 0.0, (cx_q - x0 + 0.5) * sx - 0.5],
                  [0.0, f_q * sy, (cy_q - y0 + 0.5) * sy - 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return rgb, K


def build_camera_extrinsics(
    calib_json: dict,
    cams: List[str],
) -> List[np.ndarray]:
    """Compute T_cam_ego = inv(T_ego_cam) where T_ego_cam = T_ego_base @ T_base_cam."""
    T_ego_base = np.eye(4, dtype=np.float32)
    T_ego_base[2, 3] = 2.1  # base_frame origin is 2.1 m above ground

    Ts = []
    for c in cams:
        T_base_cam = np.array(calib_json[c]["T_to_reference"], dtype=np.float32)
        T_ego_cam = T_ego_base @ T_base_cam
        T_cam_ego = np.linalg.inv(T_ego_cam).astype(np.float32)
        Ts.append(T_cam_ego)
    return Ts


def build_feed(scene_dir: str, calib: dict, Ts_list: List[np.ndarray], f_idx: int, v0: float,
               variant: str, narrow: str) -> Dict[str, np.ndarray]:
    """METEOR ONNX inputs for one frame of a reduced KITScenes scene (8 camera slots)."""
    imgs_list, Ks_list = [], []
    for c_name in CAMS_ORDER:
        bgr = cv2.imread(os.path.join(scene_dir, c_name, f"{f_idx:010d}.jpg"))
        intr = calib[c_name]["intrinsics"]
        rgb, K = fit_image_and_intrinsics(
            bgr, intr["focal_length"], intr["principal_point_u"], intr["principal_point_v"], variant
        )
        imgs_list.append(rgb.transpose(2, 0, 1))  # CHW
        Ks_list.append(K)

    # Slot 6 (front narrow): synthesized from camera_base_front_center, or black + donor
    if narrow == "front":
        intr = calib[NARROW_SOURCE_CAM]["intrinsics"]
        bgr = cv2.imread(os.path.join(scene_dir, NARROW_SOURCE_CAM, f"{f_idx:010d}.jpg"))
        rgb, K = synthesize_narrow(bgr, intr["focal_length"], intr["principal_point_u"], intr["principal_point_v"])
        imgs_list.append(rgb.transpose(2, 0, 1))
        Ks_list.append(K)
        slot6_T = build_camera_extrinsics(calib, [NARROW_SOURCE_CAM])[0]
    else:
        imgs_list.append(np.zeros((3, 432, 768), dtype=np.uint8))
        Ks_list.append(Ks_list[0].copy())
        slot6_T = Ts_list[0].copy()

    # Slot 7 (back narrow): absent -> black + pose donor (METEOR saw this in training)
    imgs_list.append(np.zeros((3, 432, 768), dtype=np.uint8))
    Ks_list.append(Ks_list[3].copy())
    slot7_T = Ts_list[3].copy()

    return {
        "imgs": np.stack(imgs_list)[None].astype(np.uint8),
        "K": np.stack(Ks_list)[None].astype(np.float32),
        "T_cam_ego": np.stack(list(Ts_list) + [slot6_T, slot7_T])[None].astype(np.float32),
        "v0": np.array([v0], dtype=np.float32),
    }


def generate_overlays(
    out_dir: str,
    samples_data: List[dict],
    predictions_crop: np.ndarray,      # [N, 3, 6, 2]
    scores_crop: np.ndarray,           # [N, 3]
    quarter_dir: str,
) -> None:
    """Generate 8 PNG overlays: 4 straight, 4 turning based on |GT y| at 3s."""
    os.makedirs(out_dir, exist_ok=True)
    N = len(samples_data)

    # Filter samples that have a valid 3s step (step 29)
    valid_3s_list: List[int] = []
    lat_list: List[float] = []
    for i in range(N):
        if samples_data[i]["gt_valid"][29]:
            valid_3s_list.append(i)
            lat_list.append(abs(samples_data[i]["gt_mirror"][29, 1]))

    valid_3s_indices = np.array(valid_3s_list)
    lat_displacements = np.array(lat_list)

    # Sort by lateral displacement
    sort_order = np.argsort(lat_displacements)
    straight_idx = valid_3s_indices[sort_order[:4]]
    turning_idx = valid_3s_indices[sort_order[-4:]]

    selected_cases = [
        ("straight", idx) for idx in straight_idx
    ] + [
        ("turning", idx) for idx in turning_idx
    ]

    t_grid = (np.arange(1, 31) * 0.1).astype(np.float32)

    for rank, (category, idx) in enumerate(selected_cases):
        s_info = samples_data[idx]
        s_id = s_info["scene_id"]
        f_idx = s_info["frame_idx"]
        uid = s_info["sample_uid"]

        # Load front image as fed to METEOR (crop variant)
        img_path = os.path.join(quarter_dir, s_id, "camera_ring_front", f"{f_idx:010d}.jpg")
        bgr = cv2.imread(img_path)
        with open(os.path.join(quarter_dir, s_id, "calibration/calib.json")) as f:
            calib = json.load(f)
        intr = calib["camera_ring_front"]["intrinsics"]
        rgb_fed, _ = fit_image_and_intrinsics(
            bgr, intr["focal_length"], intr["principal_point_u"], intr["principal_point_v"], "crop"
        )

        # Trajectories
        gt_m = s_info["gt_mirror"]
        val_m = s_info["gt_valid"]
        v0 = s_info["v0"]
        cv_traj = np.zeros((30, 2), dtype=np.float32)
        cv_traj[:, 0] = v0 * t_grid

        # 3 METEOR paths
        paths_3_6x2 = predictions_crop[idx]  # [3, 6, 2]
        scores_3 = scores_crop[idx]
        conf_3 = np.exp(scores_3) / np.exp(scores_3).sum()
        interp_3_paths = [interpolate_path(paths_3_6x2[m]) for m in range(3)]

        fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

        # Left: image
        ax_img.imshow(rgb_fed)
        ax_img.set_title(f"Front-Wide RGB ({category} sample, v0={v0:.2f} m/s)\n{uid}", fontsize=10)
        ax_img.axis("off")

        # Right: BEV plot (x forward pointing UP, y lateral pointing LEFT)
        # We plot lateral y on horizontal axis, forward x on vertical axis
        # Invert x-axis so that +y (left) is on the left side of the plot
        # Ground truth
        ax_bev.plot(gt_m[val_m > 0, 1], gt_m[val_m > 0, 0], color="black", linewidth=2.5, label="GT (Mirror)")

        # CV baseline
        ax_bev.plot(cv_traj[:, 1], cv_traj[:, 0], color="gray", linestyle="--", linewidth=1.8, label="CV Baseline")

        # 3 METEOR paths
        colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
        best_mode = int(np.argmax(conf_3))
        for m in range(3):
            is_best = (m == best_mode)
            lw = 2.5 if is_best else 1.5
            ls = "-" if is_best else "-."
            lbl = f"METEOR Mode {m} (p={conf_3[m]:.2f})" + (" [Top1]" if is_best else "")
            ax_bev.plot(interp_3_paths[m][:, 1], interp_3_paths[m][:, 0], color=colors[m], linewidth=lw, linestyle=ls, label=lbl)

        ax_bev.set_xlabel("y (left, m)", fontsize=11)
        ax_bev.set_ylabel("x (forward, m)", fontsize=11)
        ax_bev.set_title(f"BEV Trajectory ({category})", fontsize=11)
        ax_bev.grid(True, linestyle=":", alpha=0.6)
        ax_bev.legend(loc="upper left", fontsize=8)
        ax_bev.invert_xaxis()  # Positive y (left) to the left

        out_file = os.path.join(out_dir, f"overlay_{category}_{rank % 4}_{uid}.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
    print(f"Generated 8 overlays in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="METEOR KITScenes val benchmark runner")
    parser.add_argument("--scenes", default=None,
                        help="file with one scene id per line, or comma-separated ids (default: every prepared scene)")
    parser.add_argument("--variant", choices=["crop", "pad", "stretch", "cv"], required=True, help="Evaluation variant")
    parser.add_argument("--out", required=True, help="output dir")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--onnx", required=True, help="meteor_v157c3Z.onnx")
    parser.add_argument("--quarter-dir", required=True, help="reduced KITScenes scenes (prepare_kitscenes_images.py)")
    parser.add_argument("--shards-dir", required=True, help="AutoE2E kitscenes-val v3.5 shards directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--narrow", default="front", choices=["front", "zero"],
                        help="front: synthesize CAM_FRONT_NARROW from camera_base_front_center (back narrow stays black); "
                             "zero: both narrow slots black with donor K/pose (METEOR's absent-camera convention)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Resolve scenes
    if args.scenes is None:
        scenes = discover_default_scenes(args.quarter_dir)
    elif os.path.isfile(args.scenes):
        with open(args.scenes) as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print(f"Running variant '{args.variant}' on {len(scenes)} scenes:")
    for s in scenes:
        print(f"  {s}")

    # 2. Map shards and collect sample records
    scene_to_tar = map_scenes_to_shards(args.shards_dir)
    samples = []

    for s in scenes:
        poses_file = os.path.join(args.quarter_dir, s, "poses.txt")
        poses = load_scene_poses(poses_file)

        tar_path = scene_to_tar[s]
        with tarfile.open(tar_path) as tf:
            metas = sorted([n for n in tf.getnames() if n.endswith(".meta.json")])
            for m_name in metas:
                prefix = m_name[:-len(".meta.json")]
                f_idx = int(prefix.split("-f")[-1])

                # Extract ground truth from mirror
                f_traj = tf.extractfile(f"{prefix}.trajectory_xy.npz")
                z = np.load(io.BytesIO(f_traj.read()))
                gt_mirror = z["trajectory_xy_m"][:30].astype(np.float32)
                gt_valid = z["trajectory_valid"][:30].astype(np.uint8)

                # Compute v0 from poses.txt (central difference)
                p_prev = np.array(poses[f_idx - 1][1:3])
                t_prev = poses[f_idx - 1][0]
                p_next = np.array(poses[f_idx + 1][1:3])
                t_next = poses[f_idx + 1][0]
                v0 = float(np.linalg.norm(p_next - p_prev) / (t_next - t_prev))

                # Compute gt_pose from poses.txt
                p_f = np.array(poses[f_idx][1:3])
                yaw_f = Rotation.from_quat(poses[f_idx][4:8]).as_euler("ZYX")[0]
                c_yaw, s_yaw = np.cos(yaw_f), np.sin(yaw_f)
                R_inv = np.array([[c_yaw, s_yaw], [-s_yaw, c_yaw]])
                gt_pose = np.array([
                    R_inv @ (np.array(poses[f_idx + k][1:3]) - p_f) for k in range(1, 31)
                ], dtype=np.float32)

                samples.append({
                    "sample_uid": prefix,
                    "scene_id": s,
                    "frame_idx": f_idx,
                    "v0": v0,
                    "gt_mirror": gt_mirror,
                    "gt_valid": gt_valid,
                    "gt_pose": gt_pose,
                })

    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    N = len(samples)
    print(f"Total samples to process: {N}")

    # Prepare arrays
    sample_uids = np.array([s["sample_uid"] for s in samples])
    all_v0 = np.array([s["v0"] for s in samples], dtype=np.float32)
    all_gt_mirror = np.stack([s["gt_mirror"] for s in samples]).astype(np.float32)
    all_gt_valid = np.stack([s["gt_valid"] for s in samples]).astype(np.uint8)
    all_gt_pose = np.stack([s["gt_pose"] for s in samples]).astype(np.float32)

    t_grid = (np.arange(1, 31) * 0.1).astype(np.float32)

    if args.variant == "cv":
        t0 = time.time()
        # CV trajectory: x = v0 * t, y = 0
        cv_10hz = np.zeros((N, 30, 2), dtype=np.float32)
        cv_10hz[:, :, 0] = all_v0[:, None] * t_grid[None, :]

        # METEOR 6 waypoints at 0.5s intervals (t=0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        t_6wp = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
        cv_6wp = np.zeros((N, 6, 2), dtype=np.float32)
        cv_6wp[:, :, 0] = all_v0[:, None] * t_6wp[None, :]

        # Repeat across 3 modes
        pred_paths_6wp = np.repeat(cv_6wp[:, None, :, :], 3, axis=1)  # [N, 3, 6, 2]
        pred_scores = np.zeros((N, 3), dtype=np.float32)
        pred_scores[:, 0] = 1.0

        pred_paths_10hz = np.repeat(cv_10hz[:, None, :, :], 3, axis=1)  # [N, 3, 30, 2]
        total_time = time.time() - t0
        sec_per_frame = total_time / N

    else:
        # Load ONNX session
        if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21 only
            ort.preload_dlls()
        providers = ["CUDAExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
        print(f"Creating ONNX session with providers: {providers}")
        sess = ort.InferenceSession(args.onnx, providers=providers)

        # Cache calibration per scene
        scene_calibs = {}
        scene_Ts = {}
        for s in scenes:
            with open(os.path.join(args.quarter_dir, s, "calibration/calib.json")) as f:
                calib = json.load(f)
            scene_calibs[s] = calib
            scene_Ts[s] = build_camera_extrinsics(calib, CAMS_ORDER)

        pred_paths_6wp = np.zeros((N, 3, 6, 2), dtype=np.float32)
        pred_scores = np.zeros((N, 3), dtype=np.float32)
        pred_paths_10hz = np.zeros((N, 3, 30, 2), dtype=np.float32)

        print(f"Starting inference for variant '{args.variant}' on {N} samples...")
        t0 = time.time()

        for i, s_info in enumerate(samples):
            s = s_info["scene_id"]
            f_idx = s_info["frame_idx"]
            v0 = s_info["v0"]
            feed = build_feed(os.path.join(args.quarter_dir, s), scene_calibs[s], scene_Ts[s], f_idx, v0,
                              args.variant, args.narrow)
            outs = sess.run(["ego"], feed)
            ego = outs[0][0]  # [42]
            paths_3x6x2 = ego[:36].reshape(3, 6, 2)
            scores_3 = ego[36:39]

            pred_paths_6wp[i] = paths_3x6x2
            pred_scores[i] = scores_3

            # Interpolate to 10Hz
            for m in range(3):
                pred_paths_10hz[i, m] = interpolate_path(paths_3x6x2[m])

            if (i + 1) % 50 == 0 or (i + 1) == N:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                print(f"[{i + 1}/{N}] elapsed: {elapsed:.1f}s ({fps:.2f} fps, {1.0 / fps:.3f} s/frame)")

        total_time = time.time() - t0
        sec_per_frame = total_time / N
        print(f"Finished {N} samples in {total_time:.2f}s ({sec_per_frame:.4f} s/frame)")

    # 3. Compute Metrics
    print("Computing metrics against gt_mirror...")
    metrics_mirror = compute_metrics(pred_paths_10hz, pred_scores, all_gt_mirror, all_gt_valid)
    print("Computing metrics against gt_pose...")
    metrics_pose = compute_metrics(pred_paths_10hz, pred_scores, all_gt_pose, all_gt_valid)

    # 4. Save predictions npz
    pred_file = os.path.join(args.out, f"predictions_{args.variant}.npz")
    np.savez_compressed(
        pred_file,
        sample_uid=sample_uids,
        paths=pred_paths_6wp,
        scores=pred_scores,
        v0=all_v0,
        gt_mirror=all_gt_mirror,
        gt_valid=all_gt_valid,
        gt_pose=all_gt_pose,
    )
    print(f"Saved predictions to {pred_file}")

    # 5. Save metrics json
    metrics_summary = {
        "variant": args.variant,
        "narrow": args.narrow,
        "sample_count": N,
        "total_time_sec": total_time,
        "sec_per_frame": sec_per_frame,
        "metrics_mirror": metrics_mirror,
        "metrics_pose": metrics_pose,
    }
    metrics_file = os.path.join(args.out, f"metrics_{args.variant}.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # 6. Generate overlays if crop
    if args.variant == "crop":
        overlays_dir = os.path.join(args.out, "overlays")
        generate_overlays(overlays_dir, samples, pred_paths_6wp, pred_scores, args.quarter_dir)

    # Print summary
    print(f"\n===== RESULTS FOR {args.variant.upper()} =====")
    print("GT Mirror:")
    print(f"  Top1 ADE: 1s={metrics_mirror['top1']['ade']['1s']:.4f}m, 2s={metrics_mirror['top1']['ade']['2s']:.4f}m, 3s={metrics_mirror['top1']['ade']['3s']:.4f}m")
    print(f"  Top1 FDE: 1s={metrics_mirror['top1']['fde']['1s']:.4f}m, 2s={metrics_mirror['top1']['fde']['2s']:.4f}m, 3s={metrics_mirror['top1']['fde']['3s']:.4f}m")
    print(f"  Best3 minADE: 1s={metrics_mirror['best_of_3']['min_ade']['1s']:.4f}m, 2s={metrics_mirror['best_of_3']['min_ade']['2s']:.4f}m, 3s={metrics_mirror['best_of_3']['min_ade']['3s']:.4f}m")
    print(f"  Best3 minFDE: 1s={metrics_mirror['best_of_3']['min_fde']['1s']:.4f}m, 2s={metrics_mirror['best_of_3']['min_fde']['2s']:.4f}m, 3s={metrics_mirror['best_of_3']['min_fde']['3s']:.4f}m")
    print("GT Pose:")
    print(f"  Top1 ADE: 1s={metrics_pose['top1']['ade']['1s']:.4f}m, 2s={metrics_pose['top1']['ade']['2s']:.4f}m, 3s={metrics_pose['top1']['ade']['3s']:.4f}m")
    print(f"  Top1 FDE: 1s={metrics_pose['top1']['fde']['1s']:.4f}m, 2s={metrics_pose['top1']['fde']['2s']:.4f}m, 3s={metrics_pose['top1']['fde']['3s']:.4f}m")


if __name__ == "__main__":
    main()
