# Drive-JEPA: KITScenes evaluation

Evaluate the official **perception-based ViT-L NAVSIM v2 checkpoint with 20 m
route commands**, without fine-tuning. This directory contains one script,
one dependency file and this README; input data is generated locally.

## Setup (Windows / CUDA)

Tested on Python 3.11, PyTorch 2.0.1/cu117 and an RTX 3070 (8 GB). Use an
isolated environment; the released model requires legacy MMCV CUDA operators.
From the AutoE2E repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-drive-jepa
.\.venv-drive-jepa\Scripts\Activate.ps1
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r Model/accuracy_benchmark/drive_jepa/requirements.txt
$env:PYTHONPATH = "$PWD\Model"
$env:PYTHONUTF8 = "1"
hf auth login
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --setup
```

Accept the [KITScenes access terms][dataset] before downloading data.
`--setup` fetches pinned Drive-JEPA, KITScenes SDK and nuPlan sources plus the
official `drive_jepa_perception_based_agent_vitl_v2.ckpt` under `.benchmark/`.
Revisions and mandatory checksums are recorded in the evaluator; existing
source checkouts with a different revision are rejected rather than reset.

## Generate 20 m commands

Commands are not bundled. First obtain **v3.5 scene-navigation exports** at
`<navigation-root>/<scene_uuid>/scene_navigation.json` for all 117 val scenes.
These come from AutoE2E's [Lanelet2 navigation preprocessing][navigation].
Raw KITScenes and the v3.5 mirror do **not** ship these vector-route files;
reconstructing them is a separate Linux/Lanelet2 step, outside this lean script.
The published 100 m labels alone cannot supply 20 m commands.

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --generate-commands --navigation-root D:/v3.5_navigation
```

This reads pinned mirror metadata, verifies every exported route's published
hash, and applies the 20 m centreline / ±2 m lateral rule. It saves
`.benchmark/route_inputs_20m/commands.json` and its provenance. The final
checksum must match the reference evaluation; no future targets are read.
Mirror shards are cached locally. An existing command/provenance pair can
instead be supplied with `--command-file path/to/commands.json`.

## Evaluate

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes
# Optional subset smoke check:
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --scene 00fc1b9b-005e-554c-7733-c4c0adeca015 --output .benchmark/results/smoke
```

The default streams raw scenes one at a time. The full split transfers about
**473 GiB**; allow roughly **35 GiB staging space**, plus weights/HF caches.
For existing data, pass `--raw-root D:/data/kitscenes` with SDK layout
`data/val/<uuid>/`. User-supplied data is never removed; its archive origin is
recorded as unverified. `--staging-root` and `--output` change storage paths.

Outputs under `.benchmark/results/drive_jepa_v2_20m/`: `run.json`, per-scene
NPZs with IDs/timestamps/predictions/targets, and `report.json`. Re-run the
same command to resume completed scenes. Changed run identities require a
new output directory. Explicit `--scene` runs are labelled subsets.

## Population, inputs and results

The evaluation matches **AutoE2E v3.5's 117 val scenes / 11,035 samples**,
excluding the mirror's 23 `overlap_train_val` scenes. Full runs must match the
[published identity report][reference]: sorted UID SHA256, terminal LF,
`00f824202c97b47ec74561d247e8e30686e5505eef2ad2bc0acba97e41ce9925`.

Original KITScenes RGB at current time and 0.5 s earlier uses native crop28 →
512x256 preprocessing and actual front-camera calibration. Ego features use
only past/current poses; ground truth uses future recorded ego poses. The
model generates 32 trajectories and selects one by learned score, each with
8 waypoints spanning 4 s. ADE/FDE use 10 Hz interpolation from the ego origin,
with equal weight per sample. No extrapolation is used.

Previously completed, identity-verified results (**ADE / FDE**, metres):

| 1 s | 2 s | 3 s | 4 s | 5 s |
|---:|---:|---:|---:|---:|
| 0.715 / 1.345 | 1.455 / 2.917 | 2.266 / 4.774 | 3.177 / 6.880 | N/A |

Labels use offline map-matched v3.5 route geometry, not future trajectory
targets. Unknown routes use a distinct one-hot category (3,450 samples).
Route provenance and checksums are recorded alongside locally generated commands.

This is a cross-dataset adaptation: KITScenes ego origin and native front
projection conventions are retained, without an invented rear-axle offset.
V2 uses learned-score selection without optional sequential simulator comfort
calibration. These are **open-loop KITScenes ADE/FDE**, not NAVSIM PDMS/EPDMS.
The trained forward computation and strict checkpoint loading are preserved.

[dataset]: https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal
[reference]: https://huggingface.co/AutowareFoundation/auto_e2e/blob/7262c151599c1b5e4a99f0da9c3196ff9ff615fe/evaluations/overlay-replay-v1/kitscenes-val/kitscenes-epoch-5/report.json
[navigation]: ../../data_parsing/kit_scenes/navigation.py
