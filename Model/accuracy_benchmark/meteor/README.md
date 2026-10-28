# METEOR evaluation

Evaluate the released [METEOR](https://github.com/tier4/METEOR) v1.0 ONNX model on the
117 scenes of the official KITScenes-Multimodal `val` split, with ADE/FDE at 1, 2 and 3 s
(METEOR predicts 3 s).

METEOR gets the six ring cameras from the original KIT images, shrunk 4× and center-cropped
to 768×432, a front-narrow view cropped from `camera_base_front_center`, and the current
speed. It gets no map, route or motion history. It outputs 3 paths of 6 waypoints at 0.5 s,
which are linearly interpolated to 10 Hz; the scores use the top-confidence path.

## Setup

Accept the [KITScenes access terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal),
then run from the repository root:

```bash
BENCH=~/meteor-bench
python -m venv $BENCH/venv
source $BENCH/venv/bin/activate
pip install -r Model/accuracy_benchmark/meteor/requirements.txt
hf auth login
hf download AutowareFoundation/meteor meteor_v157c3Z.onnx SHA256SUMS --revision v1.0 --local-dir $BENCH/meteor
```

## Data

```bash
python Model/accuracy_benchmark/meteor/prepare_kitscenes_images.py --out $BENCH/kitscenes-val-quarter --jobs 4
```

This streams the 117 scene tars (about 508 GB) from dataset revision `6fde0034` without
saving them, and keeps about 40 GB: the 7 cameras shrunk 4×, plus calibration and poses.
Checksums are checked while streaming. Run the command again to resume.

The dense run also needs the AutoE2E v3.5 mirror shards (about 15 GB), which define the
11,035 samples:

```bash
hf download AutowareFoundation/auto_e2e --revision 7262c151599c1b5e4a99f0da9c3196ff9ff615fe --include "datasets/kitscenes-val/v3.5/shards/*" --local-dir $BENCH/hf-auto_e2e
```

## Run

```bash
python Model/accuracy_benchmark/meteor/run_keyframes.py --quarter-dir $BENCH/kitscenes-val-quarter --onnx $BENCH/meteor/meteor_v157c3Z.onnx --out $BENCH/results/keyframes
python Model/accuracy_benchmark/meteor/run_meteor_kitscenes.py --variant crop --quarter-dir $BENCH/kitscenes-val-quarter --shards-dir $BENCH/hf-auto_e2e/datasets/kitscenes-val/v3.5/shards --onnx $BENCH/meteor/meteor_v157c3Z.onnx --out $BENCH/results/dense
```

- `run_keyframes.py` scores the 202 keyframes (frame 40 + 90k with at least 3 s of future)
  against ground truth from `poses.txt`. It takes a few minutes.
- `run_meteor_kitscenes.py` scores all 11,035 samples against both the mirror's
  `trajectory_xy_m` and `poses.txt`. It takes about 2 h on the RTX 4050.
  `--variant cv` gives the constant-velocity baseline; `--narrow zero` leaves the
  front-narrow slot black.

Each run writes its predictions (`.npz`) and metrics (`.json`) to `--out`. Use `--scenes`
for a subset.

## Expected results

Top-1 path against `poses.txt` ground truth, ADE / FDE in m:

| Population | 1 s | 2 s | 3 s |
|---|---|---|---|
| 202 keyframes | 0.219 / 0.484 | 0.630 / 1.574 | 1.240 / 3.282 |
| 11,035 samples | 0.221 / 0.482 | 0.621 / 1.536 | 1.211 / 3.187 |
