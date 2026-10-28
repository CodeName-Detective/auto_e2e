"""Stream KITScenes-Multimodal scenes from Hugging Face into a reduced local copy.

Used by the METEOR benchmark (auto_e2e issue #210). Each scene tar is read
straight from the network and never saved:

* lidar, radar and the rectified stereo cameras are skipped;
* the 7 cameras METEOR can use are shrunk by exactly 4x (INTER_AREA) and
  re-encoded as JPEG, so intrinsics scale by 0.25;
* every other file (calibration, poses, GNSS, maps, processed/) is kept as is.

The tar's SHA-256 and size are checked against the published values while
streaming. A dropped connection resumes at the current byte offset with an
HTTP Range request; a checksum mismatch discards that scene and retries.
Finished scenes get ``PREP_DONE.json`` and are skipped on the next run, so the
script can be stopped and restarted at any time.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import urllib3
from huggingface_hub import HfApi, get_token
from huggingface_hub.hf_api import RepoFile

REPO_ID = "KIT-MRT/KITScenes-Multimodal"
# Revision the AutoE2E kitscenes-val v3.5 mirror was built from.
DEFAULT_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
CAMERAS = (
    "camera_ring_front",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
    "camera_base_front_center",
)
SKIP_PREFIXES = ("lidar_", "radar_", "camera_base_front_left_rect", "camera_base_front_right_rect")
SCALE_DIVISOR = 4
JPEG_QUALITY = 95
SCRIPT_VERSION = 2
MAX_RECONNECTS = 30
CONNECTION_ERRORS = (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, OSError)

log = logging.getLogger("prep")


class ResumableHashingReader:
    """File-like HTTP stream that hashes and counts every byte it returns.

    When the connection drops, it reopens the URL from the current byte offset,
    so already-streamed data is never fetched twice. Only bytes actually
    returned are counted, so a partial read lost in an exception is re-fetched.
    """

    def __init__(self, url: str, token: str, size: int, label: str) -> None:
        self._url = url
        self._token = token
        self._size = size
        self._label = label
        self._response: requests.Response | None = None
        self.sha256 = hashlib.sha256()
        self.nbytes = 0
        self.reconnects = 0
        self._open()

    def _open(self) -> None:
        headers = {"Authorization": f"Bearer {self._token}"}
        if self.nbytes:
            headers["Range"] = f"bytes={self.nbytes}-"
        self._response = requests.get(self._url, headers=headers, stream=True, timeout=(30, 60))
        self._response.raise_for_status()
        if self.nbytes and self._response.status_code != 206:
            raise ValueError(f"server ignored the Range request (HTTP {self._response.status_code})")

    def _reconnect(self, reason: str) -> None:
        if self.reconnects >= MAX_RECONNECTS:
            raise ConnectionError(f"gave up after {MAX_RECONNECTS} reconnects: {reason}")
        self.reconnects += 1
        log.warning("%s: reconnect %d at %.2f GB (%s)", self._label, self.reconnects, self.nbytes / 1e9, reason)
        self.close()
        time.sleep(min(60, 5 * self.reconnects))
        self._open()

    def read(self, size: int = -1) -> bytes:
        while True:
            assert self._response is not None  # set by _open()
            try:
                data = self._response.raw.read(size)
            except CONNECTION_ERRORS as exc:
                self._reconnect(repr(exc)[:200])
                continue
            if not data and self.nbytes < self._size:
                self._reconnect("stream ended early")
                continue
            self.sha256.update(data)
            self.nbytes += len(data)
            return data

    def close(self) -> None:
        if self._response is not None:
            self._response.close()

    def drain(self) -> None:
        while self.read(1 << 20):
            pass


def shrink_jpeg(data: bytes, dest: Path) -> tuple[int, int]:
    """Decode, shrink by SCALE_DIVISOR, re-encode; return the original (w, h)."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {dest.name}")
    h, w = image.shape[:2]
    if w % SCALE_DIVISOR or h % SCALE_DIVISOR:
        raise ValueError(f"{dest}: {w}x{h} is not divisible by {SCALE_DIVISOR}")
    small = cv2.resize(image, (w // SCALE_DIVISOR, h // SCALE_DIVISOR), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"cannot encode {dest.name}")
    dest.write_bytes(encoded.tobytes())
    return w, h


def process_scene(scene_id: str, expected: dict, args: argparse.Namespace, token: str) -> dict:
    """Stream one scene into <out>/.partial/<id>, verify it, then move it to <out>/<id>."""
    partial = args.out / ".partial" / scene_id
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{args.revision}/data/{args.split}/{scene_id}.tar"
    start = time.time()
    camera_sizes: dict[str, list[int]] = {}
    counts = {"kept_files": 0, "shrunk_images": 0, "skipped_files": 0, "skipped_bytes": 0}
    in_flight = threading.BoundedSemaphore(args.workers * 4)
    futures: list[cf.Future] = []

    def submit_shrink(data: bytes, dest: Path, camera: str) -> None:
        in_flight.acquire()

        def job() -> None:
            try:
                size = shrink_jpeg(data, dest)
                camera_sizes.setdefault(camera, list(size))
            finally:
                in_flight.release()

        futures.append(pool.submit(job))

    reader = ResumableHashingReader(url, token, expected["size"], scene_id[:8])
    try:
        with cf.ThreadPoolExecutor(args.workers) as pool, tarfile.open(fileobj=reader, mode="r|") as tar:  # type: ignore[call-overload]  # stream mode only reads
            for member in tar:
                parts = Path(member.name).parts
                if not parts or parts[0] != scene_id:
                    raise ValueError(f"unexpected tar member {member.name}")
                rel = Path(*parts[1:]) if len(parts) > 1 else Path()
                top = rel.parts[0] if rel.parts else ""
                if top.startswith(SKIP_PREFIXES):
                    counts["skipped_files"] += member.isfile()
                    counts["skipped_bytes"] += member.size
                    continue
                dest = partial / rel
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported tar member type: {member.name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = tar.extractfile(member).read()
                if top in CAMERAS and dest.suffix.lower() == ".jpg":
                    submit_shrink(data, dest, top)
                    counts["shrunk_images"] += 1
                else:
                    dest.write_bytes(data)
                    counts["kept_files"] += 1
            while reader.nbytes < expected["size"] and reader.read(1 << 20):
                pass
            for future in futures:
                future.result()
    finally:
        reader.close()

    digest = reader.sha256.hexdigest()
    if digest != expected["sha256"] or reader.nbytes != expected["size"]:
        raise ValueError(
            f"checksum mismatch: got {digest} / {reader.nbytes} B, "
            f"expected {expected['sha256']} / {expected['size']} B"
        )
    missing = [c for c in CAMERAS if c not in camera_sizes]
    if missing:
        raise ValueError(f"cameras missing from tar: {missing}")
    seconds = time.time() - start
    record = {
        "scene_id": scene_id,
        "split": args.split,
        "repo_id": REPO_ID,
        "revision": args.revision,
        "tar_sha256": digest,
        "tar_bytes": reader.nbytes,
        "reconnects": reader.reconnects,
        "seconds": round(seconds, 1),
        "mb_per_s": round(reader.nbytes / seconds / 1e6, 1),
        "scale_divisor": SCALE_DIVISOR,
        "jpeg_quality": JPEG_QUALITY,
        "original_image_size_wh": camera_sizes,
        "script_version": SCRIPT_VERSION,
        **counts,
    }
    (partial / "PREP_DONE.json").write_text(json.dumps(record, indent=2))
    final = args.out / scene_id
    shutil.rmtree(final, ignore_errors=True)
    os.replace(partial, final)
    return record


def published_tars(split: str, revision: str) -> dict[str, dict]:
    """Map scene id -> {sha256, size} for every tar in data/<split>/ at this revision."""
    tars = {}
    for entry in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=f"data/{split}", revision=revision):
        if isinstance(entry, RepoFile) and entry.path.endswith(".tar"):
            if entry.lfs is None:
                raise ValueError(f"{entry.path} has no LFS checksum")
            scene_id = Path(entry.path).stem
            tars[scene_id] = {"sha256": entry.lfs.sha256, "size": entry.lfs.size}
    return tars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output root, e.g. data/kitscenes-val-quarter")
    parser.add_argument("--split", default="val")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--scenes", nargs="*", help="only these scene ids (default: whole split)")
    parser.add_argument("--limit", type=int, help="process at most this many pending scenes")
    parser.add_argument("--jobs", type=int, default=2, help="scenes streamed in parallel")
    parser.add_argument("--workers", type=int, default=4, help="image-shrinking threads per scene")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "prep.log")],
    )
    token = get_token()
    if not token:
        parser.error("no Hugging Face token: run `hf auth login` or set HF_TOKEN (KITScenes is gated)")
    shutil.rmtree(args.out / ".partial", ignore_errors=True)

    tars = published_tars(args.split, args.revision)
    wanted = args.scenes or sorted(tars)
    unknown = [s for s in wanted if s not in tars]
    if unknown:
        parser.error(f"not in data/{args.split}/ at {args.revision[:8]}: {unknown}")
    pending = [s for s in wanted if not (args.out / s / "PREP_DONE.json").exists()]
    already_done = len(wanted) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    total_gb = sum(tars[s]["size"] for s in pending) / 1e9
    log.info("split=%s revision=%s: %d wanted, %d already done, %d to stream now (%.1f GB)",
             args.split, args.revision[:8], len(wanted), already_done, len(pending), total_gb)

    failed: dict[str, str] = {}

    def run(scene_id: str) -> None:
        for attempt in range(1, args.retries + 1):
            try:
                rec = process_scene(scene_id, tars[scene_id], args, token)
                log.info("DONE %s  %.1f GB in %.0f s (%.1f MB/s), %d images shrunk, %d files kept",
                         scene_id, rec["tar_bytes"] / 1e9, rec["seconds"], rec["mb_per_s"],
                         rec["shrunk_images"], rec["kept_files"])
                return
            except Exception as exc:  # noqa: BLE001 - retry any failure, report the last one
                wait = 300 if "429" in str(exc) else 30 * attempt
                log.warning("attempt %d/%d failed for %s: %s", attempt, args.retries, scene_id, exc)
                if attempt < args.retries:
                    time.sleep(wait)
                else:
                    failed[scene_id] = repr(exc)
                    shutil.rmtree(args.out / ".partial" / scene_id, ignore_errors=True)

    with cf.ThreadPoolExecutor(args.jobs) as scenes_pool:
        list(scenes_pool.map(run, pending))

    done = sorted(p.parent.name for p in args.out.glob("*/PREP_DONE.json"))
    (args.out / "prep_manifest.json").write_text(json.dumps({
        "repo_id": REPO_ID, "split": args.split, "revision": args.revision,
        "cameras": CAMERAS, "skipped_prefixes": SKIP_PREFIXES,
        "scale_divisor": SCALE_DIVISOR, "jpeg_quality": JPEG_QUALITY,
        "scenes_done": done, "scenes_failed": failed,
    }, indent=2))
    log.info("FINISHED: %d scenes done in total, %d failed this run%s",
             len(done), len(failed), f": {sorted(failed)}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
