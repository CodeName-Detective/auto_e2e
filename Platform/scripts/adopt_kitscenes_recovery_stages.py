#!/usr/bin/env python3
"""Adopt the July KITScenes raw ingest and Cosmos labels into the stage store.

Full Run ``a5nk56zsbcg7g6npjfpt`` already downloaded and labeled 533 scenes.
Its frozen recovery manifest lists each scene's ``raw_uri`` and
``label_uri``. This script records those artifacts, without copying them, as
``wf_autoe2e`` scene stage records so no future build repeats the download or
the teacher calls.

Dry run by default. ``--execute`` writes the write-once records; rerunning
with the same artifacts is a no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.autoe2e_stages import (
    LabelIdentity,
    SceneStage,
    label_identity,
    raw_identity,
    validate_label_meta,
)
from Platform.pipelines.autoe2e_tasks import (
    _artifact_objects,
    lookup_scene_stage,
    record_scene_stage,
)
from Platform.pipelines.dataset_publication import sha256_bytes
from Platform.pipelines.dataset_publication_tasks import _get_bytes, _split_s3_uri

MANIFEST_SCHEMA = "kitscenes_recovery_manifest_v1"
# Inputs of Full Run a5nk56zsbcg7g6npjfpt, read from flyteadmin. The label
# meta.json records teacher, prompt, and policy but not stride or uid schema.
JULY_LABELS = LabelIdentity(
    teacher="openai_compatible",
    prompt_version="action_relevant_reasoning_v3_temporal_front256",
    label_stride=10,
    label_policy_version="v2",
    uid_schema_version="v1",
)


def _load_manifest(s3, uri: str, version_id: str, sha256: str) -> dict[str, Any]:
    bucket, key = _split_s3_uri(uri)
    response = s3.get_object(Bucket=bucket, Key=key, VersionId=version_id)
    payload = response["Body"].read()
    if sha256_bytes(payload) != sha256:
        raise ValueError(f"recovery manifest SHA-256 differs from {sha256}")
    manifest = json.loads(payload)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported recovery manifest schema")
    if len(manifest["entries"]) != manifest["summary"]["scene_count"]:
        raise ValueError("recovery manifest entry count differs from its summary")
    return manifest


def _adopt_entry(
    s3,
    *,
    bucket: str,
    source_revision: str,
    entry: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    scene_id = entry["scene_id"]
    raw = raw_identity(AutoE2EDataset.KITSCENES, source_revision, scene_id)
    labels = label_identity(
        AutoE2EDataset.KITSCENES, source_revision, scene_id, JULY_LABELS
    )

    label_bucket, label_prefix = _split_s3_uri(entry["label_uri"])
    meta = json.loads(_get_bytes(
        s3, label_bucket, f"{label_prefix}/meta.json", max_bytes=1024**2
    ))
    validate_label_meta(meta, labels)
    if int(meta["num_records"]) != int(entry["expected_label_count"]):
        raise ValueError(
            f"{scene_id}: label meta has {meta['num_records']} records, "
            f"manifest expects {entry['expected_label_count']}"
        )
    raw_objects = _artifact_objects(s3, entry["raw_uri"])
    if not any(
        item["relative"].startswith(f"data/train/{scene_id}/")
        for item in raw_objects
    ):
        raise ValueError(f"{scene_id}: raw artifact has no files for the scene")

    result = {"scene_id": scene_id, "raw_files": len(raw_objects)}
    for stage, identity, uri in (
        (SceneStage.RAW, raw, entry["raw_uri"]),
        (SceneStage.LABELS, labels, entry["label_uri"]),
    ):
        existing = lookup_scene_stage(
            s3,
            bucket=bucket,
            dataset=AutoE2EDataset.KITSCENES,
            stage=stage,
            identity=identity,
        )
        if existing is not None:
            if existing != uri.rstrip("/"):
                raise ValueError(
                    f"{scene_id}: {stage.value} is already recorded at {existing}"
                )
            result[stage.value] = "already-recorded"
        elif execute:
            record_scene_stage(
                s3,
                bucket=bucket,
                dataset=AutoE2EDataset.KITSCENES,
                stage=stage,
                identity=identity,
                artifact_uri=uri,
            )
            result[stage.value] = "recorded"
        else:
            result[stage.value] = "would-record"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--manifest-version-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--datasets-bucket", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    manifest = _load_manifest(
        s3, args.manifest_uri, args.manifest_version_id, args.manifest_sha256
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(
            lambda entry: _adopt_entry(
                s3,
                bucket=args.datasets_bucket,
                source_revision=manifest["source_revision"],
                entry=entry,
                execute=args.execute,
            ),
            manifest["entries"],
        ))
    summary: dict[str, Any] = {
        "execute": args.execute,
        "source_execution_id": manifest["source_execution_id"],
        "source_revision": manifest["source_revision"],
        "scenes": len(results),
        "missing_scene_ids": manifest["missing_scene_ids"],
        "raw_files": sum(result["raw_files"] for result in results),
    }
    for stage in SceneStage:
        counts: dict[str, int] = {}
        for result in results:
            counts[result[stage.value]] = counts.get(result[stage.value], 0) + 1
        summary[stage.value] = counts
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
