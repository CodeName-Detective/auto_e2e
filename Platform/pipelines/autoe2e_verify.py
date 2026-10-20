"""verify_lineage: recompute every digest from a model version down to raw bytes.

Walks the registry version to its LoggedModel, the checkpoint lineage and the
stored checkpoint, the dataset lineage (the MLflow copy must equal the S3
original), the unit rows, the raw snapshot manifest, every label record, the
evaluation records, and then the parent recursively. The first mismatch
raises, so a pipeline change that breaks traceability fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Mapping, Optional

from Platform.pipelines.autoe2e_lineage import (
    CHECKPOINT_LINEAGE_SCHEMA,
    DATASET_LINEAGE_SCHEMA,
    EVALUATION_SCHEMA,
    parse_record,
    read_units_jsonl,
)
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes


def _download(model, relative: str) -> bytes:
    import mlflow

    with tempfile.TemporaryDirectory() as root:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model.artifact_location.rstrip('/')}/{relative}", dst_path=root,
        )
        with open(path, "rb") as handle:
            return handle.read()


def _list(model, relative: str) -> list[str]:
    import mlflow

    try:
        entries = mlflow.artifacts.list_artifacts(f"{model.artifact_location.rstrip('/')}/{relative}")
    except Exception:
        return []
    names = []
    for entry in entries:
        if entry.is_dir:
            names.extend(f"{entry.path.rsplit('/', 1)[-1]}/{child}" for child in _list(model, f"{relative}/{entry.path.rsplit('/', 1)[-1]}"))
        else:
            names.append(entry.path.rsplit("/", 1)[-1])
    return names


def _s3_bytes(s3, uri: str, version_id: Optional[str] = None) -> bytes:
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    kwargs = {"VersionId": version_id} if version_id else {}
    return s3.get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()


def _s3_sha256(s3, uri: str) -> str:
    from Platform.pipelines.dataset_publication_tasks import _split_s3_uri

    bucket, key = _split_s3_uri(uri)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(8 * 1024**2), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_model(client, s3, model_id: str, *, rehash_checkpoint: bool = True, _seen: Optional[set] = None) -> dict:
    """Verify one LoggedModel's chain; return a summary of what was checked."""
    seen = _seen if _seen is not None else set()
    if model_id in seen:
        raise ValueError(f"lineage cycle at {model_id}")
    seen.add(model_id)
    model = client.get_logged_model(model_id)
    params = model.params
    checkpoint = parse_record(
        _download(model, "lineage/checkpoint.json"),
        expected_sha256=params["lineage_sha256"], schema=CHECKPOINT_LINEAGE_SCHEMA,
    )
    if checkpoint["checkpoint"]["sha256"] != params["checkpoint_sha256"]:
        raise ValueError(f"{model_id}: checkpoint lineage names other weights")
    if int(checkpoint["stage_index"]) != int(params["stage_index"]):
        raise ValueError(f"{model_id}: stage_index param differs from lineage")
    if rehash_checkpoint and _s3_sha256(s3, params["checkpoint_uri"]) != params["checkpoint_sha256"]:
        raise ValueError(f"{model_id}: stored checkpoint bytes changed")
    summary: dict[str, Any] = {"model_id": model_id, "checkpoint_sha256": params["checkpoint_sha256"], "datasets": [], "evaluations": 0}

    dataset_ref = checkpoint["dataset"]
    if dataset_ref.get("status") != "unknown":
        summary["datasets"].append(_verify_dataset(s3, model, dataset_ref))
    for name in _list(model, "evaluations"):
        record_bytes_ = _download(model, f"evaluations/{name}")
        record = parse_record(record_bytes_, expected_sha256=sha256_bytes(record_bytes_), schema=EVALUATION_SCHEMA)
        if record["checkpoint_sha256"] != params["checkpoint_sha256"]:
            raise ValueError(f"{model_id}: evaluation {name} is for other weights")
        if record["checkpoint_lineage_sha256"] != params["lineage_sha256"]:
            raise ValueError(f"{model_id}: evaluation {name} cites another lineage")
        summary["datasets"].append(_verify_dataset(s3, model, record["dataset"]))
        summary["evaluations"] += 1

    parent = checkpoint.get("parent")
    if parent:
        parent_summary = verify_model(client, s3, parent["model_id"], rehash_checkpoint=rehash_checkpoint, _seen=seen)
        parent_model = client.get_logged_model(parent["model_id"])
        if parent_model.params["lineage_sha256"] != parent["lineage_sha256"]:
            raise ValueError(f"{model_id}: parent lineage digest differs from the parent model")
        summary["parent"] = parent_summary
    return summary


def _verify_dataset(s3, model, ref: Mapping[str, Any]) -> dict:
    dataset_id = ref["dataset_id"]
    copy = _download(model, f"lineage/datasets/{dataset_id}/dataset.json")
    lineage = parse_record(copy, expected_sha256=ref["lineage_sha256"], schema=DATASET_LINEAGE_SCHEMA)
    if lineage["dataset_id"] != dataset_id:
        raise ValueError(f"dataset lineage copy is for {lineage['dataset_id']}, not {dataset_id}")
    units_uri = lineage["units"]["uri"]
    if not units_uri.endswith("/lineage/units.jsonl"):
        raise ValueError(f"{dataset_id}: unexpected units location {units_uri}")
    original_uri = units_uri[: -len("units.jsonl")] + "dataset.json"
    if _s3_bytes(s3, original_uri) != copy:
        raise ValueError(f"MLflow copy of {dataset_id} lineage differs from S3")
    units_payload = _download(model, f"lineage/datasets/{dataset_id}/units.jsonl")
    if sha256_bytes(units_payload) != lineage["units"]["sha256"] or _s3_bytes(s3, lineage["units"]["uri"]) != units_payload:
        raise ValueError(f"{dataset_id}: units.jsonl differs from its lineage digest")
    units = read_units_jsonl(units_payload)
    if sha256_bytes(canonical_json_bytes(lineage["spec"])) != dataset_id:
        raise ValueError(f"{dataset_id}: spec digest differs")
    manifest_ref = lineage["spec"]["source_manifest"]
    manifest = _s3_bytes(s3, manifest_ref["uri"], manifest_ref["version_id"])
    if sha256_bytes(manifest) != lineage["raw"]["snapshot_manifest_sha256"]:
        raise ValueError(f"{dataset_id}: raw snapshot manifest changed")
    snapshot = json.loads(manifest)
    tar_digests = {u["unit_id"]: u["tar_sha256"] for u in snapshot.get("units", [])}
    labels_checked = 0
    for unit in units:
        if tar_digests and tar_digests.get(unit.unit_id) != unit.raw_tar_sha256:
            raise ValueError(f"{dataset_id}: {unit.unit_id} raw tar digest differs from the snapshot")
        if unit.label_id is not None:
            from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
            from Platform.pipelines.autoe2e_labels import label_record_key

            bucket = lineage["units"]["uri"].split("/")[2]
            key = label_record_key(AutoE2EDataset(lineage["spec"]["dataset"]), unit.label_id)
            if sha256_bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != unit.label_record_sha256:
                raise ValueError(f"{dataset_id}: label record of {unit.unit_id} changed")
            labels_checked += 1
    return {"dataset_id": dataset_id, "units": len(units), "labels": labels_checked}


def verify_lineage(client, s3, model_version: str, *, rehash_checkpoint: bool = True) -> dict:
    """Verify a registry version ("auto-e2e-driving-policy/N") end to end."""
    name, _, version = model_version.rpartition("/")
    registered = client.get_model_version(name, version)
    if not registered.source.startswith("models:/m-"):
        raise ValueError(f"{model_version} does not point at a LoggedModel")
    model_id = registered.source.removeprefix("models:/")
    params = client.get_logged_model(model_id).params
    for key, value in (registered.tags or {}).items():
        if key in params and params[key] != value:
            raise ValueError(f"{model_version}: version tag {key} differs from the model")
    return verify_model(client, s3, model_id, rehash_checkpoint=rehash_checkpoint)
