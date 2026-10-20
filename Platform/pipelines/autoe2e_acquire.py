"""wf_acquire_raw_snapshot: copy upstream data into an S3 raw snapshot once.

This is the only code that may reach Hugging Face. Two byte sources exist:

- ``hf``: download a unit at a pinned revision (holds the ``hf-token`` secret);
- ``recovery``: re-tar unit prefixes that an earlier Flyte run already
  downloaded, listed in a frozen recovery manifest. No Hub access.

nuPlan keeps its existing receipt-based acquisition and manifest schema.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from flytekit import PodTemplate, Resources, Secret, dynamic, map_task, task, workflow
from flytekit.types.file import FlyteFile

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset, ManifestRef, ReasoningSpec
from Platform.pipelines.autoe2e_snapshot import (
    build_manifest,
    index_bytes,
    index_tar,
    manifest_bytes,
    manifest_key,
    snapshot_prefix,
    unit_entry,
    unit_index_key,
    unit_tar_key,
    write_unit_tar,
)
from Platform.pipelines.dataset_publication import sha256_bytes
from Platform.pipelines.dataset_publication_tasks import (
    DATA_PREP_IMAGE,
    _put_immutable,
    _split_s3_uri,
)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
UPSTREAM = {
    AutoE2EDataset.KITSCENES: "KIT-MRT/KITScenes-Multimodal",
    AutoE2EDataset.L2D: "yaak-ai/L2D",
}
RECOVERY_SCHEMA = "kitscenes_recovery_manifest_v1"
# Karpenter consolidation evicted the first acquisition's download pods as
# "Underutilized"; long transfers must not be disrupted.
_DO_NOT_DISRUPT = PodTemplate(annotations={"karpenter.sh/do-not-disrupt": "true"})


@dataclass
class UnitSource:
    unit_id: str
    raw_uri: str = ""
    label_uri: str = ""
    expected_label_count: int = -1


@dataclass
class UnitEntry:
    unit_id: str
    tar_key: str
    tar_size: int
    tar_sha256: str
    index_key: str
    index_sha256: str
    member_count: int
    member_bytes: int


@dataclass
class AcquisitionPlan:
    prefix: str
    sources: List[UnitSource] = field(default_factory=list)
    provenance_json: str = "{}"


def _s3():
    import boto3

    return boto3.client("s3", region_name=AWS_REGION)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tar_tree(root: Path, unit_id: str, out_path: Path) -> list:
    """Tar every file under ``root`` (sorted, relative paths) for one unit."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    handles = []

    def _iter():
        for path in files:
            handle = path.open("rb")
            handles.append(handle)
            yield path.relative_to(root).as_posix(), path.stat().st_size, handle

    try:
        with out_path.open("wb") as out:
            write_unit_tar(out, unit_id, _iter())
    finally:
        for handle in handles:
            handle.close()
    with out_path.open("rb") as stream:
        return index_tar(stream)


def upload_unit(s3, *, bucket: str, prefix: str, unit_id: str, tar_path: Path, members: list) -> UnitEntry:
    """Upload a unit tar and its index once; an existing tar must be identical."""
    from botocore.exceptions import ClientError

    tar_key = unit_tar_key(prefix, unit_id)
    tar_sha256 = _sha256_file(tar_path)
    try:
        head = s3.head_object(Bucket=bucket, Key=tar_key)
        if head.get("Metadata", {}).get("sha256") != tar_sha256:
            raise RuntimeError(f"snapshot tar already exists with other bytes: s3://{bucket}/{tar_key}")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        s3.upload_file(str(tar_path), bucket, tar_key, ExtraArgs={"Metadata": {"sha256": tar_sha256}})
        head = s3.head_object(Bucket=bucket, Key=tar_key)
        if head.get("Metadata", {}).get("sha256") != tar_sha256:
            raise RuntimeError(f"another writer raced the snapshot tar: s3://{bucket}/{tar_key}")
    index_payload = index_bytes(members)
    _put_immutable(
        s3, bucket=bucket, key=unit_index_key(prefix, unit_id),
        payload=index_payload, content_type="application/x-ndjson",
    )
    entry = unit_entry(
        unit_id, tar_key=tar_key, tar_size=tar_path.stat().st_size, tar_sha256=tar_sha256,
        index_key=unit_index_key(prefix, unit_id), index_sha256=sha256_bytes(index_payload),
        members=members,
    )
    return UnitEntry(**{k: entry[k] for k in UnitEntry.__dataclass_fields__})


def _download_prefix(s3, uri: str, root: Path) -> None:
    from Platform.pipelines.dataset_publication_tasks import _list_objects

    bucket, key_prefix = _split_s3_uri(uri)
    for item in _list_objects(s3, bucket, key_prefix):
        target = root.joinpath(*item["relative"].split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, item["key"], str(target))


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="2", mem="4Gi"),
    retries=2,
)
def plan_acquisition(
    dataset: AutoE2EDataset,
    revision: str,
    snapshot_id: str,
    split: str,
    units: List[str],
    recovery_manifest: Optional[ManifestRef] = None,
) -> AcquisitionPlan:
    """Fix the unit list: explicit units, the SDK split, or a recovery manifest."""
    prefix = snapshot_prefix(dataset, revision, snapshot_id)
    provenance = {"method": "hf", "split": split, "upstream": UPSTREAM.get(dataset, "")}
    if recovery_manifest is not None:
        s3 = _s3()
        bucket, key = _split_s3_uri(recovery_manifest.uri)
        payload = s3.get_object(Bucket=bucket, Key=key, VersionId=recovery_manifest.version_id)["Body"].read()
        if sha256_bytes(payload) != recovery_manifest.sha256:
            raise ValueError("recovery manifest differs from its pinned SHA-256")
        recovery = json.loads(payload)
        if recovery.get("schema_version") != RECOVERY_SCHEMA or recovery["split"] != split:
            raise ValueError("recovery manifest schema or split does not match")
        if recovery["source_revision"] != revision:
            raise ValueError("recovery manifest revision does not match")
        sources = [
            UnitSource(e["scene_id"], e["raw_uri"], e["label_uri"], int(e["expected_label_count"]))
            for e in recovery["entries"]
        ]
        if units:
            wanted = set(units)
            sources = [s for s in sources if s.unit_id in wanted]
            if len(sources) != len(wanted):
                raise ValueError("requested units are missing from the recovery manifest")
        provenance.update({
            "method": "recovery",
            "recovery_manifest": {"uri": recovery_manifest.uri, "sha256": recovery_manifest.sha256},
            "source_execution_id": recovery["source_execution_id"],
            "missing_scene_ids": recovery["missing_scene_ids"],
        })
        return AcquisitionPlan(prefix, sources, json.dumps(provenance, sort_keys=True))
    if dataset is not AutoE2EDataset.KITSCENES:
        raise NotImplementedError(f"Hub acquisition is implemented for KITScenes only, not {dataset.value}")
    if not units:
        from data_parsing.kit_scenes.source import sdk_split_scene_ids

        units = list(sdk_split_scene_ids(split))
    return AcquisitionPlan(prefix, [UnitSource(u) for u in sorted(units)], json.dumps(provenance, sort_keys=True))


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_DO_NOT_DISRUPT,
    # The largest KITScenes scene is 20.1 GiB; the tree and its tar coexist,
    # and a Hub download also stages the archive before extraction.
    requests=Resources(cpu="4", mem="16Gi", ephemeral_storage="80Gi"),
    limits=Resources(cpu="4", mem="16Gi", ephemeral_storage="80Gi"),
    retries=2,
)
def acquire_unit_from_recovery(
    dataset: AutoE2EDataset,
    plan: AcquisitionPlan,
    source: UnitSource,
    datasets_bucket: str,
) -> UnitEntry:
    """Re-tar one unit an earlier run already downloaded. No Hub access."""
    s3 = _s3()
    work = Path(tempfile.mkdtemp(prefix="autoe2e-acquire-"))
    root = work / "tree"
    _download_prefix(s3, source.raw_uri, root)
    if not any((root / "data").glob(f"*/{source.unit_id}")):
        raise ValueError(f"{source.unit_id}: recovery raw prefix has no files for the unit")
    tar_path = work / "unit.tar"
    members = tar_tree(root, source.unit_id, tar_path)
    shutil.rmtree(root)
    return upload_unit(s3, bucket=datasets_bucket, prefix=plan.prefix, unit_id=source.unit_id, tar_path=tar_path, members=members)


@task(
    container_image=DATA_PREP_IMAGE,
    pod_template=_DO_NOT_DISRUPT,
    # The largest KITScenes scene is 20.1 GiB; the tree and its tar coexist,
    # and a Hub download also stages the archive before extraction.
    requests=Resources(cpu="4", mem="16Gi", ephemeral_storage="80Gi"),
    limits=Resources(cpu="4", mem="16Gi", ephemeral_storage="80Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN", mount_requirement=Secret.MountType.ENV_VAR)],
    retries=2,
)
def acquire_unit_from_hub(
    dataset: AutoE2EDataset,
    revision: str,
    split: str,
    plan: AcquisitionPlan,
    source: UnitSource,
    datasets_bucket: str,
) -> UnitEntry:
    """Download one KITScenes scene at the pinned revision, then tar it."""
    from data_parsing.kit_scenes.source import PinnedKITScenesDownloader

    work = Path(tempfile.mkdtemp(prefix="autoe2e-acquire-"))
    root = work / "tree"
    from flytekit import current_context

    # Flyte mounts the secret under its own variable name, not HF_TOKEN.
    try:
        token = current_context().secrets.get("hf-token", "HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN", "")
    downloader = PinnedKITScenesDownloader(root, revision=revision, token=token or None)
    downloader.download([source.unit_id], expected_split=split)
    for leftover in (".kitscenes_staging", ".cache"):
        shutil.rmtree(root / leftover, ignore_errors=True)
    tar_path = work / "unit.tar"
    members = tar_tree(root, source.unit_id, tar_path)
    shutil.rmtree(root)
    return upload_unit(_s3(), bucket=datasets_bucket, prefix=plan.prefix, unit_id=source.unit_id, tar_path=tar_path, members=members)


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
    retries=2,
)
def adopt_unit_labels(
    dataset: AutoE2EDataset,
    revision: str,
    reasoning: ReasoningSpec,
    source: UnitSource,
    datasets_bucket: str,
    adopted_from: str,
) -> str:
    """Copy one unit's existing labels into the label store and commit them."""
    from data_processing.contract_versions import REASONING_LABEL_POLICY_VERSION, UID_SCHEMA_VERSION

    from Platform.pipelines.autoe2e_labels import (
        LabelIdentity,
        label_attempt_prefix,
        label_id,
        label_identity,
        validate_label_meta,
    )
    from Platform.pipelines.autoe2e_store import commit_label_set, lookup_label_set
    from Platform.pipelines.dataset_publication_tasks import _list_objects

    s3 = _s3()
    identity = label_identity(dataset, revision, source.unit_id, LabelIdentity(
        teacher=reasoning.teacher, teacher_model=reasoning.teacher_model,
        prompt_version=reasoning.prompt_version, label_stride=reasoning.label_stride,
        label_policy_version=REASONING_LABEL_POLICY_VERSION, uid_schema_version=UID_SCHEMA_VERSION,
    ))
    existing = lookup_label_set(s3, bucket=datasets_bucket, dataset=dataset, identity=identity)
    if existing is not None:
        return existing["label_id"]
    bucket, key_prefix = _split_s3_uri(source.label_uri)
    objects = _list_objects(s3, bucket, key_prefix)
    by_name = {o["relative"]: o for o in objects}
    meta = json.loads(s3.get_object(Bucket=bucket, Key=by_name["meta.json"]["key"])["Body"].read())
    validate_label_meta(meta, identity)
    if int(meta["num_records"]) != source.expected_label_count:
        raise ValueError(f"{source.unit_id}: label meta count differs from the recovery manifest")
    records = 0
    for name, item in by_name.items():
        if not name.endswith("records.jsonl"):
            continue
        body = s3.get_object(Bucket=bucket, Key=item["key"])["Body"].read().decode()
        for line in body.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records += 1
            # Every record names its sample, teacher model, and prompt; all three
            # must match the identity the labels are adopted under.
            if f"-{source.unit_id}-" not in str(row.get("sample_id", "")):
                raise ValueError(f"{source.unit_id}: record {row.get('sample_id')!r} is for another unit")
            if row.get("teacher_model") != reasoning.teacher_model:
                raise ValueError(f"{source.unit_id}: record teacher_model {row.get('teacher_model')!r}")
            if row.get("prompt_version") != reasoning.prompt_version:
                raise ValueError(f"{source.unit_id}: record prompt_version {row.get('prompt_version')!r}")
    if records != int(meta["num_records"]):
        raise ValueError(f"{source.unit_id}: {records} records, meta says {meta['num_records']}")
    attempt = f"adopted-{adopted_from}"
    prefix = label_attempt_prefix(dataset, label_id(identity), attempt)
    for name, item in sorted(by_name.items()):
        payload = s3.get_object(Bucket=bucket, Key=item["key"])["Body"].read()
        _put_immutable(s3, bucket=datasets_bucket, key=f"{prefix}/{name}", payload=payload, content_type="application/octet-stream")
    record = commit_label_set(
        s3, bucket=datasets_bucket, dataset=dataset, identity=identity,
        attempt_id=attempt, label_count=int(meta["num_records"]),
    )
    return record["label_id"]


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="4Gi"),
    limits=Resources(cpu="1", mem="4Gi"),
    retries=2,
)
def finalize_snapshot(
    dataset: AutoE2EDataset,
    revision: str,
    snapshot_id: str,
    plan: AcquisitionPlan,
    entries: List[UnitEntry],
    datasets_bucket: str,
) -> ManifestRef:
    """Write the snapshot manifest last and return its pinned reference."""
    from dataclasses import asdict

    manifest = build_manifest(
        dataset=dataset, upstream=UPSTREAM[dataset], revision=revision, snapshot_id=snapshot_id,
        provenance=json.loads(plan.provenance_json), units=[asdict(e) for e in entries],
    )
    payload = manifest_bytes(manifest)
    key = manifest_key(plan.prefix)
    s3 = _s3()
    _put_immutable(s3, bucket=datasets_bucket, key=key, payload=payload, content_type="application/json")
    head = s3.head_object(Bucket=datasets_bucket, Key=key)
    return ManifestRef(
        uri=f"s3://{datasets_bucket}/{key}",
        version_id=str(head.get("VersionId") or "null"),
        sha256=sha256_bytes(payload),
    )


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def nuplan_snapshot_ref(manifest: FlyteFile, manifest_sha256: str) -> ManifestRef:
    uri = str(manifest.remote_source or manifest.path)
    bucket, key = _split_s3_uri(uri)
    head = _s3().head_object(Bucket=bucket, Key=key)
    return ManifestRef(uri=uri, version_id=str(head.get("VersionId") or "null"), sha256=manifest_sha256)


@dynamic(container_image=DATA_PREP_IMAGE, environment={"AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE})
def _acquire(
    dataset: AutoE2EDataset,
    revision: str,
    snapshot_id: str,
    split: str,
    units: List[str],
    datasets_bucket: str,
    recovery_manifest: Optional[ManifestRef],
    adopt_labels: Optional[ReasoningSpec],
    nuplan_source_manifest: Optional[FlyteFile],
    concurrency: int,
) -> ManifestRef:
    import functools

    if dataset is AutoE2EDataset.NUPLAN:
        from Platform.pipelines.nuplan_dataset import _acquire_nuplan_raw_snapshot

        if nuplan_source_manifest is None:
            raise ValueError("nuPlan acquisition needs nuplan_source_manifest")
        snapshot = _acquire_nuplan_raw_snapshot(
            source_manifest=nuplan_source_manifest, datasets_bucket=datasets_bucket,
            aws_region=AWS_REGION, concurrency=concurrency,
        )
        return nuplan_snapshot_ref(manifest=snapshot.manifest, manifest_sha256=snapshot.manifest_sha256)
    plan = plan_acquisition(
        dataset=dataset, revision=revision, snapshot_id=snapshot_id, split=split,
        units=units, recovery_manifest=recovery_manifest,
    )
    if recovery_manifest is not None:
        entries = map_task(
            functools.partial(acquire_unit_from_recovery, dataset=dataset, plan=plan, datasets_bucket=datasets_bucket),
            concurrency=concurrency,
        )(source=plan.sources)
        if adopt_labels is not None:
            map_task(
                functools.partial(
                    adopt_unit_labels, dataset=dataset, revision=revision, reasoning=adopt_labels,
                    datasets_bucket=datasets_bucket, adopted_from=snapshot_id,
                ),
                concurrency=concurrency,
            )(source=plan.sources)
    else:
        entries = map_task(
            functools.partial(
                acquire_unit_from_hub, dataset=dataset, revision=revision, split=split,
                plan=plan, datasets_bucket=datasets_bucket,
            ),
            concurrency=concurrency,
        )(source=plan.sources)
    return finalize_snapshot(
        dataset=dataset, revision=revision, snapshot_id=snapshot_id, plan=plan,
        entries=entries, datasets_bucket=datasets_bucket,
    )


@workflow
def wf_acquire_raw_snapshot(
    dataset: AutoE2EDataset,
    revision: str,
    snapshot_id: str,
    datasets_bucket: str,
    split: str = "train",
    units: List[str] = [],
    recovery_manifest: Optional[ManifestRef] = None,
    adopt_labels: Optional[ReasoningSpec] = None,
    nuplan_source_manifest: Optional[FlyteFile] = None,
    concurrency: int = 32,
) -> ManifestRef:
    """Copy upstream data into an immutable S3 raw snapshot exactly once."""
    return _acquire(
        dataset=dataset, revision=revision, snapshot_id=snapshot_id, split=split, units=units,
        datasets_bucket=datasets_bucket, recovery_manifest=recovery_manifest,
        adopt_labels=adopt_labels, nuplan_source_manifest=nuplan_source_manifest,
        concurrency=concurrency,
    )
