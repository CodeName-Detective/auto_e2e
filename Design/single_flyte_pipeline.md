# Design: Single Flyte Pipeline from Raw Data to a Traceable Model

## Document Metadata

| Field | Value |
|---|---|
| Status | Draft |
| Owner | riita10069 |
| Created | 2026-10-20 |
| Replaces | All 40 `wf_*` entry points in `Platform/pipelines/` on main at `c7fc884d` |
| Entry points | `wf_acquire_raw_snapshot` once per upstream revision, `wf_autoe2e` for everything else |
| Training path | Ray Train on KubeRay, admitted by Kueue |

## 1. Executive Summary

Producing one evaluated model today takes several separate launches. A
KITScenes model needs `wf_sharded_full_run` or a Ray training workflow, then
`wf_evaluate_kitscenes_benchmark`, then `wf_publish_full_run_overlays` from a
CodeBuild launcher, then a manual Console materialization Job. A nuPlan model
also needs `wf_acquire_nuplan_raw_snapshot`, a pack workflow, and a
hand-written S3 list of shard URIs. Results end up in different MLflow runs
and experiments, up to three model versions are created per training run,
and no record connects a checkpoint to the raw data and labels it was
trained on.

This design keeps two entry points:

- `wf_acquire_raw_snapshot` copies upstream data into an immutable S3 raw
  snapshot once per upstream revision. It is the only code that can reach
  Hugging Face.
- `wf_autoe2e` does everything else from S3: it builds or reuses a
  dataset, trains, evaluates, registers, and precomputes Console overlays.

The data flows through four layers, each immutable and addressed by digest:

```
raw snapshot ──► labels ──► published dataset ──► checkpoint ──► evaluations
 (tar + index)   (per unit)   (WebDataset shards)   (model version)   (same run)
```

Every MLflow model version carries a lineage record that names each layer
by digest, so any checkpoint can be traced back to the exact raw bytes,
labels, pack code, and parent model it came from, and every evaluation ever
run on it is on its run.

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Entry points | `wf_acquire_raw_snapshot` and `wf_autoe2e` only | One interface to learn. Acquisition has a different lifecycle and is the only code that holds the HF token |
| Upstream access | `wf_autoe2e` tasks never receive the `hf-token` secret | Processing never depends on the Hub, and the guarantee is structural |
| Raw format | Uncompressed tar per unit plus a Parquet member index | nuPlan archives already are uncompressed tar. Range reads of one member need only a key, an offset, and a length |
| Pack format | WebDataset shards, as today | Training streams samples sequentially |
| Datasets | KITScenes, nuPlan, L2D behind one adapter interface. NVIDIA PhysicalAI is deleted | The workflow never branches on the dataset. NVIDIA has no fan-out planner and no Reactive stage |
| Datasets per execution | Exactly one training dataset | Keeps lineage unambiguous. Evaluation datasets are separate inputs |
| Expensive stages | Raw is acquired once; labels are recorded per unit and never repeated | A pack contract bump must only re-pack |
| Dataset reuse | By-id index lookup by `dataset_id` | Flyte cache keys include non-semantic inputs, and DataCatalog write failures already forced a manual recovery manifest (#121) |
| Training backend | Ray only. `train_il` and `train_offline_rl` are removed | One code path for 1, 2, 4, and 8 GPUs |
| Multi-stage training | `parent` accepts a model version or a checkpoint | Chains of any depth, one execution per stage |
| Model versions | One per distinct checkpoint: `best`, and `final` only if its SHA-256 differs | No duplicate versions for the same weights |
| Evaluation-only runs | Append to the run of the evaluated version; no new version | All evaluations of one checkpoint stay on one run |
| MLflow layout | One experiment `auto-e2e`, one run per training execution | One place to compare everything |

## 3. Interface

```python
@workflow
def wf_acquire_raw_snapshot(
    dataset: AutoE2EDataset,
    upstream_revision: str,
    units: List[str],               # empty list = every unit at the revision
    datasets_bucket: str,
) -> ManifestRef: ...               # the snapshot manifest

@workflow
def wf_autoe2e(
    dataset: DatasetSpec,
    training: TrainingSpec,
    evaluation: EvaluationSpec,
    parent: Optional[ModelRef] = None,
    runtime: RuntimeOptions = RuntimeOptions(),
) -> PipelineResult: ...
```

```python
@dataclass
class DatasetSpec:                   # every field enters dataset_id
    dataset: AutoE2EDataset          # KITSCENES, NUPLAN, L2D
    source_manifest: ManifestRef     # raw snapshot manifest: URI + VersionId + SHA-256
    unit_limit: int                  # 0 = all units; N = first N in manifest order
    pack_version: str                # code constant
    image_size: int
    pack_params: Dict[str, str]      # adapter-validated, e.g. nuPlan scenario budget
    reasoning: Optional[ReasoningSpec]  # teacher, prompt_version, label_stride

@dataclass
class ModelRef:                      # exactly one field is set
    model_version: Optional[str]     # e.g. "auto-e2e-driving-policy/66"
    checkpoint: Optional[FlyteFile]  # a checkpoint that was never registered

@dataclass
class TrainingSpec:
    gpu_profile: GpuProfile          # RAY_1, RAY_2, RAY_4, RAY_8
    epochs: int                      # 0 = evaluate the parent only
    learning_rate: float
    per_rank_batch_size: int
    seed: int
    objectives: ObjectiveSpec        # loss weights

@dataclass
class EvaluationSpec:
    suites: List[EvalSuite]          # INTERNAL_VAL, KITSCENES_BENCHMARK, REACTIVE
    datasets: Dict[str, DatasetSpec] # suite name -> evaluation dataset, if not INTERNAL_VAL

@dataclass
class RuntimeOptions:                # speed only, never enters any identity
    label_concurrency: int
    pack_concurrency: int
    copy_workers: int
```

The upstream revision is read from `source_manifest`, so the manifest digest
pins it. Dataset-specific behavior lives behind the adapter for
`DatasetSpec.dataset`.

## 4. Data Layers

### 4.1 Raw snapshots

```
s3://<datasets_bucket>/<dataset>/raw-snapshots/<upstream_revision>/<snapshot_id>/
  units/<unit_id>.tar      # uncompressed tar, upstream files byte for byte
  index/<unit_id>.parquet  # one row per member
  manifest.json            # written last, write-once
```

Each tar member is named `<unit_id>/<upstream relative path>`, so the tars
also stream with WebDataset tools. JPEG and Parquet sources are already
compressed, so the tar is not.

The index has one row per member: tar key, `offset_data`, size, SHA-256, and
the adapter's parsed fields (sensor, timestamp, frame index). It is written
once by streaming the tar with the standard `tarfile` module. Reading one
member is a single ranged GET, `fs.cat_file(tar_key, offset, offset + size)`;
no FUSE and no extraction. Planners select members by querying the index
instead of listing S3.

The manifest records the dataset, upstream repository and revision, how the
bytes were obtained, and per unit: tar key, tar size, tar SHA-256, member
count, and index SHA-256. `wf_autoe2e` checks the manifest digest against
`DatasetSpec.source_manifest` before reading anything.

`wf_acquire_raw_snapshot` downloads each requested unit, verifies upstream
hashes, writes the tar and index with `IfNoneMatch="*"`, and writes the
manifest last. Rerunning it rewrites nothing.

| Dataset | Unit | Snapshot | Bytes come from |
|---|---|---|---|
| KITScenes train | scene | `kitscenes/raw-snapshots/6fde0034…/` | the 533 `raw_uri` prefixes of Full Run `a5nk56zsbcg7g6npjfpt` in the Flyte artifacts bucket, tarred in-region (1.75 TB, 14.47 million objects become 533 tars). No Hub access |
| nuPlan | archive | `nuplan/raw-snapshots/nuplan-v1.1-complete/nuplan-v1.1-complete-20260809/` | already exists. Its 341 archives are uncompressed tar despite the `.zip` name; only the index is added |
| L2D, KITScenes val and test | episode, scene | none yet | the first `wf_acquire_raw_snapshot` run is the only Hub access |

### 4.2 Labels

Labels are billed Cosmos calls and are never repeated. A label set is per
unit, and its identity excludes everything about the pack:

| Identity fields | Not included |
|---|---|
| dataset, upstream revision, unit id, teacher, `prompt_version`, `label_stride`, `REASONING_LABEL_POLICY_VERSION`, `UID_SCHEMA_VERSION` | parser, shard, and geometry versions |

Label files live under `<dataset>/labels/<label_id>/`, with a write-once
record at `<dataset>/_stages/labels/<label_id>.json` holding a digest over
every file's relative key, size, and ETag. A build verifies the digest before
use and fails on a mismatch instead of re-labeling.

A parser change can change which frames exist, but it does not re-label. The
pack JOIN treats a sample without a label as fully masked, and the
publication records `reasoning_label_count`, so lost coverage is visible.

The 533 KITScenes label sets of Full Run `a5nk56zsbcg7g6npjfpt` are copied
into the label store once. A read-only dry run on 2026-10-20 checked every
label `meta.json` against the label identity.

### 4.3 Published datasets

`dataset_id` is the SHA-256 of the canonical JSON of `DatasetSpec`.
Publications keep the `<dataset>/vN.M/` layout that the Console reads. After
the manifest gate, the build writes two write-once objects:

- `<dataset>/by-id/<dataset_id>.json`: the canonical spec, version, and
  manifest key and SHA-256;
- `<dataset>/vN.M/lineage/dataset.json` and `units.parquet`: the dataset
  lineage record (6.1), naming the raw snapshot, every label record, and the
  pack code.

The Console ignores both because they are not version directories or
shards. A publication is reused only if the index's spec equals the request,
the manifest SHA-256 matches the index and `.publication-lock.json`, the
manifest is `ready`, and every shard exists with the recorded size and
ETag. Any mismatch fails the execution.

### 4.4 Dataset adapters

| Member | Meaning |
|---|---|
| `published_name(plan)` | e.g. `kitscenes`, `kitscenes-val`, `nuplan` |
| `read_snapshot(manifest, unit_limit, pack_params)` | returns a `SourcePlan`: upstream revision, units, and pack partitions |
| `acquire_task` | downloads one unit into a snapshot. Used only by `wf_acquire_raw_snapshot` |
| `index_member(name)` | parses sensor, timestamp, and frame index from a member name |
| `label_task` | labels one unit from the snapshot, or `None` |
| `pack_task` | packs one partition from snapshot members and labels into WebDataset shards |

| | KITScenes | nuPlan | L2D |
|---|---|---|---|
| `acquire_task` | HF download of one scene | `acquire_nuplan_archive` | HF download of one episode |
| `label_task` | `generate_reasoning_labels` | `None` | `generate_reasoning_labels` |
| Partition | one scene | one sensor group (43 for train) | `pack_params["partition_size"]` episodes |
| `pack_task` | `data_processing` | `pack_nuplan_snapshot_reactive_dataset` | `data_processing` |
| `pack_params` | none | `total_scenario_limit`, `samples_per_shard`, `max_rejection_fraction` | `partition_size`, `osm_graph_snapshot` |
| Validation split | frozen holdout `kitscenes_train_dev_v4.json` | hash buckets of the log id | hash buckets |

Pack tasks read only the members their partition needs. A nuPlan pack no
longer downloads and extracts whole 157 to 250 GB archives, which is what
forces 1200Gi of ephemeral storage today, so pack partitions can be small
and many.

The code that fetches metadata from the Hub today
(`plan_fanout_partitions` through `LeRobotDatasetMetadata`,
`Model/data_parsing/kit_scenes/source.py`, and the KITScenes test direct
pack) reads the same files from the snapshot instead.

Known gaps:

- The L2D Reactive contract needs OSM navigation (14 map channels), but no
  Flyte task passes `osm_graph_snapshot_path` to the packer. The L2D adapter
  rejects a spec without `osm_graph_snapshot` until that is wired.

## 5. `wf_autoe2e` Stages

| # | Stage | Skipped when |
|---|---|---|
| 1 | Resolve the training dataset and each evaluation dataset by `dataset_id` | — |
| 2 | Build: label units without a label record, pack every partition, publish, write by-id and lineage | the dataset is already published |
| 3 | Resolve the parent: download the checkpoint, verify its SHA-256, read its `stage_index` | no parent |
| 4 | Train with Ray Train | `epochs == 0` |
| 5 | Evaluate every suite on every distinct checkpoint | — |
| 6 | Register one model version per new distinct checkpoint | `epochs == 0` |
| 7 | Precompute Console overlays for `best` | `OVLSET` is ready for the same checkpoint and dataset |

`train_ray` is one task factory. `gpu_profile` picks the `RayJobConfig`,
Kueue queue, priority class, and node selector from a single table. The task
reads the active Capacity Block end time itself.

Registration first searches `auto-e2e-driving-policy` for a version whose
`checkpoint_sha256` tag equals the checkpoint, so one set of weights never
gets two versions, even across executions.

With `epochs == 0`, nothing is trained. If the parent is a model version,
stage 5 appends to that version's run. If the parent is an unregistered
checkpoint, the execution creates a run and registers the checkpoint once
with `checkpoint_role=imported`; its checkpoint lineage marks `dataset`,
`split`, and `training` as unknown instead of guessing them.

## 6. Traceability in MLflow

All runs live in one experiment, `auto-e2e`. A training execution creates
one run; its versions and every later evaluation of them point at it.

| On the run | Content |
|---|---|
| params | flattened `DatasetSpec`, `TrainingSpec`, `ModelRef`; `gpu_profile`; Kueue queue |
| tags | `flyte_execution_id`, training image digest, git commit, capacity reservation id, `dataset_id`, `stage_index` |
| inputs (`mlflow.log_input`) | training dataset with context `training`; each evaluation dataset with context `eval:<suite>`. Source is the by-id index URI, digest is `dataset_id` |
| metrics | training curves per step; `eval/<role>/<suite>/<metric>` per checkpoint role |
| artifacts | `lineage/<checkpoint_sha256>.json` and `evaluations/<checkpoint_sha256>/<suite>/<dataset_id>.json` (6.1), validation unit list |

Each model version carries tags: `checkpoint_role`, checkpoint URI and
SHA-256, `dataset_id`, `parent_model_version`, `parent_checkpoint_sha256`,
`stage_index`, overlay manifest key and SHA-256, and the headline metrics of
each suite. An evaluation added later writes the same metric names with the
evaluation dataset's version in the suite name, for example
`eval/best/kitscenes_benchmark@kitscenes-val-v3.5/ade_3s`, so later results
never overwrite earlier ones.

### 6.1 Lineage records

Lineage is three kinds of record. Each is canonical JSON (sorted keys,
compact separators), carries a `schema_version`, is written once with
`IfNoneMatch="*"`, and is identified by the SHA-256 of its bytes. A record
refers to another record only by URI plus SHA-256, never by copying it, so
the chain is a hash chain: changing any byte upstream breaks every digest
downstream.

| Record | Written by | Location | One per |
|---|---|---|---|
| dataset lineage | stage 2, after the shards and before the by-id index | `<dataset>/vN.M/lineage/dataset.json` and `units.parquet` | published dataset |
| checkpoint lineage | stage 6, or stage 5 for an imported checkpoint | MLflow run artifact `lineage/<checkpoint_sha256>.json` | distinct checkpoint |
| evaluation record | stage 5 | MLflow run artifact `evaluations/<checkpoint_sha256>/<suite>/<dataset_id>.json` | checkpoint, suite, and evaluation dataset |

A dataset lineage is shared by every checkpoint trained on that dataset, so
its canonical copy lives with the publication in S3. Every run that uses a
dataset, for training or evaluation, also stores byte-identical copies of
`dataset.json` and `units.parquet` under
`lineage/datasets/<dataset_id>/`. The copies are checked against the
recorded SHA-256, so the hash chain holds, and MLflow alone stays enough to
trace a checkpoint even if the S3 publication is gone. Checkpoint lineage
and evaluation records live only on the MLflow run. Evaluation records are only ever added, so a
checkpoint's history is the set of its records, and no earlier record is
rewritten.

#### Dataset lineage (`autoe2e_dataset_lineage_v1`)

`units.parquet` has one row per unit used:

| Column | Meaning |
|---|---|
| `unit_id` | scene, archive, or episode id |
| `partition_id` | pack partition that consumed the unit |
| `raw_tar_key`, `raw_tar_sha256`, `raw_index_sha256` | from the raw snapshot manifest |
| `label_id`, `label_record_sha256` | null for a dataset without labels |
| `label_origin` | `reused` or `created` in this build |
| `sample_count`, `label_count` | contribution of the unit to the published shards |

`dataset.json`:

```json
{
  "schema_version": "autoe2e_dataset_lineage_v1",
  "dataset_id": "…",
  "spec": {"dataset": "kitscenes", "source_manifest": {"uri": "s3://…/manifest.json", "version_id": "…", "sha256": "…"}, "unit_limit": 0, "pack_version": "…", "image_size": 256, "pack_params": {}, "reasoning": {"teacher": "openai_compatible", "prompt_version": "action_relevant_reasoning_v3_temporal_front256", "label_stride": 10}},
  "published": {"name": "kitscenes", "version": "v3.6", "manifest_key": "kitscenes/v3.6/shards/manifest.json", "manifest_sha256": "…"},
  "raw": {"upstream": "KIT-MRT/KITScenes-Multimodal", "revision": "6fde0034…", "snapshot_manifest_sha256": "…", "units": 533},
  "labels": {"label_policy_version": "v2", "uid_schema_version": "v1", "label_set_sha256": "…", "reused": 533, "created": 0},
  "units": {"uri": "s3://…/kitscenes/v3.6/lineage/units.parquet", "sha256": "…", "rows": 533},
  "pack": {"contracts": {"uid": "v1", "parser": "v3", "shard": "v13", "geometry": "v7"}, "image_digest": "sha256:…", "git_commit": "…", "task": "data_processing"},
  "build": {"flyte_execution_id": "…", "finished_at": "2026-10-21T03:12:00Z"},
  "totals": {"samples": 42667, "shards": 404, "reasoning_labels": 4598}
}
```

`label_set_sha256` is the digest of the sorted `(unit_id, label_id,
label_record_sha256)` rows, so two datasets that share exactly the same labels
have the same value even if their packs differ. `spec` repeats the
`DatasetSpec` so that `dataset_id` can be recomputed from this record alone.

#### Checkpoint lineage (`autoe2e_checkpoint_lineage_v1`)

```json
{
  "schema_version": "autoe2e_checkpoint_lineage_v1",
  "checkpoint": {"sha256": "…", "uri": "s3://…/best.pt", "role": "best", "epoch": 6, "step": 21333},
  "stage_index": 2,
  "parent": {"model_version": "auto-e2e-driving-policy/66", "checkpoint_sha256": "…", "lineage_sha256": "…"},
  "dataset": {"dataset_id": "…", "lineage_uri": "s3://…/kitscenes/v3.6/lineage/dataset.json", "lineage_sha256": "…"},
  "split": {"policy": "kitscenes_train_dev_v4", "validation_units_sha256": "…", "validation_units": 40},
  "training": {"spec": {"gpu_profile": "RAY_8", "epochs": 10, "learning_rate": 3e-05, "per_rank_batch_size": 2, "seed": 149, "objectives": {}}, "flyte_execution_id": "…", "mlflow_run_id": "…", "image_digest": "sha256:…", "git_commit": "…", "kueue_queue": "p5en-capacity-block", "capacity_reservation_id": "cr-…"},
  "selection": {"metric": "eval/internal_val/ade_6p4s", "value": 2.633}
}
```

`parent` is `null` for a model trained from scratch. For an imported
checkpoint, `dataset`, `split`, and `training` are
`{"status": "unknown"}` instead of guesses, and `stage_index` is 0.

#### Evaluation record (`autoe2e_evaluation_v1`)

```json
{
  "schema_version": "autoe2e_evaluation_v1",
  "checkpoint_sha256": "…",
  "checkpoint_lineage_sha256": "…",
  "suite": "kitscenes_benchmark",
  "dataset": {"dataset_id": "…", "lineage_sha256": "…", "published": "kitscenes-val/v3.5"},
  "protocol": {"name": "kitscenes_paper_approximation", "version": "v1", "horizons_s": [3, 5]},
  "model_reexecuted": true,
  "metrics": {"ade_3s": 0.71, "fde_3s": 1.90, "ade_5s": 1.85, "fde_5s": 5.38},
  "per_sample": {"uri": "s3://…/per_sample.parquet", "sha256": "…", "rows": 200},
  "code": {"image_digest": "sha256:…", "git_commit": "…"},
  "flyte_execution_id": "…",
  "finished_at": "…"
}
```

`model_reexecuted` is always `true`: a suite that replays stored overlays
instead of running the model is not an evaluation and is not written as
one.

#### Links into MLflow

| Where | Key | Value |
|---|---|---|
| version tag | `lineage_sha256` | checkpoint lineage digest |
| version tag | `dataset_id`, `dataset_lineage_sha256` | from the checkpoint lineage |
| version tag | `eval.<suite>@<dataset>.<metric>` | headline metrics, copied from evaluation records |
| run input | `mlflow.log_input` | an MLflow Dataset with name `<published name>/<version>`, digest `dataset_id`, source the by-id index URI, context `training` or `eval:<suite>`, and a profile holding `dataset_lineage_sha256`, `snapshot_manifest_sha256`, `label_set_sha256`, pack contracts, and sample and shard totals |
| run artifacts | `lineage/`, `lineage/datasets/`, `evaluations/` | the records themselves, including copies of every dataset lineage the run used |

#### Verification

`verify_lineage(model_version)` walks the chain and recomputes every digest:
the version's checkpoint lineage, the dataset lineage and `units.parquet` in
S3, the raw snapshot manifest, the label records, each evaluation record,
and then the parent's chain recursively. It fails on the first mismatch. The
smoke launch plan runs it after registration, so a pipeline change that
breaks lineage fails the smoke run.

### 6.2 Questions this answers

| Question | Where |
|---|---|
| Which data trained this checkpoint? | version tag `lineage_sha256` → checkpoint lineage `dataset` → dataset lineage |
| Which raw bytes and upstream revision? | dataset lineage `raw` and `units.parquet` tar digests |
| Which labels, from which teacher and prompt? | dataset lineage `spec.reasoning` and `labels`; `units.parquet` names every label record |
| Which code packed and trained it? | dataset lineage `pack` and checkpoint lineage `training` |
| What was it initialized from? | `parent`, recursively |
| Every evaluation ever run on it, and on which data? | run metrics `eval/<role>/…`, inputs `eval:<suite>`, and `evaluations/<checkpoint_sha256>/` |

## 7. Launch Plans

Only `autoe2e_smoke` is defined: KITScenes with `unit_limit=10`, `RAY_1`, 1
epoch, `INTERNAL_VAL`. Every other run passes its specs explicitly, so its
inputs are visible on the execution page.

## 8. Deletion

| Removed | Where its logic goes |
|---|---|
| `wf_sharded_full_run`, `wf_train_il`, `wf_train_offline_rl`, `wf_full_pipeline`, `wf_ingest_train_eval` | `wf_autoe2e`. The IL and offline RL training loops are deleted |
| `wf_train_reactive_*` (9 workflows), `wf_reactive_multistage_ray_2_canary` | `train_ray` with `gpu_profile` and `parent` |
| `wf_data_ingest`, `wf_acquire_nuplan_raw_snapshot` | `wf_acquire_raw_snapshot` |
| `wf_data_processing`, `wf_generate_reasoning_labels`, `wf_create_dataset*`, `wf_create_kitscenes_*evaluation_sharded`, nuPlan pack workflows | adapters and stage 2 |
| `wf_evaluate_*` | stage 5 |
| overlay and dataset publication workflows | stages 2 and 7 |
| `wf_audit_kitscenes_*` | validation inside stages 1 and 2 |
| `wf_prepare_kitscenes_paper_approximation`, `wf_export_trajectory_report` | deleted |
| NVIDIA PhysicalAI branches and `physical_ai_av` in the data-prep image | deleted |

The CodeBuild launcher buildspecs for removed workflows and the tasks no
stage calls are deleted with them. After `wf_autoe2e` is registered, the old
entry points are archived in Flyte.

## 9. Out of Scope

- Making Kueue admission depend on an existing Capacity Block
  (ProvisioningRequest or an AdmissionCheck).
- Automatic Capacity Block purchase. `Platform/README.md` keeps purchase as
  an operator action.
- Multi-node training and EFA.
