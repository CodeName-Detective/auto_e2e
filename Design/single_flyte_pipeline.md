# Design: Single Flyte Pipeline from Dataset to Registered Model

## Document Metadata

| Field | Value |
|---|---|
| Status | Draft |
| Owner | riita10069 |
| Created | 2026-10-20 |
| Replaces | All 40 `wf_*` entry points in `Platform/pipelines/` on main at `c7fc884d` |
| Entry points | `wf_autoe2e`, plus `wf_acquire_raw_snapshot` run once per upstream revision |
| Training path | Ray Train on KubeRay, admitted by Kueue |

## 1. Executive Summary

Producing one evaluated model today takes several separate launches. A
KITScenes model needs `wf_sharded_full_run` or a Ray training workflow, then
`wf_evaluate_kitscenes_benchmark`, then `wf_publish_full_run_overlays` from a
CodeBuild launcher, then a manual Console materialization Job. A nuPlan model
also needs `wf_acquire_nuplan_raw_snapshot`, a pack workflow, and a
hand-written S3 list of shard URIs. Evaluation results end up in different
MLflow runs and experiments, and up to three model versions are created per
training run.

This design deletes every existing entry point. Upstream data is copied into
S3 exactly once per dataset revision by `wf_acquire_raw_snapshot`, the only
workflow that can reach Hugging Face. Everything else is one workflow,
`wf_autoe2e`, with a small, dataset-agnostic interface that reads raw data
only from those S3 snapshots. One `wf_autoe2e` execution:

1. resolves the requested dataset to a content-addressed `dataset_id`;
2. builds and publishes that dataset only if no valid publication exists;
3. initializes from a parent model if one is given, then trains with Ray
   Train;
4. evaluates the best and final checkpoints;
5. registers the `best` model version, and a `final` version only when it is
   a different checkpoint;
6. precomputes Console overlays for the `best` version.

Multi-stage training of any depth is a chain of executions. Each execution
takes the previous stage's model version as its parent.

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Entry points | `wf_autoe2e` for everything after acquisition, and `wf_acquire_raw_snapshot` once per upstream revision | Acquisition has a different lifecycle and is the only code that holds the HF token |
| Upstream access | Only `wf_acquire_raw_snapshot` reads Hugging Face. `wf_autoe2e` tasks never receive the `hf-token` secret | Processing must not depend on, or be rate-limited by, the Hub, and the guarantee is structural |
| Training backend | Ray only. `train_il` and `train_offline_rl` are removed | One code path for 1, 2, 4, and 8 GPUs. Kueue already admits `ray.io/rayjob` |
| Datasets per execution | Exactly one | Keeps `dataset_id`, evaluation data, and lineage unambiguous |
| Multi-stage training | `parent` input that accepts an MLflow model version or a checkpoint | Stages are chained one execution at a time, with no limit on depth |
| Model versions per execution | `best`, plus `final` only if its checkpoint SHA-256 differs from `best` | No duplicate versions for the same weights |
| Dataset skip mechanism | Published manifest lookup by `dataset_id` | Flyte cache keys include non-semantic inputs such as concurrency, and DataCatalog write failures have already forced a manual recovery manifest (#121) |
| Dataset differences | One adapter per dataset behind a fixed interface (4.3) | The workflow, stage store, publication, and training never branch on the dataset |
| Supported datasets | KITScenes, nuPlan, L2D. NVIDIA PhysicalAI is deleted | NVIDIA has no fan-out planner and no Reactive stage; only the removed `wf_full_pipeline` used it |
| Source of units | `source_manifest` is always an S3 raw snapshot manifest | Unit lists and raw bytes come from one immutable, SHA-pinned object instead of a Hub listing at run time |
| Expensive stages | Raw data is acquired once into a snapshot; reasoning labels are recorded per unit and never repeated (4.2, 4.4) | A pack contract bump must only re-pack |

## 3. Interface

```python
@workflow
def wf_acquire_raw_snapshot(
    dataset: AutoE2EDataset,
    upstream_revision: str,
    units: List[str],               # empty list = every unit at the revision
    datasets_bucket: str,
) -> ManifestRef: ...               # the snapshot manifest for wf_autoe2e

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
class DatasetSpec:
    dataset: AutoE2EDataset          # KITSCENES, NUPLAN, L2D
    source_manifest: ManifestRef     # S3 raw snapshot manifest: URI + VersionId + SHA-256
    pack_version: str                # code constant
    image_size: int
    pack_params: Dict[str, str]      # adapter-validated, e.g. nuPlan scenario budget
    reasoning: Optional[ReasoningSpec]  # teacher, prompt_version, label_stride

@dataclass
class ModelRef:
    model_version: Optional[str]     # e.g. "auto-e2e-driving-policy/66"
    checkpoint: Optional[FlyteFile]  # used when the parent is not registered
    # exactly one of the two must be set

@dataclass
class TrainingSpec:
    gpu_profile: GpuProfile          # RAY_1, RAY_2, RAY_4, RAY_8
    epochs: int                      # 0 evaluates the parent without training
    learning_rate: float
    per_rank_batch_size: int
    seed: int
    objectives: ObjectiveSpec        # loss weights

@dataclass
class EvaluationSpec:
    suites: List[EvalSuite]          # INTERNAL_VAL, KITSCENES_BENCHMARK, REACTIVE
    benchmark_dataset: Optional[DatasetSpec]

@dataclass
class RuntimeOptions:
    ingest_concurrency: int
    label_concurrency: int
    pack_concurrency: int
    copy_workers: int
```

`RuntimeOptions` changes how fast an execution runs, never what it
produces, so it never enters `dataset_id`. Dataset-specific behavior lives
behind the adapter for `DatasetSpec.dataset`, not in the workflow signature.
The upstream revision is read from `source_manifest`, so it is pinned by the
manifest digest instead of being a second, possibly inconsistent input.

## 4. Stages

### 4.1 Resolve dataset

`dataset_id` is the SHA-256 of the canonical JSON of `DatasetSpec`
(`Platform/pipelines/autoe2e_dataset.py`). Publications keep the
`<dataset>/vN.M/` layout, because the Console discovers datasets only through
version directories. After the manifest gate is written, the pipeline adds a
write-once index object `s3://<datasets_bucket>/<dataset>/by-id/<dataset_id>.json`
that records the canonical spec, the version, and the manifest key and
SHA-256. The Console ignores `by-id/` because it is not a version directory.

The resolver task (`resolve_dataset` in `Platform/pipelines/autoe2e_tasks.py`)
reads that index. A publication is reused only if all of these hold:

- the index's spec and id equal the requested ones;
- the manifest's SHA-256 matches both the index and `.publication-lock.json`;
- the manifest is `ready` and names the same dataset and version;
- every listed shard object exists with the recorded size and ETag.

If the index is valid, the task returns a `DatasetHandle` with
`published=True` and the shard URIs, and the build stage is skipped through a
Flyte `conditional`. If no index exists, it returns `published=False` and the
next minor version after the newest version directory, which the build stage
publishes under. Any validation failure fails the execution rather than
falling back to a rebuild. Two executions that build the same new spec
concurrently cannot both publish: the second one fails at the write-once
manifest or index.

### 4.2 Raw snapshots

A raw snapshot is an immutable copy of upstream data in the datasets bucket:

```
s3://<datasets_bucket>/<dataset>/raw-snapshots/<upstream_revision>/<snapshot_id>/
  units/<unit_id>/...        # upstream files, byte for byte
  manifest.json              # written last, write-once
```

The manifest lists, per unit, the S3 prefix, object count, total bytes, and
a digest over every object's relative key, size, and SHA-256. It also records
the dataset, the upstream repository and revision, and how the bytes were
obtained. `wf_autoe2e` validates the manifest digest against
`DatasetSpec.source_manifest` before reading anything, and the pack tasks
read units only from the prefixes it lists.

`wf_acquire_raw_snapshot` is the only workflow with the `hf-token` secret. It
copies each requested unit from the Hub, verifies sizes and hashes, writes
the objects with `IfNoneMatch="*"`, and writes the manifest last. Rerunning
it for the same revision and units finds the existing objects and rewrites
nothing. nuPlan already follows this pattern with
`wf_acquire_nuplan_raw_snapshot` and `nuplan_raw_snapshot_v1`; that
acquisition becomes the nuPlan branch of `wf_acquire_raw_snapshot`, and its
manifest schema is kept.

Existing data becomes snapshots without touching the Hub:

| Dataset | Snapshot | Source of the bytes |
|---|---|---|
| KITScenes train | `kitscenes/raw-snapshots/6fde0034…/` with 533 scenes | server-side copy of the `raw_uri` prefixes of Full Run `a5nk56zsbcg7g6npjfpt` from the Flyte artifacts bucket, listed in the frozen recovery manifest (1.75 TB, 14.47 million objects) |
| nuPlan | `nuplan/raw-snapshots/nuplan-v1.1-complete/nuplan-v1.1-complete-20260809/` | already exists (341 archives, 20.3 TB) |
| L2D, KITScenes val and test | none yet | the first `wf_acquire_raw_snapshot` run is the only Hub access |

### 4.3 Dataset adapters

Every dataset implements the same adapter interface
(`Platform/pipelines/autoe2e_adapters.py`):

| Member | Meaning |
|---|---|
| `published_name(plan)` | Published dataset id, e.g. `kitscenes`, `kitscenes-val`, `nuplan` |
| `read_snapshot(manifest_bytes, pack_params)` | Validates the snapshot manifest and returns a `SourcePlan`: upstream revision, unit ids with their S3 prefixes, and the pack partitions as lists of unit ids |
| `acquire_task` | Flyte task that copies one unit from upstream into a snapshot. Used only by `wf_acquire_raw_snapshot` |
| `label_task` | Flyte task that labels one unit from its snapshot prefix, or `None` if the dataset has no reasoning labels |
| `pack_task` | Flyte task that packs one partition from snapshot and label prefixes into a publication-ready directory |

| | KITScenes | nuPlan | L2D |
|---|---|---|---|
| Unit | scene | archive | episode |
| Snapshot manifest | `autoe2e_raw_snapshot_v1` | `nuplan_raw_snapshot_v1` (existing) | `autoe2e_raw_snapshot_v1` |
| `acquire_task` | HF download of one scene | `acquire_nuplan_archive` | HF download of one episode |
| `label_task` | `generate_reasoning_labels` | `None` | `generate_reasoning_labels` |
| Partition | one scene | one sensor group (43 for train) | `pack_params["partition_size"]` episodes |
| `pack_task` | `data_processing` | `pack_nuplan_snapshot_reactive_dataset` | `data_processing` |
| `pack_params` | none | `total_scenario_limit`, `samples_per_shard`, `max_rejection_fraction` | `partition_size`, `osm_graph_snapshot` |
| Split for validation | frozen holdout `kitscenes_train_dev_v4.json` | hash buckets of the log id | hash buckets |

The parsers and packers that currently fetch metadata from the Hub
(`plan_fanout_partitions` through `LeRobotDatasetMetadata`,
`Model/data_parsing/kit_scenes/source.py`, and the KITScenes test direct
pack) read the same files from the snapshot instead.

Known gaps that the adapters make explicit instead of hiding:

- nuPlan training packs are published like every other dataset. Today Ray
  reads the 43 pack directories directly from a hand-written URI list.
- The L2D Reactive contract needs OSM navigation (14 map channels), but no
  Flyte task passes `osm_graph_snapshot_path` to the packer today. The L2D
  adapter rejects a spec without `osm_graph_snapshot` until that is wired.

### 4.4 Build dataset (conditional)

`build_dataset` is one `@dynamic` task for all datasets:

1. read the `SourcePlan` from the snapshot through the adapter;
2. look up the label record of every unit in the label store;
3. run `label_task` only for units without a label record, then record what
   it produced;
4. run `pack_task` for every partition;
5. publish with `publish_dataset_partition` and
   `finalize_dataset_publication` under the allocated version, then
   `record_dataset_index` writes the by-id index and re-resolves the result
   through the same validation.

No task in this stage requests the `hf-token` secret. A unit that is not in
the snapshot fails the build with the missing unit ids; it is never fetched.

#### Label store

Labels are billed Cosmos calls and must never repeat. Their identity is
independent of the pack:

| Identity fields | Not included |
|---|---|
| dataset, upstream revision, unit id, teacher, `prompt_version`, `label_stride`, `REASONING_LABEL_POLICY_VERSION`, `UID_SCHEMA_VERSION` | parser, shard, and geometry versions |

A write-once record per unit lives at
`s3://<datasets_bucket>/<dataset>/_stages/labels/<label_id>.json`, and the
label files themselves are written under
`s3://<datasets_bucket>/<dataset>/labels/<label_id>/`. The record holds the
object count, total size, and a digest over every object's relative key,
size, and ETag. Before a build uses a record, it lists the label files and
compares the digest. A mismatch fails the execution instead of silently
re-labeling.

A parser change can change which frames exist, but it does not re-label. The
pack JOIN already treats a sample without a label as fully masked, and the
publication records `reasoning_label_count`, so lost coverage is visible
instead of silently re-billed.

The 533 KITScenes label sets from Full Run `a5nk56zsbcg7g6npjfpt` are copied
into the label store once, after each label `meta.json` is checked against
the label identity. A read-only dry run on 2026-10-20 validated all 533.

### 4.5 Resolve parent

If `parent.model_version` is set, the task downloads that version's
checkpoint and checks the SHA-256 tag on the version. If `parent.checkpoint`
is set, the task computes its SHA-256. The result is a `ParentHandle` with
the checkpoint, its SHA-256, and the parent's `stage_index`, which is 0 for
an unregistered checkpoint.

### 4.6 Train

`train_ray(dataset_handle, parent_handle, training)` is one task factory.
`gpu_profile` selects the `RayJobConfig`, Kueue queue, priority class, and
node selector from a single table, so no code outside that table names a
queue. The task reads the active Capacity Block end time itself instead of
receiving it from a CodeBuild buildspec.

With `epochs == 0` the task does not start Ray. It returns the parent
checkpoint as both `best` and `final`, which makes an evaluation-only run of
any registered model.

The task creates the MLflow run. Its outputs are the run id and the `best`
and `final` checkpoints.

### 4.7 Evaluate

Each requested suite runs for each distinct checkpoint and logs to the
training run with the prefix `eval/<role>/<suite>/`:

| Suite | Data |
|---|---|
| `INTERNAL_VAL` | validation split from the `DatasetHandle` |
| `KITSCENES_BENCHMARK` | `benchmark_dataset`, resolved through 4.1 and 4.4. The model is re-executed |
| `REACTIVE` | the existing Reactive KITScenes evaluation |

The evaluation tasks no longer take `expected_model_version`, because
registration happens after evaluation.

### 4.8 Register

The pipeline creates a `best` version of `auto-e2e-driving-policy`. It
creates a `final` version only if the final checkpoint's SHA-256 differs
from the best one. Both versions use the training run as `run_id`. Tags on
each version:

- `checkpoint_role` (`best` or `final`), checkpoint URI and SHA-256;
- `flyte_execution_id`, `dataset_id`, `dataset_manifest_sha256`;
- `parent_model_version` (empty for an unregistered parent),
  `parent_checkpoint_sha256`, and `stage_index` (parent's `stage_index`
  plus 1);
- training image digest and git commit;
- headline metrics copied from the run.

The dataset is recorded with `mlflow.log_input` on the run. Aliases are not
used. Following `parent_model_version` from any version reconstructs its
whole training chain.

### 4.9 Precompute overlays

Overlay precompute runs for the `best` version on the evaluation dataset. It
is skipped if `OVLSET` is already `ready` for the same checkpoint SHA-256
and dataset manifest SHA-256. The reasoning materialization Job becomes a
task in this stage.

## 5. Launch Plans

Only `autoe2e_smoke` is defined: a 10-scene KITScenes split, `RAY_1`, 1
epoch, `INTERNAL_VAL`. Every other run passes its specs explicitly, so the
inputs of a run are visible on its execution page.

## 6. Deletion

Everything on the model production path collapses into `wf_autoe2e`, and
acquisition into `wf_acquire_raw_snapshot`. Nothing else stays registered.

| Removed | Where its logic goes |
|---|---|
| `wf_sharded_full_run`, `wf_train_il`, `wf_train_offline_rl`, `wf_full_pipeline`, `wf_ingest_train_eval` | `wf_autoe2e`. The IL and offline RL training loops are deleted |
| `wf_train_reactive_*` (9 workflows), `wf_reactive_multistage_ray_2_canary` | `train_ray` with `gpu_profile` and `parent` |
| `wf_data_ingest`, `wf_data_processing`, `wf_generate_reasoning_labels`, `wf_create_dataset`, `wf_create_dataset_sharded`, `wf_create_kitscenes_*evaluation_sharded`, nuPlan acquire and pack workflows | snapshots (4.2), adapters (4.3), and stage 4.4 |
| `wf_evaluate_*` | stage 4.7 |
| overlay and dataset publication workflows | stages 4.4 and 4.9 |
| `wf_audit_kitscenes_target_reconstruction`, `wf_audit_kitscenes_benchmark_inventory` | validation inside 4.1 and 4.4 |
| `wf_prepare_kitscenes_paper_approximation`, `wf_export_trajectory_report` | deleted. They are not needed to produce or evaluate a model |
| NVIDIA PhysicalAI branches in `data_ingest`, `data_processing`, `generate_reasoning_labels`, and `physical_ai_av` in the data-prep image | deleted |

The CodeBuild launcher buildspecs for the removed workflows are deleted.
Tasks that no stage calls after the migration are deleted with them. After
`wf_autoe2e` is registered, the remaining old entry points are archived in
Flyte.

## 7. Out of Scope

- Making Kueue admission depend on an existing Capacity Block
  (ProvisioningRequest or an AdmissionCheck). This is needed for transparent
  capacity, but it is infrastructure work tracked separately.
- Automatic Capacity Block purchase. `Platform/README.md` keeps purchase as
  an operator action.
- Multi-node training and EFA.
