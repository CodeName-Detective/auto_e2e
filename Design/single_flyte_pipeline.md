# Design: Single Flyte Pipeline from Dataset to Registered Model

## Document Metadata

| Field | Value |
|---|---|
| Status | Draft |
| Owner | riita10069 |
| Created | 2026-10-20 |
| Replaces | All 40 `wf_*` entry points in `Platform/pipelines/` on main at `c7fc884d` |
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

This design deletes every existing entry point and provides exactly one
workflow, `wf_autoe2e`, with a small, dataset-agnostic interface. One
execution:

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
| Entry points | `wf_autoe2e` only. No other workflow is registered | One interface to learn, search, and keep correct |
| Training backend | Ray only. `train_il` and `train_offline_rl` are removed | One code path for 1, 2, 4, and 8 GPUs. Kueue already admits `ray.io/rayjob` |
| Datasets per execution | Exactly one | Keeps `dataset_id`, evaluation data, and lineage unambiguous |
| Multi-stage training | `parent` input that accepts an MLflow model version or a checkpoint | Stages are chained one execution at a time, with no limit on depth |
| Model versions per execution | `best`, plus `final` only if its checkpoint SHA-256 differs from `best` | No duplicate versions for the same weights |
| Dataset skip mechanism | Published manifest lookup by `dataset_id` | Flyte cache keys include non-semantic inputs such as concurrency, and DataCatalog write failures have already forced a manual recovery manifest (#121) |
| KITScenes scene list | The frozen recovery manifest `recovery-manifests/24afc76f…a12.json` in the checkpoints bucket | Same 533 train-split scenes (source revision `6fde0034`) as the full runs since July |

## 3. Interface

```python
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
    dataset: Dataset                 # KITSCENES or NUPLAN
    source_revision: str             # pinned upstream revision
    split_manifest: ManifestRef      # S3 URI + VersionId + SHA-256
    pack_version: str                # code constant
    image_size: int
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
behind the `Dataset` value inside tasks, not in the workflow signature.

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

For KITScenes, `split_manifest` points at the frozen recovery manifest
(schema `kitscenes_recovery_manifest_v1`, 533 entries, 1 missing scene). The
resolver reads only its scene ids and `source_revision`. It does not reuse
the manifest's raw or label artifact pointers, because `pack_version` and
`reasoning` decide those. The manifest has no validation holdout, so the
existing deterministic scene-level split in the training code still selects
it.

### 4.2 Build dataset (conditional)

This stage reuses the existing tasks:

| Dataset | Tasks |
|---|---|
| KITScenes | ingest, reasoning labels, pack (`_map_dataset_partitions`), navigation-quality audit |
| nuPlan | `acquire_nuplan_archive` via `_acquire_nuplan_raw_snapshot`, scene pack |

Then it calls `publish_dataset_partition` and `finalize_dataset_publication`
under the allocated version, and `record_dataset_index` writes the by-id
index and re-resolves the result through the same validation.

#### Scene stage store

Ingest and reasoning labeling are the expensive stages: one KITScenes scene
is about 17,000 raw files and 3.3 GB, and labels are billed Cosmos calls.
Their outputs do not depend on the pack contract, so a new `pack_version`,
`SHARD_SCHEMA_VERSION`, `GEOMETRY_VERSION`, or `PARSER_VERSION` must never
repeat them. Today the ingest cache key contains `PARSER_VERSION`, so the
v2 to v3 bump on 2026-07-25 made every scene look un-ingested.

Each scene therefore gets two stage identities that are independent of the
pack:

| Stage | Identity fields | Not included |
|---|---|---|
| raw | dataset, `source_revision`, scene id | any parser, shard, or geometry version |
| labels | raw identity, teacher, `prompt_version`, `label_stride`, `REASONING_LABEL_POLICY_VERSION`, `UID_SCHEMA_VERSION` | parser, shard, and geometry versions |

A write-once record per scene and stage lives at
`s3://<datasets_bucket>/<dataset>/_stages/<stage>/<stage_id>.json`. It holds
the artifact URI, the object count, the total size, and a digest over every
object's relative key, size, and ETag. Before a build uses a record, it lists
the artifact and compares the digest. A mismatch fails the execution instead
of silently re-ingesting or re-labeling.

The build looks up both records for every scene first. It runs ingest only
for scenes without a raw record and labeling only for scenes without a label
record, then writes the records for what it produced. Pack and publication
always run from those artifacts.

A parser change can change which frames exist, but it does not re-label. The
pack JOIN already treats a sample without a label as fully masked, and the
publication records `reasoning_label_count`, so lost coverage is visible
instead of silently re-billed.

The 533 KITScenes scenes from Full Run `a5nk56zsbcg7g6npjfpt` are adopted
once, without copying, by
`Platform/scripts/adopt_kitscenes_recovery_stages.py`. It reads the
`raw_uri` and `label_uri` of each entry in the frozen recovery manifest,
checks the label `meta.json` against the label identity, and writes the stage
records. After that, the first `wf_autoe2e` KITScenes run ingests and labels
nothing.

### 4.3 Resolve parent

If `parent.model_version` is set, the task downloads that version's
checkpoint and checks the SHA-256 tag on the version. If `parent.checkpoint`
is set, the task computes its SHA-256. The result is a `ParentHandle` with
the checkpoint, its SHA-256, and the parent's `stage_index`, which is 0 for
an unregistered checkpoint.

### 4.4 Train

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

### 4.5 Evaluate

Each requested suite runs for each distinct checkpoint and logs to the
training run with the prefix `eval/<role>/<suite>/`:

| Suite | Data |
|---|---|
| `INTERNAL_VAL` | validation split from the `DatasetHandle` |
| `KITSCENES_BENCHMARK` | `benchmark_dataset`, resolved through 4.1 and 4.2. The model is re-executed |
| `REACTIVE` | the existing Reactive KITScenes evaluation |

The evaluation tasks no longer take `expected_model_version`, because
registration happens after evaluation.

### 4.6 Register

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

### 4.7 Precompute overlays

Overlay precompute runs for the `best` version on the evaluation dataset. It
is skipped if `OVLSET` is already `ready` for the same checkpoint SHA-256
and dataset manifest SHA-256. The reasoning materialization Job becomes a
task in this stage.

## 5. Launch Plans

Only `autoe2e_smoke` is defined: a 10-scene KITScenes split, `RAY_1`, 1
epoch, `INTERNAL_VAL`. Every other run passes its specs explicitly, so the
inputs of a run are visible on its execution page.

## 6. Deletion

Everything on the model production path collapses into `wf_autoe2e`. Nothing
else stays registered.

| Removed | Where its logic goes |
|---|---|
| `wf_sharded_full_run`, `wf_train_il`, `wf_train_offline_rl`, `wf_full_pipeline`, `wf_ingest_train_eval` | `wf_autoe2e`. The IL and offline RL training loops are deleted |
| `wf_train_reactive_*` (9 workflows), `wf_reactive_multistage_ray_2_canary` | `train_ray` with `gpu_profile` and `parent` |
| `wf_data_ingest`, `wf_data_processing`, `wf_generate_reasoning_labels`, `wf_create_dataset`, `wf_create_dataset_sharded`, `wf_create_kitscenes_*evaluation_sharded`, nuPlan acquire and pack workflows | stage 4.2 |
| `wf_evaluate_*` | stage 4.5 |
| overlay and dataset publication workflows | stages 4.2 and 4.7 |
| `wf_audit_kitscenes_target_reconstruction`, `wf_audit_kitscenes_benchmark_inventory` | validation inside 4.1 and 4.2 |
| `wf_prepare_kitscenes_paper_approximation`, `wf_export_trajectory_report` | deleted. They are not needed to produce or evaluate a model |

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
