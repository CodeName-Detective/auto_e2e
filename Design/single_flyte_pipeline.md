# Design: Single Flyte Pipeline from Dataset to Registered Model

## Document Metadata

| Field | Value |
|---|---|
| Status | Draft |
| Owner | riita10069 |
| Created | 2026-10-20 |
| Replaces | The 40 `wf_*` entry points in `Platform/pipelines/` on main at `c7fc884d` |
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

This design replaces those entry points with one workflow, `wf_autoe2e`. One
execution:

1. resolves the requested dataset to a content-addressed `dataset_id`;
2. builds and publishes that dataset only if no valid publication exists;
3. trains with Ray Train, unless a trained checkpoint is supplied;
4. evaluates the best and final checkpoints;
5. registers two MLflow model versions, `best` and `final`, both from the
   training run;
6. precomputes Console overlays for the `best` version.

Smoke and full runs are two launch plans of the same workflow, not two
workflows.

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Training backend | Ray only. `train_il` and `train_offline_rl` are removed | One code path for 1, 2, 4, and 8 GPUs. Kueue already admits `ray.io/rayjob` |
| Datasets per execution | Exactly one | Multi-stage training (nuPlan, then KITScenes) chains executions through `init_checkpoint` |
| Model versions per execution | Two: `best` and `final` | `best` by the validation selector, `final` for reproducibility of the last epoch |
| Dataset skip mechanism | Published manifest lookup by `dataset_id` | Flyte cache keys include non-semantic inputs such as concurrency, and DataCatalog write failures have already forced a manual recovery manifest (#121) |

## 3. Inputs

`wf_autoe2e` takes three dataclasses so that launch plans stay readable.

```python
@dataclass
class DatasetSpec:
    dataset: Dataset                 # KITSCENES or NUPLAN
    source_revision: str             # pinned upstream revision
    split_manifest: str              # frozen scene list (S3 URI + SHA-256)
    pack_version: str                # code constant, e.g. DATASET_PACK_VERSION
    image_size: int
    reasoning: Optional[ReasoningSpec]  # teacher, prompt_version, label_stride

@dataclass
class TrainingSpec:
    gpu_profile: GpuProfile          # RAY_1, RAY_2, RAY_4, RAY_8
    init_checkpoint: Optional[FlyteFile]     # previous stage, e.g. nuPlan
    trained_checkpoint: Optional[FlyteFile]  # set to skip training
    epochs: int
    learning_rate: float
    per_rank_batch_size: int
    seed: int
    objectives: ObjectiveSpec        # trajectory, BEV, route, reasoning weights

@dataclass
class EvaluationSpec:
    suites: List[EvalSuite]          # INTERNAL_VAL, KITSCENES_BENCHMARK, REACTIVE
    benchmark_dataset: Optional[DatasetSpec]
```

Operational knobs such as `ingest_concurrency` or `label_workers` stay as
separate workflow inputs. They never enter `dataset_id`.

## 4. Stages

### 4.1 Resolve dataset

`dataset_id` is the SHA-256 of the canonical JSON of `DatasetSpec`. The
resolver task looks for
`s3://<datasets_bucket>/<dataset>/<dataset_id>/shards/manifest.json`.

A publication is valid only if all of these hold:

- the manifest exists and its SHA-256 matches the value in
  `.publication-lock.json`;
- the manifest's recorded `DatasetSpec` equals the requested one;
- every listed shard object exists with the recorded size.

If the publication is valid, the task returns a `DatasetHandle` (dataset id,
manifest URI, manifest SHA-256, shard URIs, validation split) and the build
stage is skipped through a Flyte `conditional`. If a lock exists but the
manifest does not, another execution is building the same dataset. The task
fails with a clear message instead of building a second copy.

### 4.2 Build dataset (conditional)

This stage reuses the existing tasks:

| Dataset | Tasks |
|---|---|
| KITScenes | ingest, reasoning labels, pack (`_map_dataset_partitions`), navigation-quality audit |
| nuPlan | `acquire_nuplan_archive` via `_acquire_nuplan_raw_snapshot`, scene pack |

Then it calls `publish_dataset_partition` and `finalize_dataset_publication`
under the `dataset_id` prefix. Per-partition Flyte cache stays enabled, so a
failed build resumes from finished partitions.

### 4.3 Train

`train_ray(handle, training_spec)` is one task factory. `gpu_profile` selects
the `RayJobConfig`, Kueue queue, priority class, and node selector from a
single table, so no workflow hard-codes a queue name. The task reads the
active Capacity Block end time itself instead of receiving it from a
CodeBuild buildspec.

If `trained_checkpoint` is set, training is skipped and that checkpoint is
used as both `best` and `final`.

The task creates the MLflow run. Its outputs are the run id and the `best`
and `final` checkpoints.

### 4.4 Evaluate

Each requested suite runs for both checkpoints and logs to the training run
with the prefix `eval/<role>/<suite>/`:

| Suite | Data |
|---|---|
| `INTERNAL_VAL` | validation split from the `DatasetHandle` |
| `KITSCENES_BENCHMARK` | `benchmark_dataset`, resolved through 4.1 and 4.2. The model is re-executed |
| `REACTIVE` | the existing Reactive KITScenes evaluation |

The evaluation tasks no longer take `expected_model_version`, because
registration happens after evaluation.

### 4.5 Register

The pipeline creates two versions of `auto-e2e-driving-policy`, both with
`run_id` set to the training run. Tags on each version:

- `checkpoint_role` (`best` or `final`), checkpoint URI and SHA-256;
- `flyte_execution_id`, `dataset_id`, `dataset_manifest_sha256`;
- training image digest and git commit;
- headline metrics copied from the run.

The dataset is recorded with `mlflow.log_input` on the run. Aliases are not
used.

### 4.6 Precompute overlays

Overlay precompute runs for the `best` version on the evaluation dataset. It
is skipped if `OVLSET` is already `ready` for the same checkpoint SHA-256
and dataset manifest SHA-256. The reasoning materialization Job becomes a
task in this stage.

## 5. Launch Plans

| Launch plan | Dataset | GPU profile | Epochs | Suites |
|---|---|---|---|---|
| `autoe2e_smoke` | 10-scene KITScenes split | `RAY_1` | 1 | `INTERNAL_VAL` |
| `autoe2e_kitscenes_full` | full KITScenes split | `RAY_8` | 10 | all |
| `autoe2e_nuplan_full` | full nuPlan split | `RAY_8` | 3 | `INTERNAL_VAL`, `REACTIVE` |

A nuPlan to KITScenes run is two executions. The second one receives the
first one's `best` checkpoint as `init_checkpoint`.

## 6. Migration

| Current entry points | Fate |
|---|---|
| `wf_sharded_full_run`, `wf_train_il`, `wf_train_offline_rl`, `wf_full_pipeline`, `wf_ingest_train_eval` | Removed. Replaced by `wf_autoe2e` |
| `wf_train_reactive_*` (9 workflows), `wf_reactive_multistage_ray_2_canary` | Removed. Covered by `gpu_profile` and launch plans |
| `wf_data_ingest`, `wf_data_processing`, `wf_generate_reasoning_labels`, `wf_create_dataset`, `wf_create_dataset_sharded`, `wf_create_kitscenes_*evaluation_sharded`, nuPlan acquire and pack workflows | Removed as entry points. Their tasks become stage 4.2 |
| `wf_evaluate_*` | Removed as entry points. Their tasks become stage 4.4 |
| overlay and publication workflows | Removed as entry points. Their tasks become stage 4.6 |
| `wf_audit_*`, `wf_prepare_kitscenes_paper_approximation`, `wf_export_trajectory_report` | Kept as separate tools. They are not on the model production path |

The CodeBuild launcher buildspecs for the removed workflows are deleted.
After the new workflow is registered, the removed entry points are archived
in Flyte.

## 7. Out of Scope

- Making Kueue admission depend on an existing Capacity Block
  (ProvisioningRequest or an AdmissionCheck). This is needed for transparent
  capacity, but it is infrastructure work tracked separately.
- Automatic Capacity Block purchase. `Platform/README.md` keeps purchase as
  an operator action.
- Multi-node training and EFA.

## 8. Open Questions

1. Should `split_manifest` for KITScenes be the existing frozen recovery
   manifest, or a new manifest generated from the dataset policy?
2. Does the Console need any version tag beyond those in 4.5 to find
   overlays?
3. Should the `final` version be skipped when it is the same checkpoint as
   `best`?
