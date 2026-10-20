# How to Use MLflow

Every `wf_autoe2e` run lands in the experiment `auto-e2e`. Read a checkpoint
on its **LoggedModel page** (Experiment → Models → the model), not on a run:
MLflow 3 shows params, dataset-tied metrics, datasets, runs, versions, and
artifacts there.

## What is where

| Object | Holds |
|---|---|
| Training run | training curves, system metrics, the training dataset input, the parent model input, and the produced models as outputs |
| Evaluation run | the evaluated model and the evaluation dataset as inputs |
| LoggedModel (`best`, `final` if different, or `imported`) | params with every lineage digest and the flattened specs; metrics measured on named datasets; artifacts `lineage/checkpoint.json`, `lineage/datasets/<dataset_id>/{dataset.json,units.jsonl}`, `evaluations/<suite>@<dataset>/<dataset_id>.json`; a generated description linking the parent |
| Registry version of `auto-e2e-driving-policy` | `source = models:/<model_id>` plus the identity tags (`checkpoint_sha256`, `lineage_sha256`, `dataset_id`, `stage_index`, …) for filtering |

Params, not tags, carry the digests, because the LoggedModel page does not
display tags. The OSS registry has no `model_id` column, so a version reaches
its LoggedModel only through its `source`.

## Answering common questions

| Question | Where |
|---|---|
| What data trained this checkpoint? | params `dataset_id`, `dataset.*`; artifact `lineage/datasets/<dataset_id>/dataset.json` |
| Which raw bytes and labels? | `dataset.json` `raw` and `labels`; `units.jsonl` lists every unit's tar digest and label record |
| What was it initialized from? | param `parent_model_id` and the description link, recursively |
| How did it score, and on what? | the Metrics table: each row names its dataset (`name/version`, digest) and the run that measured it |
| Is the chain intact? | every `wf_autoe2e` run ends with `verify_lineage`; it can also be run by hand from `Platform/pipelines/autoe2e_verify.py` |

## Rules

- One set of weights gets one LoggedModel and one version, across all runs.
  Registration finds an existing model by `checkpoint_sha256`.
- Evaluations are added, never rewritten. A later evaluation on another
  dataset version becomes another metric row with its own dataset.
- An evaluation is recorded only if the model was re-executed on the data.
- Clean up probes by id. `mlflow gc` without `--run-ids` permanently deletes
  every soft-deleted run on the server, not only one experiment.
