# Looped Transformer Flow Distillation

![Looped Transformer Flow Distillation cover](assets/readme-cover.png)

Looped Transformer Flow Distillation is a lightweight research codebase for replacing the repeated
latent-refinement loop inside looped, recurrent-depth, and attractor-style transformer teachers.
The first target is controlled loop-dynamics replacement rather than full language-model
compression: keep the teacher backbone and language-model head fixed, train a student dynamics
module on the teacher latent trajectory, and measure whether the student endpoint preserves the
teacher-head logits.

The motivating trajectory is

```text
z_0, z_1, ..., z_K
```

where `K` is the teacher loop depth. A student is evaluated with `S` learned rollout or transition
steps. The core question is whether `S` can be smaller, cheaper, or more structured than `K` while
preserving endpoint behavior.

The current methods include local flow matching, compositional flow maps, MeanFlow-style
average-velocity objectives, Shortcut-style consistency, and DEQ-style equilibrium students. External
teacher repositories stay isolated behind narrow adapters.

## Project Scope

This repository is for trajectory-aware distillation of looped transformer computation.

In scope:

- training smaller dynamics modules that replace the teacher's recurrent loop block;
- evaluating `K` teacher steps versus `S` student steps under a fixed teacher head;
- comparing local flow matching, compositional flow maps, MeanFlow, Shortcut, and DEQ variants;
- keeping local metrics, resolved configs, and run summaries under `outputs/`;
- supporting synthetic teachers and external Attractor, Huginn, Parcae, and Ouro-style adapters.

Out of scope for the current baseline:

- claiming standalone student language-model compression unless the student head is explicitly
  trained;
- judging loop replacement by the untrained `student_head` metrics;
- vendoring external teacher repositories into this codebase.

## Features

- **Trajectory extraction** from synthetic teachers, Attractor and Huginn checkpoints, and configurable
  external teacher adapters.
- **Offline distillation** from saved shard/manifest datasets containing latent trajectories,
  tokens, masks, metadata, and optional endpoint logits.
- **Live teacher distillation** where text batches are tokenized and passed through a teacher
  during training without materializing trajectory shards.
- **Latent student models** with time and interval conditioning for velocity prediction,
  endpoint reconstruction, and rollout-based objectives.
- **Teacher-head quality evaluation** with endpoint KL, NLL/PPL deltas, top-1 agreement, and
  top-k overlap.
- **Compositional flow-map losses** for learning interval transitions that can compress a larger
  teacher depth `K` into a smaller student transition budget `S`.
- **Local reproducibility artifacts** under `outputs/`, including resolved configs, metrics,
  reports, and plot-ready CSV/JSON files.

## Repository Layout

```text
configs/                  Hydra configuration groups
src/loopdistill/data/     trajectory shards, manifests, collators, and datamodules
src/loopdistill/teachers/ teacher adapters and mock teacher implementation
src/loopdistill/models/   student flow model and DEQ wrapper
src/loopdistill/losses/   flow matching, reconstruction, MeanFlow, and Shortcut losses
src/loopdistill/metrics/  endpoint language-model quality metrics
src/loopdistill/train.py  Lightning training entrypoint
tests/                    unit and smoke tests
```

The project does not vendor Attractor, Parcae, Ouro, TorchCFM, TorchDEQ, or MeanFlow source
trees. External implementations are used through dependencies or explicit adapter paths.

## Installation

The dependency source of truth is `pyproject.toml`.

```bash
uv sync --extra dev
```

For development and validation:

```bash
uv run pytest
```

Optional GPU-specific dependencies can be installed with the `a100` extra when the target
environment supports them:

```bash
uv sync --extra dev --extra a100
```

## Quick Start

Run the local smoke workflow with the synthetic teacher:

```bash
uv run loopdistill-extract teacher=mock data=trajectory
uv run loopdistill-train data=trajectory trainer.max_epochs=1
```

By default, Hydra writes outputs to:

```text
outputs/<pipeline_name>/<run_id>/
```

Each run directory is intended to contain local metrics, a resolved configuration, and a short
run summary so results can be inspected without relying on an external service.

## Workflow

### 1. Extract Teacher Trajectories

Offline distillation starts by saving teacher trajectories to shard files and indexing them with
a JSONL manifest:

```bash
uv run loopdistill-extract \
  teacher=mock \
  data=trajectory \
  extraction.num_samples=16 \
  extraction.seq_len=24
```

The generated manifest can then be passed to training:

```bash
uv run loopdistill-train \
  paths.manifest_path=outputs/trajectories/mock/manifest.jsonl
```

### 2. Train A Student

Training is configured through Hydra groups for the teacher, data source, student, losses,
trainer, and evaluation metrics:

```bash
uv run loopdistill-train \
  data=trajectory \
  student=small \
  loss=p0 \
  trainer.max_epochs=1
```

The P0 loss combines latent flow matching, endpoint latent reconstruction, optional endpoint KL,
and a small stability regularizer. Compositional, MeanFlow, Shortcut, and DEQ modules are available
as reusable components for research runs that need those objectives.

For an Attractor-140M live-teacher baseline:

```bash
uv run loopdistill-train \
  experiment=blackwell_live_attractor140_p0_full \
  eval_quality.enabled=true \
  eval_quality.rollout_steps=8 \
  teacher.max_depth=8 \
  loss.rollout_steps=8
```

In this example, `K=8` is the teacher target depth and `S=8` is the student rollout budget.
Changing `teacher.max_depth` and `eval_quality.rollout_steps` gives K/S comparisons such as
K16/S8.

### 3. Evaluate Endpoint Quality

When endpoint logits are available, `eval_quality.enabled=true` logs language-model quality metrics
under `eval_quality/val/*` and the configured final-evaluation prefix, which defaults to
`eval_quality/test/*`. Live teacher runs can evaluate two endpoint projections:

- `teacher_head`: rolls the student to `z_K_student` and projects it through the original teacher
  `ln_f/lm_head`;
- `student_head`: projects through the student's own LM head.

The `teacher_head` path is the primary replacement test for looped-layer distillation because the
teacher backbone and LM head stay fixed and only the loop dynamics are replaced. The `student_head`
path is diagnostic unless the run explicitly trains the student head with a logit or endpoint
objective.

```bash
uv run loopdistill-train \
  experiment=blackwell_live_attractor140 \
  eval_quality.enabled=true \
  teacher.return_logits=true \
  teacher.logit_depths=final
```

## External Teachers

Attractor integration is configured through `configs/teacher/attractor.yaml`. The default remote
layout expects an external repository, checkpoint directory, and text dataset path:

```text
/workspace/external/Attractor
/workspace/external/models/attractor-140m
/workspace/datasets/fineweb-edu
```

Example extraction with an Attractor teacher:

```bash
uv run loopdistill-extract \
  teacher=attractor \
  experiment=blackwell_attractor140 \
  extraction.num_samples=2 \
  extraction.batch_size=1 \
  extraction.seq_len=16 \
  'extraction.depths=[0,1,2]' \
  teacher.max_depth=2
```

Live teacher distillation is available through the `live_text` datamodule and the
`blackwell_live_attractor140` experiment config:

```bash
uv run loopdistill-train \
  experiment=blackwell_live_attractor140 \
  output_dir=outputs/live_distill/$(date -u +%Y%m%d_%H%M%S)
```

Parcae-style external repositories can be connected through `ExternalRepoTeacher` by providing
the repository path, module path, class name, checkpoint, and tokenizer configuration. Ouro
models are configured through the Hugging Face-based `OuroTeacher`.

### Huginn Recurrent Teacher

Huginn integration is configured through `configs/teacher/huginn.yaml`. The intended remote layout
keeps the official repository, model weights, and sampled pretraining data outside this repo:

```text
/workspace/external/recurrent-pretraining
/workspace/external/models/huginn-0125
/workspace/datasets/huginn_train_random_50gib_20260618_1600
```

The Huginn adapter uses the official Hugging Face interface with `trust_remote_code=true` and
projects student endpoints through Huginn's original teacher head. The recurrent student templates
train on already-tokenized Huginn parquet rows and keep the full available context:

```bash
uv run loopdistill-train \
  experiment=blackwell_live_huginn_recurrent_k16s8 \
  run_id=live_huginn_recurrent_k16s8_<unique_id> \
  output_dir=outputs/live_distill/live_huginn_recurrent_k16s8_<unique_id>
```

For a cheaper student rollout budget, use:

```bash
uv run loopdistill-train \
  experiment=blackwell_live_huginn_recurrent_k16s4 \
  run_id=live_huginn_recurrent_k16s4_<unique_id> \
  output_dir=outputs/live_distill/live_huginn_recurrent_k16s4_<unique_id>
```

These configs use `input_length: null`, so the data loader does not impose a sequence-length cap.
Rows with 4097 Huginn tokens are passed as 4096-token next-token inputs by dropping the last target
token.

## Configuration

The main Hydra config is `configs/config.yaml`. Common override points include:

- `teacher=<name>`: teacher adapter (`mock`, `attractor`, `huginn`, `ouro`, `parcae`)
- `data=<name>`: offline trajectory data or live text batches
- `student=<name>`: student model size and architecture
- `loss=<name>`: distillation objective
- `trainer.*`: Lightning trainer settings
- `paths.manifest_path`: JSONL manifest for offline trajectory training
- `output_dir`: destination for run artifacts

Hydra resolves every run configuration into the output directory, making command-line overrides
auditable after the run finishes.

## Outputs

Runs should write structured artifacts that are easy to compare across experiments:

```text
outputs/<pipeline>/<run_id>/
  artifacts/
  metrics/
  plots/
  reports/
```

Typical metrics include training loss, validation loss, endpoint KL, NLL/PPL deltas, top-1
agreement, top-k overlap, rollout reconstruction error, and fixed-point or stability diagnostics
when those objectives are enabled.

### Validation And Final Evaluation Semantics

Training-time validation and language-model quality evaluation are intentionally separated:

- `val/*` metrics are distillation losses on a held-out validation split. For Huginn recurrent
  runs, this is a cheap file-level holdout from the sampled Huginn parquet data.
- `eval_quality/val/*` metrics roll the student endpoint forward and compare its teacher-head
  logits against the teacher endpoint on the same validation batches. Huginn recurrent configs
  default to `every_n_epochs=1`, and the cadence can be changed with
  `HUGINN_QUALITY_EVERY_N_EPOCHS=<n>`.
- Huginn recurrent configs keep the old parquet test loader only as an in-distribution sanity
  check. Its metrics are written under `in_distribution_test/*` and
  `eval_quality/in_distribution_test/*`, with the JSON summary at
  `metrics/in_distribution_test.json`.

This in-distribution check is not a reasoning benchmark. Scientific claims about reasoning should
come from a separate benchmark pass, for example `arc_challenge`, `commonsense_qa`, or other CORE
tasks, comparing the Huginn teacher against the trained student endpoint under the same prompt and
teacher-head projection policy.

## Development

Run the test suite before changing shared data, loss, or training code:

```bash
uv run pytest
```

The tests cover manifest parsing, shard roundtrips, mask-aware losses, quality metrics, and
Lightning smoke training. New objectives should include a small deterministic test before being
used with a real teacher checkpoint.
