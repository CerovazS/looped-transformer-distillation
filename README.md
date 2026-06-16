# Looped Transformer Flow Distillation

![Looped Transformer Flow Distillation cover](assets/readme-cover.png)

Local, lightweight training suite for distilling looped transformer latent trajectories with
flow matching, endpoint/logit distillation, latent reconstruction, stability regularization,
and experimental MeanFlow/Shortcut/DEQ objectives.

The project intentionally avoids vendoring Attractor, Parcae, Ouro, TorchCFM, TorchDEQ, or
MeanFlow repositories. External code is accessed through narrow adapters or package
dependencies.

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run loopdistill-extract teacher=mock data=trajectory
uv run loopdistill-train data=trajectory trainer.max_epochs=1
```

Outputs are written under `outputs/<pipeline>/<run_id>/` and include local metrics, resolved
configuration, and a run summary.

## Attractor/FineWeb Smoke

On the Blackwell RunPod with Attractor and FineWeb-Edu under `/workspace`:

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

The default Attractor config is read-only and expects:

- `/workspace/external/Attractor`
- `/workspace/external/models/attractor-140m`
- `/workspace/datasets/fineweb-edu`

For live teacher training without writing latent shards:

```bash
uv run loopdistill-train \
  experiment=blackwell_live_attractor140 \
  output_dir=outputs/live_distill/live_smoke_$(date -u +%Y%m%d_%H%M%S)
```

Live mode reads token batches from FineWeb-Edu, calls the teacher under `torch.no_grad()`, and
optimizes only `student.parameters()`.
FineWeb-Edu `sample-10BT` only exposes a `train` split, so the live config uses non-overlapping
HF slice expressions for train/validation/test by default.

When `eval_quality.enabled=true`, validation/test also compute endpoint language-model metrics:
KL between final student and teacher logits, teacher/student NLL and PPL deltas, top-1 agreement,
and top-k overlap. These metrics are logged under `eval_quality/val/*` and
`eval_quality/test/*`; the training objective remains FM/reconstruction/stability unless the loss
config explicitly enables endpoint KL.

## Current Scope

- P0 implemented: shard/manifest dataset, mock teacher extraction, student latent transformer,
  FM/endpoint/logit/reconstruction/stability losses, Hydra/Lightning training.
- P0 real smoke implemented: Attractor 140M teacher adapter, FineWeb-Edu text extraction, and
  Blackwell smoke configs.
- P0 live smoke implemented: token-only FineWeb datamodule plus live teacher trajectory generation
  inside the Lightning training step.
- P0 quality eval implemented: final-logit KL, NLL/PPL teacher-student-delta, top-1 agreement, and
  top-k overlap on held-out validation/test slices.
- P1 implemented as reusable modules: MeanFlow JVP loss, Shortcut consistency loss, TorchDEQ
  wrapper.
- Parcae and Ouro adapters are still thin non-vendored integration points; they fail with explicit
  setup errors unless the corresponding external model path/checkpoint is configured.
