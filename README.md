# Looped Transformer Flow Distillation

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

## Current Scope

- P0 implemented: shard/manifest dataset, mock teacher extraction, student latent transformer,
  FM/endpoint/logit/reconstruction/stability losses, Hydra/Lightning training.
- P0 real smoke implemented: Attractor 140M teacher adapter, FineWeb-Edu text extraction, and
  Blackwell smoke configs.
- P1 implemented as reusable modules: MeanFlow JVP loss, Shortcut consistency loss, TorchDEQ
  wrapper.
- Parcae and Ouro adapters are still thin non-vendored integration points; they fail with explicit
  setup errors unless the corresponding external model path/checkpoint is configured.
