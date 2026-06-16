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

## Current Scope

- P0 implemented: shard/manifest dataset, mock teacher extraction, student latent transformer,
  FM/endpoint/logit/reconstruction/stability losses, Hydra/Lightning training.
- P1 implemented as reusable modules: MeanFlow JVP loss, Shortcut consistency loss, TorchDEQ
  wrapper.
- Attractor, Parcae, and Ouro adapters are thin non-vendored integration points; they fail with
  explicit setup errors unless the corresponding external model path/checkpoint is configured.
