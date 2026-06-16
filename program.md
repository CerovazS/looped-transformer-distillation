# Looped Transformer Flow Distillation - Program

## Goal

Build a lightweight, local distillation suite for Solve The Loop / Attractor-style looped transformers. The current P0 path distills only the recurrent loop dynamics: the teacher backbone and original LM head stay available, while a small `StudentFlowModel` learns to map loop latents with flow-matching-style supervision.

This file is the handoff document for agents joining the project. It should stay forward-looking and operational: current state, active experiments, useful entry points, and next decisions. It is not a timestamped journal.

## Current Repo State

- GitHub repo: `CerovazS/looped-transformer-distillation`, branch `main`.
- Local HEAD at last update: `29ae085` (`Pin torch for CUDA 12.8 pods`).
- Recent important commits:
  - `076e0a0 Add teacher-head quality evaluation`
  - `29ae085 Pin torch for CUDA 12.8 pods`
- Stack: `uv` + Hydra + Lightning + Rich logging.
- Required output layout: `outputs/<pipeline>/<run_id>/{artifacts,metrics,plots,reports}`.
- Every run must use a unique `run_id`; never overwrite old metrics, checkpoints, logs, or reports.
- Git identity before commits/pushes must be:
  - `Luca Cerovaz <204510867+CerovazS@users.noreply.github.com>`
  - no agent attribution in commits, tags, PRs, or generated text.

## Scientific Setup

The current baseline is not a full LM compression. It replaces the teacher's looped latent dynamics and then evaluates the resulting final latent through the original teacher head.

- Teacher primary target: Attractor `140m`.
- Attractor `140m` config:
  - total params: `168,053,760`
  - backbone layers: `7`
  - loop/fixed-point blocks: `1`
  - hidden size: `1024`
  - MLP intermediate size: `4096`
  - loop block params: `12,587,008`
  - `max_iter=64`, `min_iter=6`
- Current student config `blackwell_live_attractor140_p0_full`:
  - total params: about `20.1M`
  - dynamics-only params excluding token embedding and logit head: about `3.29M`
  - no-logit params: about `11.68M`
- Main metric path for loop replacement:
  - `tokens -> teacher backbone -> z_0`
  - `z_0 -> student rollout/flow -> z_K_student`
  - `z_K_student -> teacher.transformer.ln_f/lm_head -> logits_student`
  - compare against `z_K_teacher -> teacher head -> logits_teacher`
- `student_head` metrics are diagnostic only unless `endpoint_kl_weight` or explicit head training is enabled. Current useful quality signal is `teacher_head`.

## Implemented Capabilities

- Offline trajectory extraction to Torch shards and `.jsonl` manifests.
- Live teacher training mode that avoids materializing all intermediate latents.
- Attractor adapter without vendoring external code.
- P0 losses:
  - linear flow matching on teacher latent pairs
  - latent reconstruction after rollout
  - stability/fixed-point residual penalty
  - endpoint KL available but currently off in the main baseline
- P1 building blocks present but not yet the main baseline:
  - MeanFlow/JVP loss
  - Shortcut/compositional consistency
  - TorchDEQ wrapper
- Quality evaluation:
  - `KL(logits_student_K, logits_teacher_K)`
  - NLL teacher/student/delta
  - PPL teacher/student/delta
  - top-1 agreement
  - top-k overlap
  - projections: `student_head`, `teacher_head`

## Useful Entry Points

- Training CLI: `loopdistill-train`
  - module: `src/loopdistill/train.py`
- Offline trajectory extraction CLI: `loopdistill-extract`
  - module: `src/loopdistill/extract_trajectories.py`
- Token shard creation CLI: `loopdistill-tokenize`
  - module: `src/loopdistill/tokenize_text.py`
- Token shard data path:
  - `src/loopdistill/data/token_shards.py`
  - `src/loopdistill/data/datamodule.py`
- Attractor teacher adapter:
  - `src/loopdistill/teachers/attractor.py`
- Student model:
  - `src/loopdistill/models/student.py`
- P0 losses:
  - `src/loopdistill/losses/p0.py`
- Shortcut and MeanFlow:
  - `src/loopdistill/losses/shortcut.py`
  - `src/loopdistill/losses/meanflow.py`
- Quality metrics:
  - `src/loopdistill/metrics/quality.py`
- Main live baseline config:
  - `configs/experiment/blackwell_live_attractor140_p0_full.yaml`
- Data config:
  - `configs/data/token_shards.yaml`
- Teacher config:
  - `configs/teacher/attractor.yaml`
- Eval quality config:
  - `configs/eval_quality/default.yaml`

## Local Verification

Last known local smoke after the Torch pin:

```bash
uv run pytest tests/test_quality_metrics.py tests/test_live_training.py tests/test_token_shards.py
```

Expected result from last run: `13 passed`.

## Remote RTX 5090 Handoff

This is a separate single-GPU pod used for sequential baselines. It is not the earlier 2x Blackwell pod.

- SSH:
  - `ssh root@157.157.221.30 -p 57496 -i ~/.ssh/id_ed25519`
  - proxy alternative: `ssh -tt cfnijhpgb6n3e7-64411fe1@ssh.runpod.io -i ~/.ssh/id_ed25519`
- Remote repo:
  - `/workspace/looped-transformer-distillation`
- Remote Attractor repo:
  - `/workspace/external/Attractor`
- Remote models:
  - `/workspace/external/models/attractor-140m`
  - `/workspace/external/models/attractor-370m`
  - `/workspace/external/models/attractor-770m`
- Active tmux session at last parent-thread snapshot:
  - `loopdistill_seq_5090_20260616_200258`
- Sequential run script:
  - `/tmp/run_loopdistill_seq_5090.sh`
- Logs:
  - `/workspace/loopdistill_logs`

### RTX 5090 Environment Workaround

The RTX 5090 pod has system Torch `2.8.0+cu128`. Torch `2.12.0` failed with the CUDA 12.8 driver, so the repo pins `torch==2.8.0`.

Use a system-site-packages venv and call Python directly:

```bash
cd /workspace/looped-transformer-distillation
uv venv /tmp/loopdistill-venv --system-site-packages --python /usr/local/bin/python

UV_PROJECT_ENVIRONMENT=/tmp/loopdistill-venv uv sync --extra dev --inexact \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package triton \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-cuda-runtime-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cufile-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package nvidia-nvjitlink-cu12 \
  --no-install-package nvidia-nvtx-cu12
```

Then run with:

```bash
export PYTHONPATH=src
/tmp/loopdistill-venv/bin/python -m loopdistill.train ...
```

Do not use `uv run` on this pod after the workaround; it may try to install skipped CUDA wheel packages again.

## Data Assets

Large token shard root on the RTX 5090 pod:

```text
/workspace/loopdistill_tokens/fineweb_edu_attractor140_s512_n262144_v8192_t8192
```

Shard status:

- `train.pt`: shape `(195799, 512)`, int32 tokens, bool attention mask.
- `val.pt`: shape `(8192, 512)`, int32 tokens, bool attention mask.
- `test.pt`: shape `(8192, 512)`, int32 tokens, bool attention mask.
- Verified split hygiene:
  - zero duplicates within each split
  - zero row-hash overlap between train/val/test

Smaller shard also exists for quick checks:

```text
/workspace/loopdistill_tokens/fineweb_edu_attractor140_s512_n32768_v2048_t2048
```

## Active Experiments

These were launched from the parent thread on the RTX 5090 pod. Treat this section as last-known state and refresh it before making decisions.

### Run 1: K8 Target, Student 8 Steps

- Run id: `rtx5090_p0_attractor140_k8s8_s512_e15_20260616_200258`
- Status: completed in the parent thread.
- Output:
  - `/workspace/looped-transformer-distillation/outputs/live_distill/rtx5090_p0_attractor140_k8s8_s512_e15_20260616_200258`
- Setup:
  - target depth `K=8`
  - student rollout steps `8`
  - `seq_len=512`
  - batch size `16`
  - `15` epochs
  - train limit `1024` batches
  - val limit `64` batches
  - test limit `128` batches
- Final known test metrics:
  - `test/loss`: `138.39463806152344`
  - `test/loss_fm`: `110.99850463867188`
  - `test/loss_latent_reconstruction`: `27.39617347717285`
  - `eval_quality/test/teacher_head/kl_student_teacher`: `0.00040682722465135157`
  - `eval_quality/test/teacher_head/nll_teacher`: `3.1544244289398193`
  - `eval_quality/test/teacher_head/nll_student`: `3.1545722484588623`
  - `eval_quality/test/teacher_head/nll_delta`: `0.0001477748155593872`
  - `eval_quality/test/teacher_head/ppl_teacher`: `23.96455192565918`
  - `eval_quality/test/teacher_head/ppl_student`: `23.968168258666992`
  - `eval_quality/test/teacher_head/ppl_delta`: `0.003619670867919922`
  - `eval_quality/test/teacher_head/top1_agreement`: `0.9877872467041016`
  - `eval_quality/test/teacher_head/top5_overlap`: `0.9875877499580383`
- Interpretation:
  - The teacher-head replacement path works very well for K8/8.
  - The `student_head` path remains bad because the student LM head is not the trained objective.

### Run 2: K16 Target, Student 16 Steps

- Run id: `rtx5090_p0_attractor140_k16s16_s512_e15_20260616_213245`
- Status at last parent-thread snapshot: in progress.
- Setup:
  - target depth `K=16`
  - student rollout steps `16`
  - `seq_len=512`
  - batch size `8`
  - `15` epochs
  - train limit `512` batches
  - val limit `64` batches
  - test limit `128` batches
- Last known epoch 3 metrics:
  - train loss: `299.9974`
  - val loss: `235.0574`
  - `eval_quality/val/teacher_head/kl_student_teacher`: `0.0006543`
  - `eval_quality/val/teacher_head/nll_teacher`: `3.131119`
  - `eval_quality/val/teacher_head/nll_student`: `3.131421`
  - `eval_quality/val/teacher_head/nll_delta`: `0.000302`
  - `eval_quality/val/teacher_head/ppl_delta`: `0.00744`
  - `eval_quality/val/teacher_head/top1_agreement`: `0.9834785`
  - `eval_quality/val/teacher_head/top5_overlap`: `0.9831184`
- Trend:
  - improving from epoch 0; not yet as good as K8/8 at the last snapshot.

### Run 3: K16 Target, Student 8 Steps

- Run id pattern: `rtx5090_p0_attractor140_k16s8_s512_e15_<timestamp>`
- Status at last parent-thread snapshot: planned third run in the same tmux script, likely not started yet.
- Setup:
  - target depth `K=16`
  - student rollout steps `8`
  - `seq_len=512`
  - batch size `8`
  - `15` epochs
  - train limit `512` batches
  - val limit `64` batches
  - test limit `128` batches
- Scientific purpose:
  - test actual inference-step compression: match a deeper teacher endpoint with fewer student integration steps.

## Monitoring Commands

Read the current tmux state:

```bash
ssh root@157.157.221.30 -p 57496 -i ~/.ssh/id_ed25519 \
  'tmux capture-pane -pt loopdistill_seq_5090_20260616_200258 -S -80'
```

Check GPU:

```bash
ssh root@157.157.221.30 -p 57496 -i ~/.ssh/id_ed25519 \
  'nvidia-smi'
```

List live distillation outputs:

```bash
ssh root@157.157.221.30 -p 57496 -i ~/.ssh/id_ed25519 \
  'ls -lt /workspace/looped-transformer-distillation/outputs/live_distill | head'
```

Inspect final test metrics for a run:

```bash
ssh root@157.157.221.30 -p 57496 -i ~/.ssh/id_ed25519 \
  'python -m json.tool /workspace/looped-transformer-distillation/outputs/live_distill/<run_id>/metrics/test.json | head -120'
```

## Useful Training Command Pattern

```bash
cd /workspace/looped-transformer-distillation
export PYTHONPATH=src
PY=/tmp/loopdistill-venv/bin/python
DATA_ROOT=/workspace/loopdistill_tokens/fineweb_edu_attractor140_s512_n262144_v8192_t8192

"$PY" -m loopdistill.train \
  experiment=blackwell_live_attractor140_p0_full \
  data.train_path=${DATA_ROOT}/train.pt \
  data.val_path=${DATA_ROOT}/val.pt \
  data.test_path=${DATA_ROOT}/test.pt \
  data.num_workers=2 \
  data.pin_memory=true \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.max_epochs=15 \
  trainer.limit_val_batches=64 \
  trainer.limit_test_batches=128 \
  trainer.enable_checkpointing=true \
  trainer.log_every_n_steps=25 \
  eval_quality.every_n_epochs=1 \
  eval_quality.run_on_test=true \
  teacher.return_logits=false \
  run_id=<unique_run_id> \
  output_dir=outputs/live_distill/<unique_run_id> \
  data.batch_size=<16_or_8> \
  trainer.limit_train_batches=<1024_or_512> \
  'live.depths=[0,1,2,3,4,5,6,7,8]' \
  teacher.max_depth=8 \
  loss.rollout_steps=8 \
  eval_quality.rollout_steps=8
```

For K16 use depths `[0,...,16]`, `teacher.max_depth=16`, and rollout steps `16` or `8` depending on the experiment.

## Next Decisions

- Refresh the RTX 5090 run state before launching anything new.
- Compare K8/8, K16/16, and K16/8 using the `teacher_head` metrics, not `student_head`.
- If K16/8 degrades materially, prioritize Shortcut/compositional loss before MeanFlow; it directly targets fewer-step composition.
- Decide whether to train a student LM head. If yes, turn endpoint/logit KL back on and judge `student_head`; otherwise keep using `teacher_head` as the primary replacement metric.
- Add plot generation for:
  - `plots/loss_curves.png`
  - `plots/nfe_quality.png`
  Current local metrics are machine-readable, but the output standard expects at least one visual artifact for non-trivial runs.
- Consider adding a clean `data.max_train_samples` or shard subsetting option rather than relying only on `trainer.limit_train_batches` for small baselines.
- After the three RTX 5090 baselines finish, the next scientific branch should be one of:
  - longer K16 training if K16/16 is still improving
  - Shortcut/compositional K16 target with 8 student steps
  - MeanFlow/iMF only after the supervised flow baseline is stable

## Do Not Break

- Do not vendor Attractor, Parcae, or Ouro into this repo.
- Do not materialize all intermediate activations unless the run explicitly needs offline shards.
- Do not judge loop replacement quality from the untrained `student_head`.
- Do not run new experiments into an existing output directory.
- Do not use `uv pip install`; use `uv add` for dependency changes.
- Do not use `uv run` on the RTX 5090 workaround environment.
- Always pass `HF_TOKEN` explicitly when using gated HuggingFace assets.
- Keep `program.md` forward-looking; durable experiment results belong in run outputs and Flywheel nodes.
