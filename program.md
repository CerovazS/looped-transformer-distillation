# Piano Di Implementazione: Looped Transformer Flow Distillation Suite

## Sintesi

Costruire una repo locale leggera, non un fork, in `src/loopdistill/`, con stack `uv` + Hydra + Lightning. La suite deve generare traiettorie teacher da Attractor, Parcae e Ouro, salvarle in shard Torch, e addestrare studenti piccoli solo sui latenti loopati con loss FM, endpoint/logit, ricostruzione, stabilità, poi MeanFlow/Shortcut/FlowMap/DEQ.

Riferimenti tecnici usati: [Flow Maps di Sander Dieleman](https://sander.ai/2026/05/06/flow-maps.html), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [TorchDEQ](https://github.com/locuslab/torchdeq), [Gsunshine/meanflow](https://github.com/Gsunshine/meanflow), [Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow), [Lyy-iiis/pMF](https://github.com/Lyy-iiis/pMF), [Attractor](https://github.com/jacobfa/Attractor), [Parcae](https://github.com/sandyresearch/parcae), [Ouro HF collection](https://huggingface.co/collections/ByteDance/ouro).

## Stato Attuale

- [x] P0.1 repo skeleton: `uv`, Hydra, Lightning, Rich logging, output standard e smoke tests locali.
  Postmortem: la suite gira end-to-end con teacher mock e `uv run pytest`.
- [x] P0 dataset/shard core: `TrajectoryRecord`, shard `.pt`, manifest `.jsonl`, collator mask-aware e `TrajectoryDataModule`.
  Postmortem: roundtrip e Lightning smoke test coprono shape `[B,K+1,L,D]` e padding variabile.
- [x] P0 student/loss core: `StudentFlowModel`, FM lineare, endpoint KL opzionale, latent reconstruction e stability.
  Postmortem: scientificamente ancora mock-only, ma la forma del training loop e dei logging file e' pronta.
- [x] P1 modules iniziali: MeanFlow JVP, Shortcut consistency e wrapper TorchDEQ come moduli riusabili.
  Postmortem: presenti come building block, non ancora collegati a run scientifici.
- [x] P0.2 smoke reale: estrazione traiettorie Attractor da FineWeb-Edu su Blackwell.
  Postmortem: `attractor-140m` produce shard reali con latenti `z_0...z_K`; smoke verificato su 2 sample, `seq_len=16`, `K=2`.
- [x] P0.3 smoke reale: training baseline su traiettorie Attractor.
  Postmortem: training GPU 1 epoca/2 batch completato con `FM + reconstruction + stability`; metriche locali scritte.
- [x] P0.5 smoke live teacher: training senza shard latenti, con Attractor chiamato nel training step.
  Postmortem: live smoke su Blackwell completato; il teacher non entra nell'optimizer, output solo in `outputs/live_distill/...`.
- [x] P0.6 eval quality endpoint: metriche KL/NLL/PPL/top-k su logits finali teacher/student.
  Postmortem: `eval_quality` e' separato dalla loss; validation resta leggera, quality gira ogni N epoch o in test quando abilitata.
- [ ] P0.7 baseline piccola riproducibile: estrazione offline o live su 1k/128 token, `K=8`, training breve e report.
  Mancante: run non-smoke con subset sufficiente, metriche confrontabili e curve.

## Ambiente Blackwell Attivo

- Pod RunPod: `runpod-blackwell-stl` / `runpod-stl`.
- GPU: `2x NVIDIA RTX PRO 6000 Blackwell Server Edition`, circa 94 GiB ciascuna.
- Repo remota: `/workspace/looped-transformer-distillation`.
- Repo Attractor remota: `/workspace/external/Attractor`.
- Checkpoint remoti: `/workspace/external/models/attractor-140m`, `/workspace/external/models/attractor-370m`, `/workspace/external/models/attractor-770m`.
- FineWeb-Edu remoto: `/workspace/datasets/fineweb-edu` e cache HF sotto `/workspace/.cache/huggingface`.

## Checklist Attiva

- [x] Implementare `AttractorTeacher` non-vendored, con caricamento locale da `checkpoint_dir`.
  Verifica: smoke remoto completato su `attractor-140m` con latenti reali.
- [x] Estendere `loopdistill-extract` per dataset testuali reali.
  Verifica: estrazione FineWeb-Edu in shard `.pt` senza sovrascrivere manifest.
- [x] Aggiungere config Hydra remote/Blackwell per Attractor 140M.
  Verifica: `experiment=blackwell_attractor140` genera manifest reale su `/workspace`.
- [x] Allenare baseline minima sullo shard reale.
  Verifica: `train.csv`, `val.csv`, `test.json` e summary scritti; loss non-NaN su 2 batch.
- [x] Aggiungere training live teacher.
  Verifica: `experiment=blackwell_live_attractor140` completa fit/test su batch token-only e non scrive shard latenti.
- [x] Aggiungere metriche qualità per uso effettivo dello student.
  Verifica: test locali coprono `KL(logits_student_K, logits_teacher_K)`, NLL/PPL teacher-student-delta, top1 agreement e top-k overlap.
- [ ] Scalare offline o live smoke a baseline P0 piccola.
  Verifica: 1k sample, `seq_len=128`, `K=8`, run_id unico, report locale e metriche stabili.
- [ ] Solo dopo la baseline supervisionata: MeanFlow/Shortcut/DEQ.
  Verifica: ogni nuova loss deve avere test sintetico prima del run su teacher reale.

## Decisioni Correnti

- FineWeb-Edu basta per P0: il primo run usa un subset piccolo, non tutto il corpus.
- Il primo teacher e' `attractor-140m`; `370m/770m` restano P1 per costo e storage.
- Per P0 non si salvano full logits per ogni depth: sono troppo grandi. Si parte con latenti, residual, solver iters; KL endpoint resta opzionale.
- Per eval quality si richiedono solo logits finali `K`: `teacher.return_logits=true` e `teacher.logit_depths=final`. Questo evita di salvare full logits a ogni depth.
- FineWeb-Edu `sample-10BT` espone solo `train`; live validation/test usano slicing HF non sovrapposto (`train[:98%]`, `train[98%:99%]`, `train[99%:]`) invece di riusare lo stesso split.
- La traiettoria Attractor P0 puo' essere ricostruita richiamando il solver con `max_iter_override=k` per ogni depth richiesto. E' piu' lenta di un hook interno o di `return_trajectory=True`, ma evita patch alla repo esterna.
- Le sequenze P0 sono fixed length dopo tokenizzazione/troncamento; l'attention mask resta nel formato shard, ma Attractor ignora il padding mask nel backend causale corrente.
- Offline e live restano due path supportati: offline per debug/riproducibilita' e live per non materializzare tutti i latenti. In live mode il teacher e' chiamato in `torch.no_grad()`, i latenti sono clonati prima della loss, e l'optimizer usa solo `student.parameters()`.
- Le metriche `val/*` misurano la loss di distillazione frequente; le metriche `eval_quality/val/*` e `eval_quality/test/*` misurano l'uso effettivo dello student come language model.

## Comandi Verificati Su Blackwell

Smoke estrazione:

```bash
cd /workspace/looped-transformer-distillation
RUN_ID=smoke_attractor_$(date -u +%Y%m%d_%H%M%S)
uv run loopdistill-extract \
  teacher=attractor \
  experiment=blackwell_attractor140 \
  run_id=$RUN_ID \
  paths.manifest_path=outputs/trajectories/$RUN_ID/manifest.jsonl \
  extraction.num_samples=2 \
  extraction.batch_size=1 \
  extraction.seq_len=16 \
  'extraction.depths=[0,1,2]' \
  teacher.max_depth=2
```

Smoke training:

```bash
cd /workspace/looped-transformer-distillation
MANIFEST=outputs/trajectories/<smoke_run_id>/manifest.jsonl
RUN_ID=train_smoke_attractor_$(date -u +%Y%m%d_%H%M%S)
uv run loopdistill-train \
  experiment=blackwell_attractor140 \
  paths.manifest_path=$MANIFEST \
  output_dir=outputs/loopdistill/$RUN_ID \
  data.batch_size=1 \
  student.hidden_dim=128 \
  student.num_layers=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1 \
  trainer.enable_checkpointing=false
```

Smoke live teacher:

```bash
cd /workspace/looped-transformer-distillation
RUN_ID=live_smoke_attractor_$(date -u +%Y%m%d_%H%M%S)
uv run loopdistill-train \
  experiment=blackwell_live_attractor140 \
  run_id=$RUN_ID \
  output_dir=outputs/live_distill/$RUN_ID
```

Nota: se FineWeb-Edu non e' ancora materializzato nella cache Arrow di `datasets`, il primo live run puo' spendere circa 1-2 minuti a generare lo split `sample-10BT` dalla cache parquet. I run successivi riusano la cache.

## Architettura Della Repo

- `pyproject.toml`: gestito solo con `uv`; dipendenze P0: `torch`, `lightning`, `hydra-core`, `omegaconf`, `transformers`, `datasets`, `safetensors`, `torchmetrics`, `rich`, `jsonlines`, `numpy`, `einops`, `torchcfm`, `torchdiffeq`, `torchdeq`. Extra A100: `flash-attn`/FA2 dove compatibile.
- `configs/`: Hydra con gruppi `teacher`, `student`, `data`, `loss`, `trainer`, `experiment`. Ogni oggetto istanziabile usa `_target_`.
- `src/loopdistill/data/`: manifest, shard loader, collator mask-aware, Lightning `TrajectoryDataModule`.
- `src/loopdistill/teachers/`: adapter read-only per teacher esterni, senza vendoring: `AttractorTeacher`, `ParcaeTeacher`, `OuroTeacher`, poi `LoopFormerTeacher`.
- `src/loopdistill/models/`: `StudentFlowModel`, time/interval embeddings, latent transformer block, heads `velocity`, `flow_map`, opzionale `logit_head`.
- `src/loopdistill/losses/`: FM lineare P0, endpoint/logit KL, latent reconstruction, stability, poi MeanFlow, Shortcut, FlowMap PSD/LSD, C-DEQ.
- `src/loopdistill/metrics/`: `QualityEvaluator` per KL endpoint, NLL/PPL teacher-student-delta, top1 agreement e top-k overlap.
- `src/loopdistill/train.py`: entrypoint Hydra + Lightning.
- `src/loopdistill/extract_trajectories.py`: estrazione teacher offline, sempre in output directory unica.
- `outputs/<pipeline>/<run_id>/`: `artifacts/`, `metrics/`, `plots/`, `reports/`, con CSV/JSON locali obbligatori.

## Interfacce Pubbliche

- `TrajectoryRecord`: unità logica descritta nel manifest `.jsonl`.
  Campi: `sample_id`, `teacher_id`, `teacher_repo`, `teacher_ckpt`, `tokenizer_id`, `dataset_id`, `split`, `shard_path`, `row`, `seq_len`, `K`, `dtype`, `latent_shape`, `logit_shape`, `metadata`.

- Shard P0: `.pt` con dizionario tensoriale.
  Campi obbligatori: `tokens`, `attention_mask`, `teacher_id`, `K`, `z` con shape `[K+1, L, D]`, `logits` con shape `[K+1, L, V]` oppure top-k compresso, `loss_K`, `residual_norm`, `solver_iters`.
  Default P0: shard `.pt` + manifest `.jsonl`; `zarr` diventa P1 se gli shard diventano troppo grandi.

- `TeacherRunner`:
  `run_batch(tokens, attention_mask, depths: list[int]) -> TrajectoryBatch`.
  Deve restituire latenti loopati normalizzati, logits per depth, residual norm e solver iterations se disponibili. Ouro usa `transformers` e `config.total_ut_steps`; Attractor/Parcae usano adapter repo-specifici.

- `StudentFlowModel`:
  `forward(z_t, t, delta, tokens=None, attention_mask=None, context=None, mode="velocity") -> StudentOutput`.
  Output: `velocity`, opzionale `z_next`, opzionale `avg_velocity`, opzionale `logits`.

- `LossModule`:
  `compute(batch, student, loss_cfg) -> dict[str, Tensor]`, con ritorno sempre scalari loggabili e metriche diagnostiche.

- `QualityEvaluator`:
  `compute(batch, student) -> dict[str, Tensor]`, richiede `z`, `tokens`, `attention_mask` e logits teacher finali. Esegue rollout student fino a `z_K_student`, proietta `logits_student_K`, confronta contro `logits_teacher_K` e next-token targets.

## Loss E Metodi

- P0 FM lineare: campiona coppie teacher `(a,b)` dalla traiettoria, interpola `z_t = (1-s) z_a + s z_b`, target `v = (z_b - z_a)/(t_b - t_a)`, loss masked MSE su latenti.
- P0 endpoint/logit: KL temperatura-scalata tra logits student endpoint e `logits_K`; supportare full logits o top-k compresso.
- P0 eval quality: non ottimizza. Misura `KL(logits_student_K, logits_teacher_K)`, `NLL teacher`, `NLL student`, `NLL delta`, `PPL teacher`, `PPL student`, `PPL delta`, `top1 agreement`, `top-k overlap`.
- P0 latent reconstruction: rollout student fino a `K` e MSE masked contro `z_K`.
- P0 stability: se esiste una mappa `F`, penalizzare `||F(z_K)-z_K||`; per DEQ/Attractor usare anche residual teacher quando disponibile.
- P1 MeanFlow: port PyTorch locale della formula MeanFlow con `torch.func.jvp`; default autocast off nel JVP, target detach, fallback finite-difference.
- P1 Shortcut/compositional: `F(z,t,d) ≈ F(F(z,t,d/2), t+d/2, d/2)`, con stop-gradient sul target composto.
- P1 FlowMap: Lagrangiana = supervisione diretta tra stati teacher; Euleriana = vincoli via JVP/derivata spaziale; PSD/LSD come self-distillation progressiva.
- P1 DEQ/C-DEQ: wrapper `DEQLatentLoop` intorno a `torchdeq.get_deq`, log di `abs_trace`, `rel_trace`, `nstep`, `sradius`; default `hook_ift=False`, solver `fixed_point_iter` o Anderson limitato.
- P2 OT-CFM/SB-CFM: usare `torchcfm` solo per matcher e OT sampler; OT su `[B,L,D]` passa da pooling/masked cost, non flatten cieco come default.

## Connessione Con Le Repo Esterne

- TorchCFM: dipendenza P0 per `ConditionalFlowMatcher`; non importare `runner/`, UNet o vecchio Lightning. Usare `OTPlanSampler` solo P1/P2.
- TorchDEQ: dipendenza P1 per solver e implicit differentiation; incapsulare in moduli nostri Hydra/Lightning.
- MeanFlow/pMF: non dipendere dai repo JAX/TPU; portare solo formule, sampler `t,r`, adaptive weighting, dual-head `u/v`. pMF PyTorch è inference-only, quindi solo riferimento.
- Attractor/Parcae/Ouro: non forkare. Adapter sottili che caricano checkpoint/tokenizer e registrano hook sui loop states.
- Ouro: usare HF `ByteDance/Ouro-*`; passare sempre `HF_TOKEN` esplicitamente. P0 supporta full recurrent steps; adaptive exit è P1.

## A100/FlashAttention2

- Training standard: `bf16-mixed`, `attn_implementation="flash_attention_2"` per modelli HF compatibili, gradient checkpointing configurabile.
- JVP MeanFlow: disabilitare FA2/SDPA fused se incompatibile; eseguire JVP in fp32 o autocast off.
- DDP: test esplicito per `torch.func.jvp`; se DDP rompe, usare path `model.module` con sincronizzazione o fallback finite-difference.
- Log obbligatori: wall-clock, max CUDA memory, NFE/loop count, batch effective tokens.

## Metriche E Output Attesi

- Metriche: latent MSE per depth, logit KL, NLL/PPL teacher-student-delta, top1 agreement, top-k overlap, downstream accuracy, NFE/loop vs quality, fixed-point residual, solver steps, latency, memory peak.
- File locali: `metrics/train.csv`, `metrics/val.csv`, `metrics/test.json`, `reports/run_summary.md`, `plots/loss_curves.png`, `plots/nfe_quality.png`, `artifacts/config_resolved.yaml`.
- Ogni run ha `run_id` unico; nessun overwrite di shard, metriche, checkpoint o report.

## Roadmap

- P0.1 Repo skeleton: `uv`, Hydra, Lightning, Rich logging, output standard, unit test smoke.
- P0.2 Trajectory extraction: Attractor e Parcae prima, Ouro HF subito dopo; shard `.pt` + manifest.
- P0.3 Student baseline: small latent transformer con time/delta embeddings e FA2 dove possibile.
- P0.4 Training baseline: FM lineare + KL endpoint + latent reconstruction + stability, validazione e report locali.
- P0.5 Quality eval: logits finali teacher/student, next-token NLL/PPL e top-k agreement su validation/test held-out.
- P1.1 MeanFlow/iMF: JVP loss, dual-head `u/v`, adaptive weighting, finite-difference fallback.
- P1.2 Compositional/Shortcut: consistency su salti multipli e curva NFE/quality.
- P1.3 DEQ/C-DEQ: TorchDEQ wrapper, solver metrics, residual-aware distillation.
- P1.4 OT/SB/FlowMap: TorchCFM OT ablations, PSD/LSD, Eulerian/Lagrangian variants.
- P2 Scaling: zarr storage, multi-node A100, Ouro adaptive exit, LT2/MELT-style memory experiments.

## Test Plan

- Unit: manifest parsing, shard roundtrip, mask-aware MSE/KL, shape `[B,L,D]`, variable length.
- Teacher smoke: ogni adapter produce `z[0..K]`, logits, tokenizer metadata e non cambia modello in training mode.
- Loss tests: FM target esatto su traiettoria lineare; MeanFlow JVP confrontato con finite difference su MLP piccolo; Shortcut loss zero su mappa lineare perfetta.
- Lightning smoke: 2 batch train/val CPU e 2 batch CUDA bf16.
- Quality metrics: test deterministici per KL/NLL/PPL/top1/top-k e verifica che `eval_quality/val/*` finisca in `metrics/val.csv`.
- A100 smoke: FA2 forward/backward, poi MeanFlow JVP con FA2 disattivabile.
- Regression: nessun run sovrascrive output precedenti; metriche CSV/JSON sempre presenti.

## Assunzioni E Default

- Default storage P0: shard `.pt`, non zarr.
- Default teacher P0: Attractor + Parcae; Ouro HF entra nello stesso milestone se il caricamento `transformers` espone gli stati loopati con hook puliti.
- Default path scientifico: prima distillazione supervisionata da traiettorie teacher, poi MeanFlow/Shortcut/DEQ. Questo riduce rischio rispetto a partire subito con JVP e solver impliciti.
- Default commerciale/licenza: copiare solo codice MIT quando necessario; repo CC BY-NC-SA o senza licenza sono riferimento, non sorgente di codice.
