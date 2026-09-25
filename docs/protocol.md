# Unified training and evaluation protocol (draft v0.1)

Every model in the benchmark is trained and evaluated under this protocol.
Anything a model does differently is listed as a deviation in its run record.
Items marked **OPEN** still need a decision.

## Why a protocol is needed: what the reference codebases actually do

Checked in the ChangeMamba repository (commit `9ce9cec`, April 2026), SCD task
on SECOND:

1. **Checkpoint selection on the test set.** The loader called "Validation"
   is built from `test_dataset_path` / `test_data_name_list`
   (`changedetection/tasks/metadata.py`). The model is evaluated every 500
   iterations and the best test SeK is kept (`selection_metric` returns
   `eval_results["Validation"]["sek"]`). Reported test scores are therefore
   maxima over roughly `max_iters / 500` looks at the test set.
2. **A different SeK.** Its evaluator scores a 37-class "from → to" map,
   `(c1 − 1)·6 + c2` with 0 for no change (`SemanticChangeEvaluator(num_class=37)`),
   instead of the SECOND definition: 7 classes per date, both dates
   accumulated in one matrix. On SECOND test, with pseudo-predictions derived
   from the ground truth, the two definitions differ by +0.8 to +1.1 SeK
   points for boundary errors and by **−13.5 points** for a 20 % class-noise
   error (`results/metrics_sek_definitions_second.csv`). They are not
   interchangeable, and they penalise error types differently.
3. **Overall accuracy averaged per image**, not per pixel.

This protocol fixes all three: a held-out validation split, the SECOND SeK
definition as the primary score (the from → to variant is reported as
secondary, `SeK_fromto`, for comparison with papers that use it), and pixel
accumulation over the whole test set (`docs/metrics.md`).

## Data

| split | source | n | list |
|---|---|---|---|
| train | SECOND train minus val | 2671 | `splits/SECOND/train.txt` |
| val | 10 % of SECOND train, deterministic hash split | 297 | `splits/SECOND/val.txt` |
| test | SECOND test | 1694 | `splits/SECOND/test.txt` |

The val split is the 10 % of train ids with the lowest
`blake2b-64("robustcd-second-val-v1|<id>")`. It was not tuned. It is
representative of the rest: change ratio 0.193 vs 0.200, and the class shares
of changed pixels are within 1.7 points (`splits/SECOND/README.json`, which
also stores the sha256 of each list).

Input: full 512 × 512 tiles, RGB uint8. Labels are decoded with the strict
SECOND colour table (`robustcd.metrics.labels`). Every job stages
`SECOND.tar` to `$SLURM_TMPDIR`.

## Training

- **Augmentation: geometric only**, applied identically to both dates and all
  labels: random horizontal and vertical flips and random 90° rotations.
  **No photometric, noise, blur or resampling augmentation.** Such
  augmentations overlap with the benchmark's degradation families and would
  let a model acquire robustness at training time to exactly what is being
  measured. ChangeMamba's SECOND pipeline already complies (crop, flips,
  rot90). Random crop: **OPEN**, either full tiles or 512 crops, which on
  512 tiles is a no-op.
- **Normalisation**: each model's own (e.g. ImageNet mean/std), recorded.
- **Optimiser, learning-rate schedule, loss**: the model's published recipe.
  Loss design is part of a method (e.g. CSF-Mamba's SeK-oriented term), so
  it is not unified. For reference, ChangeMamba SCD uses AdamW (lr 1e-4,
  wd 5e-3), StepLR (step 10k, γ 0.5), and CE + Lovász-softmax on the change
  and both semantic heads plus a semantic-consistency MSE on unchanged pixels.
- **Budget: OPEN.** Proposal: one fixed number of iterations × batch size
  for every model, set from the pilot. ChangeMamba's README uses batch 16
  with 20k / 50k / 80k iterations depending on model size.
- **Precision**: bf16 autocast on L40S, fp32 master weights.
- **Seeds: OPEN.** Proposal: 3 seeds per model on clean data. The degraded
  evaluation then reuses those same checkpoints, since degradations are
  test-time only.

## Checkpoint selection

Evaluate on **val** every N iterations (**OPEN**, e.g. 1000). Keep the
checkpoint with the best val SeK (SECOND definition, `SCDMeter`). **The test
set is scored exactly once per trained model and per condition**, with that
checkpoint.

## Evaluation

1. Each model repo writes prediction PNGs only (`<pred>/im1`, `<pred>/im2`,
   and optionally `<pred>/change`). There is no test-time augmentation.
2. `scripts/evaluate.py` scores them. SeK, mIoU and Fscd are the primary
   scores; the `*_fromto` scores are secondary; confidence intervals come
   from a 1000-replicate image bootstrap.
3. Degraded conditions reuse the same checkpoint. Relative drops, with paired
   bootstrap intervals, come from `scripts/compare.py`.
4. Full tile vs valid interior for misregistration: **OPEN**. One of the two
   is stated as primary.

## Hardware and software

NVIDIA L40S (48 GB) on Vulcan (Alliance). The stack is pinned in
`cluster/alliance/requirements-gpu.txt` (torch 2.5.1, mamba_ssm 2.2.4), and
the exact freeze is stored with the results. Throughput numbers (images/s,
memory) are measured on the same GPU type.
