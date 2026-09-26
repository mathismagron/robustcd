# Unified training and evaluation protocol (v0.2)

Every model in the benchmark is trained and evaluated under this protocol.
Anything a model does differently is listed as a deviation in its run record.
The choices below favour the credibility of the comparison over compute cost;
the rationale for each is given in "Decisions and rationale" at the end.
Two quantities are **provisional until the pilot**, and the pilot rules that
fix them are stated in advance so that they cannot be tuned after seeing test
numbers.

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
  rot90). **No cropping: every model trains on full 512 × 512 tiles**, the
  same size it is tested on. A model that cannot fit the batch in memory uses
  gradient accumulation to reach the effective batch size.
- **Normalisation**: each model's own (e.g. ImageNet mean/std), recorded.
- **Optimiser, learning-rate schedule, loss**: the model's published recipe.
  Loss design is part of a method (e.g. CSF-Mamba's SeK-oriented term), so
  it is not unified. For reference, ChangeMamba SCD uses AdamW (lr 1e-4,
  wd 5e-3), StepLR (step 10k, γ 0.5), and CE + Lovász-softmax on the change
  and both semantic heads plus a semantic-consistency MSE on unchanged pixels.
- **Budget: one budget for every model, counted in training samples.**
  Effective batch 16, 50,000 iterations, i.e. 800,000 samples, about 300
  epochs of the 2671-tile train split. This is the ChangeMamba-Small
  recipe, and it is **provisional until the pilot** (rule below). Each
  model's learning-rate schedule is rescaled to this budget. For example, a
  StepLR published for 50k iterations is kept as is, and a cosine schedule
  spans the full budget.
- **Pilot rule for the budget, fixed in advance.** Before the full campaign,
  train the pilot models (ChangeMamba, a CNN baseline, then CSF-Mamba) for
  1.5 × the budget, logging val SeK. Keep the budget if, for every pilot
  model, the best val SeK over the last 30 % of the budget is within 0.5 SeK
  point of the best val SeK over the extended run. Otherwise raise the budget
  to the smallest multiple of 10k iterations that satisfies the rule. The
  decision uses **val only**; the test set is not scored during the pilot.
- **Precision**: bf16 autocast on L40S, fp32 master weights.
- **Seeds: 3 per model** (seeds 0, 1, 2). A seed fixes weight
  initialisation, data order and augmentation draws. Every seed is reported,
  as mean ± sample std and as individual values; no seed is dropped. The
  degraded evaluation reuses the same 3 checkpoints, since degradations are
  test-time only, so the robustness results also carry training variance.
  cuDNN non-determinism is accepted and documented rather than forced off:
  deterministic kernels are not available for all operations and would slow
  some models unevenly.

## Checkpoint selection

Evaluate on **val** every **2,000 iterations** (32k samples, about 12
epochs), which gives 25 checks over the budget. Keep the checkpoint with the
best val SeK (SECOND definition, `SCDMeter`, full tile). Also keep the
**last** checkpoint. The best-val checkpoint gives the headline numbers; the
last-checkpoint test scores are reported in the appendix as a check that
selection is not driving the ranking. **The test set is scored exactly once
per trained model and per condition**, only after all training runs are
complete.

## Evaluation

1. Each model repo writes prediction PNGs only (`<pred>/im1`, `<pred>/im2`,
   and optionally `<pred>/change`). There is no test-time augmentation.
2. `scripts/evaluate.py` scores them. SeK, mIoU and Fscd are the primary
   scores; the `*_fromto` scores are secondary; confidence intervals come
   from a 1000-replicate image bootstrap.
3. Degraded conditions reuse the same checkpoint. Relative drops, with paired
   bootstrap intervals, come from `scripts/compare.py`.
4. **Primary: full tile, the same pixels in every condition.** Secondary:
   the valid interior of misregistered sets. For the secondary score, the
   clean predictions are rescored on **the same mask**
   (`evaluate.py --valid-from <degraded set>` on the clean predictions).
   Clean vs degraded is therefore always compared on an identical pixel set,
   which the paired bootstrap requires.

## Model adapters

The list of retrained methods and the selection criteria are in
`docs/models.md`. Each model is integrated through an adapter in
`robustcd/adapters/<model>/`.
The adapter keeps the upstream model, loss, optimiser, schedule, augmentation
and normalisation, and replaces only what the protocol fixes (loop, seeds,
precision, selection, export). An adapter is accepted only after:

1. its label and class conversions are unit-tested against both colour tables;
2. where the authors release a checkpoint, running it through the adapter
   with upstream decoding reproduces the published number (within the
   tolerance expected from that number having been selected on test);
3. every deviation from the upstream recipe is listed in the model's README.

Evaluation-time rules shared by all adapters: **fp32 inference, no TTA,
full 512 tiles**. Semantic maps are decoded as argmax over the land-cover
classes only, then masked by the predicted change map. Per-model details are
in `cluster/models/<model>/README.md`; ChangeMamba is the first
(`cluster/models/changemamba/README.md`).

## Hardware and software

NVIDIA L40S (48 GB) on Vulcan (Alliance). The stack is pinned in
`cluster/alliance/requirements-gpu.txt` (torch 2.5.1, mamba_ssm 2.2.4), and
the exact freeze is stored with the results. Throughput numbers (images/s,
memory) are measured on the same GPU type.

## Reporting rules

- Every training run that is started is reported, including failed or
  diverged runs, with the reason. A diverged run is restarted with the same
  seed once. If it diverges again, it is reported as such.
- Each run stores its config, git commit, `pip freeze`, SLURM job id, val
  curve, best/last checkpoint iteration and wall-clock time.
- Headline tables give mean ± std over 3 seeds. Every interval comes from the
  image bootstrap (1000 replicates), and relative drops use the paired
  bootstrap between clean and degraded on the same checkpoint.
- Deviations from this protocol are listed per model in a table in the paper.

## Decisions and rationale

| decision | choice | why |
|---|---|---|
| crop | none, full 512 tiles | Train and test then see the same tile size and context. Misregistration and resampling degradations interact with image extent, so training on smaller crops would put the models in a different regime at test time. |
| budget | one sample budget for all (16 × 50k), confirmed by a pre-registered val-only pilot rule | Published recipes differ mainly in how long they train, so a shared budget removes the most common confound in cross-paper comparisons. The pilot rule makes sure no model is compared before it has converged, without looking at test. |
| seeds | 3 per model, all reported | This is the minimum that gives a spread. Robustness differences between models may be of the same order as seed variance, and a benchmark that cannot show this cannot rank them. |
| val interval | every 2k iterations (25 checks); best-val headline, last-checkpoint in appendix | The interval is dense enough to catch the peak and sparse enough to limit the optimism of taking the maximum over many noisy checks on 297 val tiles. The last-checkpoint scores reveal any ranking that depends on selection. |
| misregistration mask | full tile primary; valid interior secondary, clean rescored on the same mask | A full tile keeps the pixel set identical across all families and severities, and matches what a user of the product receives. The secondary score separates the vacated border strip (at most ~8.5 % of pixels at 16 px) from misalignment in the interior. |

Compute implication: 8–10 models × 3 seeds = 24–30 training runs of 800k
samples each, plus the pilot at 1.5 × budget. The wall-clock time per run is
measured in the pilot, before the campaign is scheduled.
