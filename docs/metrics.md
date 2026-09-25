# Metrics

One implementation of SeK, mIoU, Fscd (semantic change detection) and
F1 / IoU / precision / recall (binary change detection), used for every model in
the benchmark. Model repos **write predictions only**; `scripts/evaluate.py`
scores all of them. No model is scored by its own repository's metric code.

```
robustcd/metrics/
  scd.py        # confusion accumulation, SeK / mIoU / Fscd / binary scores, SCDMeter, BCDMeter
  labels.py     # SECOND colour table, strict RGB <-> index decoding
  bootstrap.py  # image-level bootstrap CI, paired bootstrap for differences / relative drops
  io.py         # confusion.npz save / load / align
scripts/
  evaluate.py   # score a prediction directory -> metrics.json + confusion.npz
  compare.py    # paired comparison of two evaluated runs
tests/test_metrics.py
```

## Definitions

H is the K × K confusion matrix accumulated over the whole test set and over
**both dates** (2N maps), class 0 = no change.

| metric | definition |
|---|---|
| binary matrix | TN = H[0,0]; FP = Σ H[0, j>0]; FN = Σ H[i>0, 0]; TP = Σ H[i>0, j>0] |
| mIoU | mean of IoU(no change) and IoU(change) on the binary matrix |
| kappa_n0 | Cohen's kappa of H with H[0,0] set to 0 |
| SeK | kappa_n0 · exp(IoU_change − 1) |
| Fscd | harmonic mean of Pscd = Σ_{i>0} H[i,i] / #predicted changed and Rscd = Σ_{i>0} H[i,i] / #true changed |

All of these are invariant to transposing H (tested), so the row/column
convention cannot introduce a discrepancy.

## Conventions that change the number

These are the choices that make "SeK of model X" differ between papers. Each is
fixed here and should be stated once in the paper.

1. **Dataset-level accumulation, not per-image averaging.** Only the
   dataset-level form is exposed. Per-image matrices are stored solely for
   the bootstrap. The Bi-SRNet utilities that many SCD repos copy contain both
   forms (`SCDD_eval_all` and `SCDD_eval`). The table below shows how far
   apart they are on SECOND.
2. **Consistent predictions.** A semantic map must be 0 wherever the predicted
   change map is 0. If a model writes `<pred>/change/`, the scorer applies it
   with `compose_prediction`; otherwise the im1/im2 maps are taken as final.
3. **Strict label decoding.** A colour outside the SECOND table raises an error
   instead of being mapped to the nearest class. Anti-aliased or JPEG-saved
   predictions would otherwise be scored silently wrong.
4. **Out-of-range predictions raise an error.** They are never dropped.
5. **Degenerate denominators.** Precision, recall and IoU are NaN when their
   denominator is 0. F1 and Fscd are 0 when TP = 0 and something was
   predicted or present. The reference returns NaN from `hmean([nan, 0])` for
   an all-unchanged prediction; this module returns Fscd = 0.
6. **Valid mask.** `--valid-from <rendered set>` removes the vacated border
   strip of a degraded set from both dates. Without it, the full tile is
   scored. Pick one protocol for the headline numbers.

## Verification

Unit tests (`python tests/test_metrics.py`, 13 tests): perfect prediction gives
1; the all-unchanged prediction is handled; agreement with an independent
loop-based SeK; transpose invariance; streaming accumulation equals a single
pass; valid mask equals cropping; range checks; strict decoding; binary scores
on a hand-computed case; bootstrap determinism; and the paired interval being
narrower than the combined marginals.

Reference equivalence: with `ROBUSTCD_REFERENCE_UTILS=<Bi-SRNet utils/utils.py>`,
the test suite compares against `SCDD_eval_all`. That file is not vendored.

**SECOND test, all 1694 images.** These are pseudo-predictions built from the
ground truth, **not model outputs**. They exercise the full pipeline and
quantify the aggregation effect.
[`results/metrics_validation_second.csv`](../results/metrics_validation_second.csv)

| pseudo-prediction | SeK (this module) | SeK (`SCDD_eval_all`) | SeK, per-image mean (`SCDD_eval`) | gap | SeK 95 % CI (image bootstrap) |
|---|---|---|---|---|---|
| ground truth | 1.0000 | 1.0000 | 1.0000 | 0 | [1.0000, 1.0000] |
| GT shifted 2 px | 0.8084 | 0.8084 | 0.7623 | −0.046 | [0.8042, 0.8131] |
| GT shifted 4 px | 0.6831 | 0.6831 | 0.6209 | −0.062 | [0.6768, 0.6894] |
| GT shifted 8 px | 0.5006 | 0.5006 | 0.4301 | −0.071 | [0.4921, 0.5090] |
| GT shifted 16 px | 0.3001 | 0.3001 | 0.2311 | −0.069 | [0.2910, 0.3094] |
| GT, 20 % class noise | 0.7273 | 0.7273 | 0.6996 | −0.028 | [0.7260, 0.7284] |

- This module and `SCDD_eval_all` agree exactly (maximum absolute difference 0
  over SeK, mIoU and Fscd) in all six conditions.
- Per-image averaging lowers SeK by 3 to 7 points on these pseudo-predictions.
  That is comparable to the gaps separating recent SCD methods, so the
  aggregation choice alone could reorder a leaderboard. The size of the gap for
  real model outputs still has to be measured on the trained models.
- The bootstrap intervals are narrow (± 0.001 to 0.009) because these
  pseudo-predictions spread their errors uniformly over images. Real models
  make errors that vary much more from image to image, so expect wider
  intervals. Use the actual intervals from `metrics.json`, not these.
- The end-to-end CLI run (PNG write → `evaluate.py` → RGB decode) reproduces
  the in-memory value (SeK 0.6831 for the 4 px condition).

## Usage

```bash
# score one run (predictions as index or SECOND-colour PNGs)
python scripts/evaluate.py scd --gt /data/SECOND/test --pred runs/cnn/pred_clean \
    --out runs/cnn/eval_clean

# same model on a degraded set, scored on its valid interior
python scripts/evaluate.py scd --gt /data/SECOND/test --pred runs/cnn/pred_misreg-shift_int-s2 \
    --valid-from /scratch/.../misreg-shift_int-s2 --out runs/cnn/eval_misreg-shift_int-s2

# paired relative drop with CI
python scripts/compare.py runs/cnn/eval_clean runs/cnn/eval_misreg-shift_int-s2 --key SeK

# binary (LEVIR-CD)
python scripts/evaluate.py bcd --gt-dir /data/LEVIR-CD/test/label --pred runs/cnn/levir_pred \
    --out runs/cnn/levir_eval
```

In Python, for validation loops during training:

```python
from robustcd.metrics import SCDMeter, compose_prediction
m = SCDMeter()
for p1, p2, ch, g1, g2 in loader_outputs:          # numpy, class indices
    m.update(compose_prediction(p1, ch), compose_prediction(p2, ch), g1, g2)
print(m.compute()["SeK"])
```

Using the same meter for checkpoint selection during training removes one more
source of cross-paper variance: the validation metric that picks the checkpoint.
