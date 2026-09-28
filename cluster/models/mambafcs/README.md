# Mamba-FCS

Adapter: `robustcd/adapters/mambafcs/`. Upstream is
[Buddhi19/MambaFCS](https://github.com/Buddhi19/MambaFCS) (MIT, IEEE JSTARS
2026), pinned at commit `5c01f1ac`. The code is imported from a checkout that
must be named `MambaFCS` (`~/ext/MambaFCS`), because upstream imports itself
as `MambaFCS.changedetection...`.

| | |
|---|---|
| Model | `STMambaSCD`: VMamba-Base siamese encoder (depths 2-2-15-2, dim 128, `v3noz`, drop path 0.6), change decoder with FFT fusion, change-guided semantic decoders |
| Parameters | 206.1 M (measured) |
| Encoder init | ImageNet `vssm_base_0229_ckpt_epoch_237.pth` (ChangeMamba Zenodo record 15479555, md5 `3e41…1e71`) |
| Released checkpoint | `SECOND_SeK_0.255.pth` (HF `buddhi19/MambaFCS`, sha256 `b1f3…1522`, 825 MB, loads strictly: 1312/1312 keys) |
| Published SECOND | OA 88.62, Fscd 65.78, mIoU 74.07, SeK 25.50 (README table) |
| Class order | ChangeMamba order (`annotations/MambaFCS.ipynb`); `SECOND_TO_CM` reused, test-checked |
| CUDA kernel | same `selective_scan_cuda_oflex` sources as ChangeMamba, so the extension built for ChangeMamba is reused |

## Setup and jobs

```bash
bash ~/robustcd/cluster/models/mambafcs/setup.sh               # login node: code + weights, checks the kernel
cd $SCRATCH/robustcd/logs
sbatch ~/robustcd/cluster/models/mambafcs/validate.sbatch      # released checkpoint + timing (accum 4 and 2)
sbatch --export=ALL,SEED=0,MAX_ITERS=75000,ACCUM=<from timing> ~/robustcd/cluster/models/mambafcs/train.sbatch
```

## Kept from upstream (`changedetection/script/train_MambaSCD.py`, SECOND)

- **Loss:** `1.0·CE(change) + 0.5·(CE(t1)+CE(t2)) + 0.5·(Lovász(t1)+Lovász(t2)+Lovász(change)) + 0.05·MSE(unchanged) + 0.5·SeK_Loss`.
  - Semantic class 0 is ignored.
  - The SeK term is off at the first iteration only.
- **Optimiser:** AdamW, lr 1e-4, weight decay 5e-4; StepLR with γ 0.5.
- **Augmentation, in upstream order, using upstream's `imutils` functions:**
  - random left-right and up-down flips;
  - rotation by 90/180/270° (never 0°, as upstream);
  - swap of the two dates with their labels (p = 0.5).
- **Normalisation:** ImageNet mean/std.
- **Heads:** the change head is decoded by argmax over 2 channels.

## Deviations (protocol)

1. **No photometric augmentation.** Upstream jitters brightness (0.8–1.2),
   contrast and saturation (0.9–1.1) independently on each date. This
   overlaps with the radiometric degradation family, so it is removed, as for
   PerASCD.
2. **Batch and schedule.**
   - Upstream uses batch 2 for 800k samples (400k iterations) and halves the
     learning rate every 10k iterations, i.e. every 20k samples.
   - The protocol budget is also 800k samples (16 × 50k), so the step is kept
     at 20k samples: `--lr-step 1250` iterations. The learning rate is kept at
     1e-4 despite the 8× larger batch.
   - As upstream, the learning rate falls below 1e-7 after about 12.5k
     iterations. Most of the budget is therefore spent at a negligible
     learning rate, which is how upstream's own schedule behaves.
3. **Selection.** Upstream selects on test SeK (every 5k iterations, then
   every 1k after 30k). robustcd selects on val SeK every 2k iterations.
4. **Precision.** bf16 autocast for the network. All losses are computed in
   fp32 outside autocast: the SeK loss sums soft confusion matrices over every
   changed pixel of a batch.
5. **Numerical guards.** Each one acts only where upstream would produce a NaN
   or crash; otherwise the loss is identical to upstream's.
   - *SeK term skipped when non-finite.* `SeK_Loss` takes
     `log(kappa·exp(1.5·mIoU) + 1e-7)`, and the soft kappa is negative for
     some batches while the heads are near chance. Measured: NaN for 1 of 4
     random decoder inits on real SECOND tiles, and for 25 of 60 random-logit
     draws. The count is logged as `sek_skipped`.
   - *Lovász terms with no valid pixel give 0.* Upstream returns an empty
     tensor, which would break the sum.
   - *CE with no valid pixel gives 0.* Upstream gives NaN, e.g. for the
     semantic terms of a micro-batch without any changed pixel.
6. **Date-swap draw.** It uses `random` instead of `np.random`. The
   distribution is the same, but the draw is correctly reseeded per
   DataLoader worker.
7. **Accumulation.** The effective batch is 16 through gradient
   accumulation; `ACCUM` is set from the timing job. BatchNorm layers in the
   decoders see one micro-batch (upstream sees 2).
8. **Backbone loading is checked.** Upstream catches every exception while
   loading and continues from scratch. Here the load fails if fewer than 100
   tensors match. Measured result:
   - 398 tensors loaded;
   - 4 unexpected (the `classifier.*` head);
   - 8 missing (`outnorm0–3`, created by the SCD backbone, as upstream).

## Validation

- Unit tests (`tests/test_mambafcs_adapter.py`, 4 tests):
  - the class order and normalisation match upstream's files;
  - upstream `SCDD_eval_all(..., 37)` equals the 7-class SECOND definition,
    so the published SeK is directly comparable to robustcd SeK;
  - the CE guard equals the mean CE when valid pixels exist.
- CPU smoke run: 3 iterations with accumulation, val, checkpoints. It uses a
  PyTorch reference scan instead of the CUDA kernel and 64-px crops
  (`--debug-crop`, tests only).
- **Released checkpoint on the full SECOND test** (job on L40S, fp32, no TTA,
  `results/model_checks/mambafcs_released.json`): SeK **0.2526** [95% CI
  0.2412, 0.2637], mIoU 0.7406, Fscd 0.6542, against 0.2550 / 0.7407 / 0.6578
  published. SeK is −0.24 pt from the published value, inside the CI, so the
  adapter is accepted. It is not an exact match: ChangeMamba-T reproduced its
  published value to the fourth decimal. Upstream and protocol decodings are
  identical. Throughput: 8.3 img/s.
- **Cost** at effective batch 16, bf16, full 512 tiles (60 iterations):

  | Micro-batch | s/it | Max memory |
  |---|---|---|
  | 4 (`ACCUM=4`) | 1.51 | 21.7 GiB |
  | 8 (`ACCUM=2`) | 1.58 | 39.9 GiB |

  `ACCUM=4` is kept: it is both faster and far from the memory limit. That
  gives 20.9 h per 50k seed, which fits one 24 h allocation, and about 31 h
  for the 75k pilot, which resubmits itself once.
- Released checkpoint, first 16 test tiles on CPU, upstream decoding: SeK
  0.2543. At n = 16 this only shows that classes are mapped correctly; a
  permuted class order would collapse SeK. The comparison with the
  published 0.2550 is the full 1694-tile check in `validate.sbatch`.
