# Ding-codebase models: SCanNet, TED, Bi-SRNet, HRSCD-str4 (and SSCD-l)

Adapter: `robustcd/adapters/ding/`. Upstream code is imported from pinned
checkouts. Neither repository has a licence file, so no upstream code is copied
into robustcd.

| Model | Upstream | Commit | Params | Recipe |
|---|---|---|---|---|
| SCanNet | DingLei14/SCanNet `models/SCanNet.py` | `9c80d463` | 27.9 M | SC loss (margin 0.1) + pseudo labels |
| TED | DingLei14/SCanNet `models/TED.py` | `9c80d463` | 24.2 M | same as SCanNet |
| Bi-SRNet | DingLei14/Bi-SRNet `models/BiSRNet.py` | `012f35aa` | 23.4 M | SC loss (margin 0.0) |
| HRSCD-str4 | DingLei14/Bi-SRNet `models/Daudt/HRSCD4.py` | `012f35aa` | 13.7 M | no SC loss |
| SSCD-l (not in the campaign) | DingLei14/Bi-SRNet `models/SSCDl.py` | `012f35aa` | 23.3 M | no SC loss |

Parameter counts are measured from the instantiated models.

## Setup and jobs

```bash
bash ~/robustcd/cluster/models/ding/setup.sh             # login node: code, ResNet-34, SCanNet checkpoint
cd $SCRATCH/robustcd/logs
sbatch ~/robustcd/cluster/models/ding/validate.sbatch    # checkpoint check + per-model timing
sbatch --export=ALL,MODEL=scannet,SEED=0 ~/robustcd/cluster/models/ding/train.sbatch
```

## What is kept from upstream

- **Models**: instantiated exactly as the training scripts do, with an ImageNet
  ResNet-34 encoder. HRSCD-str4 has its own encoder trained from scratch.
- **Input normalisation**: per-date SECOND statistics (`MEAN_A/STD_A`,
  `MEAN_B/STD_B` from `datasets/RS_ST.py`).
- **Augmentation**: `rand_rot90_flip_SCD`, i.e. rot90 with p = 0.5, then one of
  four flips. It is geometric only, as the protocol requires, and a test checks
  that it reproduces the upstream function draw for draw.
- **Loss**: `0.5 * CE(A) + 0.5 * CE(B)` with class 0 ignored, plus the weighted BCE
  (0.25 / 0.75) on the change logit, plus `ChangeSimilarity` on classes 1..6 for
  the models whose recipe uses it. Each model uses its own repository's loss
  module (the cosine margin differs: 0.1 in SCanNet, 0.0 in Bi-SRNet).
- **Optimiser**: SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 on all
  parameters, poly decay `(1 - it/total)^1.5` applied at every iteration.
- **SCanNet/TED pseudo labels** (`train_SCD_psd.py`): re-implemented in
  `psd.py`, because the script executes on import. A test checks it against the
  upstream `AverageThred` and `calc_conf`. Rules:
  - the teacher is the network at the best val Fscd, flip-averaged over four passes;
  - the stage is active once the best val Fscd exceeds 0.6;
  - per-class running thresholds use the 60th percentile, clipped to [0.5, 0.9];
  - labels are added only on unchanged pixels where both dates are confident
    and agree on the class.
- **Change decision**: `sigmoid(logit) > 0.5`.

## Deviations (protocol)

1. **Batch and length.** Effective batch 16 for 50k iterations (800k samples).
   Upstream uses batch 8 for 50 epochs of 2375 images (about 14.8k iterations,
   119k samples). The learning rate is kept at the published 0.1. The poly
   schedule is stretched over 50k iterations.
2. **Val split and selection.** Upstream uses its own 2375 / 593 split of the
   SECOND train set and selects on val Fscd. robustcd uses its 2671 / 297 split
   (`splits/SECOND`) and selects on val SeK every 2000 iterations. The released
   SCanNet checkpoint was trained on upstream's 2375-image split. Every
   evaluation here is on the test split, which is disjoint from both.
3. **Pseudo-label teacher updates** can only happen at protocol evaluations
   (every 2000 iterations, i.e. 32k samples ≈ 12 epochs). Upstream checks once
   per epoch. The teacher is still chosen on val Fscd, as upstream.
4. **Precision**: bf16 autocast for training (upstream fp32). Losses are computed
   in fp32 on fp32-cast logits. Inference is fp32.
5. **Semantic decoding**: argmax over classes 1..6 (protocol). Upstream takes the
   argmax over all 7 channels. Class 0 is ignored by the loss, so its logit is
   untrained. Both decodings are reported for the released checkpoint.
6. **Pilot**: poly decay depends on the schedule length, so a 75k pilot is a
   separate run (`MAX_ITERS=75000 RUN_NAME=pilot75k`), not an extension of the
   seed-0 run as for ChangeMamba. These models are cheap, so this costs little.
7. **HRSCD-str4 recipe**: the Bi-SRNet repository ships the model but no
   dedicated script. It is trained with Bi-SRNet's loop without the SC loss,
   since the SC loss is Bi-SRNet's contribution, not Daudt et al.'s.
8. **TED recipe**: same script and pseudo-label stage as SCanNet, so that TED
   vs SCanNet isolates the transformer.
9. **Bi-SRNet version.** Three versions of the model exist upstream. We use
   the Bi-SRNet repository's current `models/BiSRNet.py` (commit `012f35aa`,
   February 2026).
   - 2022 `models/BiSRNet.py` (last at commit `f926a0ab`): cannot be
     instantiated. `__init__` calls `initialize_weights(self.head, ...)`, but
     `head` belongs to the encoder (`AttributeError`, checked).
   - 2022 root-level `BiSRNet.py`, the original upload: runs, with one shared
     SR module per date and unscaled attention.
   - February 2026 update, the one we use: the author's working version.
     Compared with the root-level file, it adds `1/sqrt(d)` scaling of the
     attention logits and leaves SR/CotSR at PyTorch's default init.
   - The SCanNet repository's copy has two separate SR modules. It is not used.

   No Bi-SRNet checkpoint is released, so published numbers cannot be
   reproduced for any of these versions. The choice is recorded here and in
   `run_config.json`.

## Validation

- Unit tests: `tests/test_ding_adapter.py` (upstream checkouts under `$ROBUSTCD_EXT`).
- Released SCanNet checkpoint on SECOND test: `results/model_checks/scannet_released.json`.
