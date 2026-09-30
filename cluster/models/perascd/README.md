# PerASCD under the robustcd protocol

PerASCD (SathShen/PerASCD, MIT) has a siamese **PerA ViT-G/16-1024**
foundation encoder (40 blocks, 1024-d). The encoder is wrapped in a ViT-Adapter
with multi-scale deformable attention. On top sit a cascade gated decoder, two
7-way semantic heads and a 1-channel change head. The model has 548.2 M
parameters and costs 1.51 TMACs per 512 × 512 pair.

- Code: `legacy` branch, pinned at `a4d808a6cfb5df7efeee186730ac26b4504c9ed6`. Upstream
  code is imported from the pinned clone and never vendored.
- Adapter: `robustcd/adapters/perascd/`, with tests in `tests/test_perascd_adapter.py`.

## Files and weights

| File | Source | Use |
|---|---|---|
| `pera_ViTG161024.params` (9 097 269 032 B) | PerA README → Google Drive | initialisation of every protocol run |
| `PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth` | HF `SathShen/PerASCD-Checkpoint`, zip `PerASCD_260128115444_vitg01min0Clip15LsscTau001` | adapter acceptance check only |

`setup.sh` writes both sha256 values to `~/weights/perascd/SHA256SUMS`.

**The foundation file is the one upstream used.** The torch archive inside
`pera_ViTG161024.params` is named `pera_ViTGall22601_ep42_auto`, which is
exactly the `pretrained_pera_path` in legacy `train.py`. It holds 997 tensors:
the DINO-style student and teacher, plus heads. Upstream loads
`["model"]["teacher.backbone.*"]`, i.e. 486 tensors. The key index was read
without downloading the file:

- all 486 tensors match the backbone by name and shape, including
  `pos_embed` 1 × 785 × 1024 (the ViT is built at 448 = 28 patches);
- none is unexpected;
- the 239 backbone state entries left at initialisation all belong to the
  adapter: interactions 172, SPM 44, norm1–4 BatchNorms 20, up-sampler 2,
  level embedding 1. On the cluster, `load_state_dict` reports 229 of them as
  missing: PyTorch fills in the 10 BatchNorm `num_batches_tracked` counters
  without listing them.

`load_pretrained` enforces the same checks on every run. Upstream loads with
`strict=False` and prints a success line whatever matched.

## Recipe kept from upstream

Legacy `train.py`, SECOND run `vitg01m0C15LsscTau001`:

| Item | Upstream | Protocol run |
|---|---|---|
| Encoder | PerA ViT-G, fine-tuned (`is_freeze_backbone=False`) | same |
| Drop path | 0.3, uniform (DINOv2 batch-subset stochastic depth) | same |
| Loss | `0.5·(CE_A + CE_B)` (class 0 ignored) `+ wBCE(change) + SoftSemanticConsistency(τ=0.01)` on classes 1..6 | same, computed in fp32 |
| Optimiser | SGD, lr 0.1, momentum 0.9, Nesterov, wd 1e-5 on all parameters | same |
| LR schedule | per-iteration linear warm-up over the first 10 %, then poly 1.5 to 0 | same rule over 50k iterations (warm-up 5k) |
| Gradient clipping | global norm 1.5 | same (`LoopConfig.clip_grad_norm`) |
| Geometric augmentation | rot90 (p 0.5), then one of {none, V, H, both} flips, applied to both dates | same (`rot90_flip_ding`, same distribution) |
| Photometric augmentation | `CDMColorJitter(0.2, 0.2, 0.1, 0.1)` | **removed** (protocol) |
| Normalisation | PerA mean/std on [0, 1], both dates | same |
| Batch | 4 × 2 accumulation = 8 | **16** (micro-batch 16/ACCUM) |
| Length | 50 epochs (≈ 119–148 k samples, see note below) | **50 k iterations = 800 k samples** |
| Precision | fp16 autocast + GradScaler | **bf16 autocast**, no scaler |
| Selection | best **Fscd on the test split** (epoch 40 released) | best **SeK on robustcd val** (297 train tiles) |
| Pseudo labels | code present but commented out | not used (as upstream) |

Note on upstream length: `datasets/SECOND/train_info.txt` lists 2375 training
tiles, but the `SECONDbi` layout that `train.py` reads is not published. The
upstream run therefore saw either 2375 or 2968 tiles per epoch.

## Deviations that do not change the model function

- **Attention via PyTorch SDPA.** Upstream `MemEffAttention` calls xformers
  `memory_efficient_attention`, and upstream `requirements.txt` pins xformers
  0.0.32. Without xformers, the upstream code falls back to an explicit N × N
  attention and also keeps every attention map in eval mode. The robustcd env
  has no xformers, so the adapter uses `F.scaled_dot_product_attention` (same
  function; the test checks it against the explicit path to 1e-5). The kept
  attention maps are read only by upstream visualisation code.
- **Deformable attention** is the upstream CUDA op, compiled for sm_89 by
  `validate.sbatch` into `~/ext/perascd_msda_site`. As upstream does, it casts
  its value and attention inputs to fp32 under autocast.
- **Activation checkpointing** (`CKPT=adapter|full`), used only if memory
  requires it:
  - `adapter` is upstream's own `with_cp` (injectors/extractors);
  - `full` also checkpoints the 40 ViT blocks. This is exact: the test compares
    loss and gradients with and without it, with drop path active.
- **Empty micro-batches.** A micro-batch without any changed pixel would make
  upstream's `NLLLoss(mean, ignore 0)` NaN. Its segmentation term is set to 0.
- **Class order.** The released checkpoint predicts in ChangeMamba order, and
  `decode_released` maps it to robustcd order. Protocol runs train directly in
  robustcd order.

## Commands (Vulcan)

```bash
# login node, once (≈ 13 GB of downloads)
cd ~/robustcd && git pull && bash cluster/models/perascd/setup.sh
# GPU: compile the op, released-checkpoint check, timing
cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/perascd/validate.sbatch
# after choosing ACCUM / CKPT from the timing lines
sbatch --export=ALL,SEED=0,ACCUM=4,CKPT=none ~/robustcd/cluster/models/perascd/train.sbatch
sbatch --export=ALL,SEED=0,ACCUM=4,CKPT=none,MAX_ITERS=75000,RUN_NAME=pilot75k ~/robustcd/cluster/models/perascd/train.sbatch
```

A warm-up/poly schedule depends on its total length, so the 75k pilot is a
separate run, as for the Ding models (amendment A1 paired rule; decision in
`docs/protocol.md`).

## Acceptance (Vulcan job 1239790, 2026-09-30, L40S)

- **Released checkpoint reproduced exactly.** Upstream decoding gives SeK
  0.2611, Fscd 0.6641 and mIoU 0.7433, i.e. 0.0000 from the published values
  on all three. Protocol decoding is identical (SeK 95 % CI [0.2498, 0.2724],
  1000 image replicates).
  - Load: 988 keys, 6 CAGM renames, 0 missing, 0 unexpected.
  - The run used the SDPA attention and the compiled MSDA op (built in 38 s).
  - fp32 inference at batch 4: 5.39 images/s, 3.7 GiB peak memory (explicit
    upstream attention: ≈ 25 GiB at batch 8).
  - Record: `results/model_checks/perascd_released.json`.
- **Timing.** Effective batch 16, full 512 tiles, PerA initialisation, 30
  iterations; the mean excludes the first 10.

  | ACCUM (micro-batch) | Checkpointing | s/it | Peak GiB | 50k iterations |
  |---|---|---|---|---|
  | **4 (4)** | **none** | **1.902** | **24.3** | **26.4 h** |
  | 4 (4) | full | 2.419 | 13.5 | 33.6 h |
  | 8 (2) | none | 2.078 | 15.8 | 28.9 h |
  | 2 (8) | full | 2.521 | 19.9 | 35.0 h |

  Validation on the 297 val tiles takes about 105 s, i.e. about 0.7 h over
  the 25 evaluations of a run. The global gradient norm before clipping is
  1.7–2.0 in these first iterations, so upstream's 1.5 clip is active.
- **Setting used: ACCUM=4, CKPT=none.** It is the fastest, and its
  micro-batch of 4 is upstream's own (4 × 2 accumulation). The adapter and
  decoder BatchNorms therefore see upstream's per-pass batch statistics.
  - Per seed: ≈ 27 h, i.e. two 24 h segments with the 22.5 h guard.
  - 75k pilot: ≈ 41 h.
  - Three seeds plus the pilot: ≈ 122 L40S GPU-h.
