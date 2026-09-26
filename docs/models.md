# Model selection (v0.1)

This document fixes which methods the benchmark retrains, why, and in which
order they are integrated. Repositories, licences and released checkpoints
were checked on 2026-09-26 (GitHub API, Hugging Face API). Each adapter pins
an exact upstream commit in its `cluster/models/<model>/README.md`.

## Selection criteria

1. **Official public code** containing the training pipeline, not only the
   model definition. Methods whose code is unavailable are excluded,
   whatever their published score.
2. **Released SECOND checkpoint preferred.** With a checkpoint, the adapter is
   validated by reproducing the published number (protocol, "Model adapters",
   rule 2). Without one, the adapter is validated through unit tests and a
   plausibility check against published scores. That check is weaker and is
   labelled as such in the paper.
3. **Coverage of the factors the paper analyses**: architecture family
   (CNN / hybrid CNN-Transformer / state-space / foundation model), capacity
   (≈5 M to ≈550 M parameters), and pretraining source (ImageNet supervised
   vs remote-sensing self-supervised). Robustness differences can then be
   related to these factors rather than only ranked.
4. **Widely used as a reference on SECOND**, so readers can place results.
5. **Trainable within the protocol** on one 48 GB L40S: full 512 tiles,
   effective batch 16 (gradient accumulation allowed), shared sample budget.
6. **Licence.** Upstream code is never vendored. Adapters import it from a
   pinned clone. Repositories without a licence file can be used for
   research, but no upstream code is redistributed from them.

## Semantic change detection track (SECOND, Hi-UCD)

| # | Method | Venue | Family | Backbone / pretraining | Code (licence) | Released SECOND ckpt |
|---|---|---|---|---|---|---|
| 1 | HRSCD strategy 4 | CVIU 2019 | CNN (classic multi-task) | FC-EF-Res, from scratch | `DingLei14/Bi-SRNet` `models/Daudt/HRSCD4.py` (no licence file) | no |
| 2 | Bi-SRNet | TGRS 2022 | CNN + semantic reasoning | ResNet-34 / ImageNet | `DingLei14/Bi-SRNet` (no licence file) | no |
| 3 | SCanNet | TGRS 2024 | Hybrid CNN-Transformer | ResNet-34 + CSWin blocks / ImageNet | `DingLei14/SCanNet` (no licence file) | **yes** (Google Drive) |
| 4 | ChangeMask | ISPRS P&RS 2022 | Encoder-Transformer-Decoder | EfficientNet-B0 / ImageNet | `Z-Zheng/pytorch-change-models` (torchange, Apache-2.0) | no |
| 5 | ChangeMamba (MambaSCD-Tiny) | TGRS 2024 | State space | VMamba-T / ImageNet | `ChenHongruixuan/ChangeMamba` (Apache-2.0) | **yes**, validated (SeK_fromto 0.2208) |
| 6 | Mamba-FCS | JSTARS 2026 | State space + frequency fusion, SeK loss | VMamba-B / ImageNet | `Buddhi19/MambaFCS` (MIT) | **yes** (HF `buddhi19/MambaFCS`, `SECOND_SeK_0.255.pth`) |
| 7 | PerASCD | 2026 | Foundation model | PerA ViT-G (≈548 M) / RS self-supervised | `SathShen/PerASCD` (MIT) | **yes** (HF `SathShen/PerASCD-Checkpoint`, 4.1 GB) |
| 8 | CSF-Mamba | author's method | State space, lightweight | VMamba-T / ImageNet | local repo | own |

Optional ninth entry, added only if budget allows: **TED**, the pure-CNN
triple encoder-decoder from the SCanNet paper. It costs almost nothing to
integrate because it shares SCanNet's repository and loop, and it isolates the
effect of SCanNet's transformer on robustness.

### Why these and not others

- **HRSCD-str4** is the simplest competitive multi-task design. It gives the
  benchmark a low-capacity, no-pretraining floor. Without such a floor, a
  claim like "large models are more robust" cannot be tested.
- **Bi-SRNet** is the most commonly reported CNN on SECOND. Its SSCD-l
  variant is redundant with it and is not included.
- **SCanNet** and **TED** come from the same authors and share one codebase
  with Bi-SRNet, so a single adapter covers three or four methods. SCanNet is
  the only CNN/hybrid entry with a released SECOND checkpoint.
- **ChangeMask** provides a second, independent code lineage (Zheng et
  al., torchange) outside the Ding et al. codebase, and a small
  EfficientNet encoder. torchange contains the model, a SECOND loader and
  SECOND metrics.
- **Mamba-FCS** is a second state-space method with a released checkpoint.
  It uses VMamba-B, so ChangeMamba-T vs Mamba-FCS confounds architecture with
  size. This is stated explicitly. If budget allows, ChangeMamba-B (MambaSCD-Base,
  released on Zenodo record 14037769) disentangles the two.
- **PerASCD** is the foundation-model representative. It has a released
  checkpoint and an MIT licence, and its encoder is pretrained on remote
  sensing data. The authors also released a VMamba-B variant of the same
  framework, which is a candidate control for "foundation encoder vs
  state-space encoder under the same decoder".

Excluded:

- **SAM-CD, BIT, ChangeFormer**: binary change detection only (see the BCD
  track below).
- **Methods without public training code**: excluded regardless of score.
- **HRSCD strategies 1–3**: dominated by strategy 4 and redundant.

## Binary change detection track (LEVIR-CD)

LEVIR-CD has no semantic labels, so the semantic models above cannot be
trained on it without redesign. The BCD track therefore uses four reference
binary models covering the same families at low cost:

| Method | Family | Code |
|---|---|---|
| FC-Siam-diff | CNN | `rcdaudt/fully_convolutional_change_detection` (no licence file) |
| BIT | Hybrid CNN-Transformer | `justchenhao/BIT_CD` (no licence file) |
| ChangeFormer | Transformer | `wgcban/ChangeFormer` (MIT) |
| ChangeMamba (MambaBCD-Tiny) | State space | `ChenHongruixuan/ChangeMamba` (Apache-2.0), same adapter code |

Semantic models are also scored on the binary change map of SECOND and
Hi-UCD (F1 of the change channel). The binary metric is therefore available
for all 8 models on those two datasets.

## Protocol consequences (to record in each model README)

- **PerASCD colour jitter.** Its recipe includes photometric augmentation,
  which the protocol forbids (it overlaps the radiometric degradation family).
  It is removed as a documented deviation. Recommended ablation: retrain one
  cheap model (Bi-SRNet) and PerASCD *with* colour jitter, one seed each, to
  measure how much training-time augmentation mitigates the radiometric
  family. This is a result in itself.
- **PerASCD memory and cost.** ViT-G at 512 × 512 with effective batch 16
  will need gradient accumulation, and probably activation checkpointing, on
  48 GB. It also needs the MultiScaleDeformableAttention op compiled for
  sm_89, as was done for VMamba. Its per-iteration cost must be measured with
  a 200-iteration timing job before committing seeds.
- **Shared budget.** 800 k samples (≈270 SECOND epochs) exceeds several
  published schedules. Best-val selection guards against over-training. The
  pilot rule is applied to every model's seed-0 run, as for ChangeMamba.
- **Mamba-FCS SeK loss** and **CSF-Mamba's SeK-oriented term** are part of
  their methods. They are kept, as are all published losses.
- **Class order.** Every adapter needs a LUT test against the robustcd SECOND
  order, as done for ChangeMamba (`SECOND_TO_CM`).

## Integration order

1. **SCanNet repository adapter** → SCanNet (checkpoint validation), then
   Bi-SRNet, HRSCD-str4 and TED from the same code. Cheapest methods; one
   adapter validates three or four.
2. **Mamba-FCS**: reuses the VMamba kernel build and the ChangeMamba adapter
   structure; checkpoint validation.
3. **ChangeMask** (torchange): independent lineage; no checkpoint.
4. **PerASCD**: most expensive (custom op, 4 GB checkpoint, memory); started
   once the pipeline has been exercised on the cheaper models.
5. **CSF-Mamba**: last, by decision.
6. **BCD track** (LEVIR-CD reader first).

## Cost estimate (to be replaced by measured timings)

Per seed at 50 k iterations × effective batch 16 on one L40S, bf16 training.
Only ChangeMamba-T is measured (1.23 s/it). The rest are order-of-magnitude
guesses from backbone size.

| Method | Estimated GPU-h / seed | × 3 seeds |
|---|---|---|
| HRSCD-str4, Bi-SRNet, TED, SCanNet, ChangeMask | 5–10 each | 75–150 total |
| ChangeMamba-T (measured) | 17.5 | 52 |
| CSF-Mamba | ~15–20 | ~50 |
| Mamba-FCS (VMamba-B) | 30–40 | 90–120 |
| PerASCD (ViT-G) | 50–80 | 150–240 |
| BCD track (4 models, LEVIR-CD) | 3–10 each | 40–100 |

Order of magnitude: **450–700 L40S GPU-h** for training, plus evaluation
(≈ 20 degraded test sets × models × seeds, inference only).
