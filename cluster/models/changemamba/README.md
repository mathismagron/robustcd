# ChangeMamba (MambaSCD-Tiny) under the robustcd protocol

Upstream: https://github.com/ChenHongruixuan/ChangeMamba (Apache-2.0), pinned at
`9ce9cec13f9ea14bc0ad91f071577ec9b3a97983`. Weights: Zenodo record 15479555
(CC-BY-4.0), md5-verified.

**Variant: MambaSCD-Tiny.** Its encoder is VMamba-Tiny with the same ImageNet
initialisation as CSF-Mamba, so the two Mamba models are compared at matched
encoder size and pretraining. Small and Base can be added later as a scale
ablation.

## What is upstream and what is ours

| part | source |
|---|---|
| model, loss (`SCDTrainer.train_step`), AdamW (lr 1e-4, wd 5e-3), StepLR (10k, 0.5), dataset class, geometric augmentation, normalisation | upstream, unchanged |
| training loop: seeds, epoch-wise resumable order, bf16 autocast, accumulation, time guard | `robustcd/adapters/changemamba/train.py` |
| checkpoint selection: robustcd SeK on `splits/SECOND/val.txt` every 2k iterations (upstream: 37-class SeK on the **test set** every 500) | ours |
| prediction export in SECOND class order, fp32, no TTA, scored by `scripts/evaluate.py` | `predict.py` |
| label conversion, SECOND ↔ ChangeMamba class order (they differ, see `robustcd/adapters/changemamba/__init__.py`) | `prepare.py` |
| semantic decoding: argmax over classes 1..6 (upstream: over 0..6, where class 0 is never trained) | ours; `--decode full` reproduces upstream |
| selective-scan CUDA extension, rebuilt with an extra `sm_89` target for L40S | upstream source, one-line build patch |

Verified locally on CPU, with a PyTorch scan in place of the CUDA kernel:
- the full train → validate → checkpoint → resume → time-guard exit (code 3) → resume → predict → evaluate path runs;
- the ImageNet backbone loads (200/200 tensors identical);
- the released MambaSCD-Tiny checkpoint loads with 0 missing, mismatched or unexpected keys;
- the class tables are checked against both colour maps (`tests/test_changemamba_adapter.py`);
- a resumed run replays the same data order.

Not verified locally: multi-worker data loading (the local sandbox cannot run
DataLoader workers) and anything on GPU.

**Validated on Vulcan** (job 1187205, L40S; `results/model_checks/changemamba_tiny_released.json`):
- the kernel matches the PyTorch scan: fwd 1.7e-6, bwd 1.6e-6, bf16 input 4.5e-3;
- the released checkpoint with upstream decoding gives from-to SeK **0.2208**,
  which reproduces the published value exactly;
- the same checkpoint under the SECOND definition scores **SeK 0.2334
  [0.2217, 0.2447]**, mIoU 0.7333, Fscd 0.6344;
- restricted and full decodings give identical scores.

## Steps on Vulcan

```bash
# 1. login node, once: code @ pinned commit, extra deps, weights
bash ~/robustcd/cluster/models/changemamba/setup.sh

# 2. GPU, ~30 min: build kernel (sm_89), numeric kernel check, reproduce the released
#    checkpoint's published from-to SeK (0.2208) with upstream decoding, then robustcd scores
cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/changemamba/build_and_validate.sbatch

# 3. pilot = seed 0 at 1.5 x budget (pre-registered budget rule, docs/protocol.md)
sbatch --export=ALL,SEED=0,MAX_ITERS=75000 ~/robustcd/cluster/models/changemamba/train.sbatch
```

Step 3 only after step 2 prints `PASS selective_scan_cuda_oflex` and a
from-to SeK close to 0.2208. The published value was selected on the test set
at upstream's own best iteration, so a close match validates data, class
order, normalisation and inference; it is not expected to be bit-exact.

Outputs of a run (`$SCRATCH/robustcd/runs/changemamba_tiny/seed<k>/`):
`run_config.json`, `train_log.jsonl`, `val_log.jsonl`, `best_model.pth`
(best val SeK within the budget), `last_model_budget.pth` and
`best_model_extended.pth` (pilot only), `latest.pth` (for resuming),
`summary.json`, `slurm_jobs.txt`.
