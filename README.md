# robustcd

A reproducible robustness benchmark for bi-temporal change detection (binary
and semantic) under physically grounded product degradations.

Existing robustness benchmarks for Earth observation cover single-date tasks.
Change detection fails in a different way: when the *two dates* of a product
differ in ways unrelated to real change. `robustcd` renders such differences
deterministically and evaluates models on them under one protocol.

## Degradation families

| family | status | docs |
|---|---|---|
| Misregistration between dates (integer, sub-pixel, affine, local warp) | implemented, validated on SECOND test | [docs/misregistration.md](docs/misregistration.md) |
| Radiometric / atmospheric differences | planned | |
| Resolution loss and resampling | planned | |
| Sensor noise | planned | |
| Cross-sensor harmonisation mismatch | planned | |

Every family follows the same conventions. Date 1 is the reference and only
date 2 is degraded. Labels are left untouched, because the degradation is an
error in the product and not a change on the ground. Parameters are seeded per
`(dataset, sample, family, mode, severity)`, so a render is byte-identical on
any machine. Each rendered set carries a per-sample manifest recording exactly
what was applied.

## Metrics

A single implementation of SeK, mIoU, Fscd and binary F1/IoU scores every
model: model repos only write predictions, and `scripts/evaluate.py` scores
them. It matches the widely used Bi-SRNet `SCDD_eval_all` exactly on the full
SECOND test set, and adds image-level bootstrap intervals and paired
clean-vs-degraded comparisons. See [docs/metrics.md](docs/metrics.md), including
the conventions that make published SeK differ between papers.

## Layout

```
robustcd/
  degradations/   # one module per family + measurement utilities + loaders
  datasets/       # dataset readers (SECOND-style layout; dir or .zip)
  metrics/        # SeK / mIoU / Fscd / F1, bootstrap, confusion I/O
scripts/          # renderers, evaluate.py, compare.py, validation CLIs
tests/            # python tests/test_metrics.py (or pytest)
cluster/          # Alliance (Vulcan) env setup, data preparation job, see cluster/README.md
docs/             # design notes and verification per component
results/          # verification outputs
```

## Install

Core dependencies are `numpy`, `scipy` and `pillow` only. `torch` is optional,
imported lazily by the `torch.utils.data.Dataset` wrapper.

Local (uv):

```bash
uv venv && uv pip install -e .
```

Digital Research Alliance clusters (Narval):

```bash
module load python/3.11 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -e .
```

## Quick start

```bash
# render the misregistration grid for SECOND test
python scripts/render_misregistration.py \
    --source /path/to/SECOND/test --out /path/to/degraded \
    --dataset SECOND --split test --jobs 8

# verify the applied displacement against what is measured in the pixels
python scripts/validate_misregistration.py --source /path/to/SECOND/test --limit 24 --jobs 4
```

See [docs/misregistration.md](docs/misregistration.md) for the design
rationale, severity ladders, ground-truth convention and verification results.
