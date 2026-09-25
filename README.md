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

## Layout

```
robustcd/
  degradations/   # one module per family + measurement utilities + loaders
  datasets/       # dataset readers (SECOND-style layout; dir or .zip)
scripts/          # offline renderers and validation CLIs
docs/             # per-family design notes and verification figures
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
