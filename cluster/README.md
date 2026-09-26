# Running on Alliance clusters (Vulcan)

Account: `aip-hervete` (the only SLURM association on Vulcan; `def-hervete` is rejected with "Invalid account"). Code in `$HOME/robustcd` (git), environments in
`$HOME/envs/`, data and outputs in `$SCRATCH/robustcd/`. `$SCRATCH` is purged
after a period of inactivity: the original archives on the workstation stay the
reference copy, and everything under `$SCRATCH/robustcd/data` can be rebuilt
with `prepare_second.sbatch`.

```
$SCRATCH/robustcd/
  raw/SECOND/            SECOND_total_test.zip, SECOND_train_set.rar  (copied from the workstation)
  data/SECOND/{train,test}/{im1,im2,label1,label2}/
  data/SECOND.tar        staging tar for $SLURM_TMPDIR
  checks/                dataset verification reports
  logs/                  SLURM output
```

## 1. Code on the cluster

From a login node (login nodes have internet access):

```bash
cd ~ && git clone <your GitHub repo URL> robustcd     # or: git -C ~/robustcd pull
```

For a private repository, cloning needs either an SSH key on Vulcan registered
with GitHub, or an HTTPS personal access token. The alternative is to copy from
the workstation: `scp -r ~/Projects/robustcd vulcan:~/`.

## 2. Tooling environment (once, login node)

```bash
bash ~/robustcd/cluster/alliance/setup_env.sh
```

This ends by running `tests/test_metrics.py`; all tests should pass (the
reference-equivalence test is skipped on the cluster).

## 3. Extract and verify SECOND (CPU job)

```bash
mkdir -p $SCRATCH/robustcd/logs && cd $SCRATCH/robustcd/logs
sbatch ~/robustcd/cluster/alliance/prepare_second.sbatch
squeue -u $USER                      # wait for it to finish
tail -n 30 prep-second-*.out
```

The job checks the archive sizes, extracts both splits and verifies **every
sample**: stream counts, shapes, strict label decoding, and no-change
consistency between dates. It then compares against the local reference in
`results/data_checks/` (sample count, sha256 of the id list, per-class pixel
counts). It must print `reference: MATCH` for train and test, then `DONE`.

Expected: train 2968 samples, test 1694 samples, 512×512, 0 errors.

### No RAR tool on the cluster

If step 3 exits with `no RAR extractor`, extract on the workstation (`bsdtar`
ships with libarchive on Arch) and send a tar instead:

```bash
mkdir -p /tmp/second/train && bsdtar -xf ~/Téléchargements/second_dataset/SECOND_train_set.rar -C /tmp/second/train
tar -cf /tmp/SECOND_train.tar -C /tmp/second train
scp /tmp/SECOND_train.tar vulcan:/scratch/<user>/robustcd/raw/SECOND/     # absolute path, see `ssh vulcan 'echo $SCRATCH'`
ssh vulcan 'mkdir -p $SCRATCH/robustcd/data/SECOND && tar -xf $SCRATCH/robustcd/raw/SECOND/SECOND_train.tar -C $SCRATCH/robustcd/data/SECOND'
```

Then resubmit `prepare_second.sbatch`. It skips the train extraction because
`train/im1` already exists, and still verifies both splits.

## 4. GPU training environment

Vulcan nodes carry 4 × NVIDIA L40S (48 GB, sm_89), and the scheduler also
exposes GPU shards (`shard:l40s`, 4 per GPU) for small jobs. The Alliance
wheelhouse ships prebuilt `mamba_ssm` and `causal_conv1d` for cp311, so
nothing has to be compiled for the Mamba kernels. Versions are pinned in
`cluster/alliance/requirements-gpu.txt`.

```bash
bash ~/robustcd/cluster/alliance/setup_gpu_env.sh            # login node, once
cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/alliance/gpu_smoke_test.sbatch
tail -n 20 gpu-smoke-*.out
```

The smoke test runs on one L40S. It checks bf16 matmul throughput, compares
the `causal_conv1d` and `selective_scan` CUDA kernels numerically against
their pure-PyTorch references (forward and gradients), runs a `Mamba` block
forward/backward under bf16 autocast, and times the staging of `SECOND.tar`
into `$SLURM_TMPDIR`. It must end with `ALL PASS`. The resolved package
versions are written to `~/envs/robustcd-gpu/freeze.txt`; commit a copy with
the results.

VMamba-based models (ChangeMamba, CSF-Mamba) ship their own selective-scan
CUDA extension, which is not part of this env. It gets built per model repo in
a later step.

## 5. Models

Each model has its own directory with setup, validation and training jobs:

| Directory | Models | Setup |
|---|---|---|
| `cluster/models/changemamba/` | MambaSCD-Tiny | `setup.sh`, then `build_and_validate.sbatch` (kernel build) |
| `cluster/models/ding/` | SCanNet, TED, Bi-SRNet, HRSCD-str4 | `setup.sh`, then `validate.sbatch` |

## Why a staging tar

Training reads ~19k small PNGs per epoch. On Lustre `$SCRATCH` that is slow
and loads the shared filesystem. Jobs should instead do
`tar -xf $SCRATCH/robustcd/data/SECOND.tar -C $SLURM_TMPDIR` at start-up and
read from node-local disk.
