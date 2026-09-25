# Running on Alliance clusters (Vulcan)

Account: `def-hervete`. Code in `$HOME/robustcd` (git), environments in
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

## Why a staging tar

Training reads ~19k small PNGs per epoch. On Lustre `$SCRATCH` that is slow
and loads the shared filesystem. Jobs should instead do
`tar -xf $SCRATCH/robustcd/data/SECOND.tar -C $SLURM_TMPDIR` at start-up and
read from node-local disk.
