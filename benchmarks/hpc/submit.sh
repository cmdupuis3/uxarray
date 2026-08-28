#!/usr/bin/env bash
# Submits a sharded asv run on derecho as three chained PBS jobs.
#
#   setup   one node, once: fixture cache, machine file, asv environments and
#           the wheel, then the shard plan. Everything the shards would
#           otherwise race each other to create on the filesystem they share.
#   shards  a job array, one node each, exclusive. A ``time_*`` result is only
#           worth having if nothing else is competing for the machine, so one
#           shard per node rather than several.
#   merge   one node, once: combines the shards' results directories and shows
#           the run.
#
# The merge depends on ``afteranyarray`` rather than ``afterokarray`` on
# purpose: a shard that fails still leaves results worth merging, and a suite
# with one long-broken benchmark should not cost you the other eighty.
#
# Usage:
#
#   PBS_ACCOUNT=UXXX0001 ./benchmarks/hpc/submit.sh
#   PBS_ACCOUNT=UXXX0001 THREADS=8 ./benchmarks/hpc/submit.sh
#   PBS_ACCOUNT=UXXX0001 SHARDS=8 REV=main^! ./benchmarks/hpc/submit.sh
#
# THREADS overrides the config's NUMBA_NUM_THREADS and is recorded in the
# environment name, so runs at different thread counts do not overwrite each
# other's results. Leave it unset to take the config's value.
#
set -euo pipefail

: "${PBS_ACCOUNT:?set PBS_ACCOUNT to your project code, e.g. PBS_ACCOUNT=UXXX0001}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHARDS="${SHARDS:-4}"
# A single commit by default. ``asv run`` takes a range, so ``main^!`` is main's
# tip alone and ``base..head`` is every commit between.
REV="${REV:-HEAD^!}"
QUEUE="${QUEUE:-main}"
CONFIG="${CONFIG:-asv.conf.hpc.json}"
# Empty means stage.pbs derives it from NCAR_HOST or the node name.
ASV_MACHINE="${ASV_MACHINE:-}"
WALLTIME="${WALLTIME:-12:00:00}"
SETUP_WALLTIME="${SETUP_WALLTIME:-02:00:00}"
# Off /glade/derecho/scratch so every shard shares one fixture cache. The
# default in ``_fixtures.cache_dir`` follows the checkout, which would give a
# second working tree its own empty cache and re-read every source.
CACHE_DIR="${UXARRAY_BENCH_CACHE_DIR:-/glade/derecho/scratch/$USER/uxarray-bench}"
# Submit from a shell that already has ``asv`` on PATH -- ``-V`` below carries
# that environment into all three jobs, which is both simpler and less brittle
# than reactivating inside them. Set ASV_ACTIVATE only if you would rather the
# jobs do it themselves; it is eval'd once per stage.
export ASV_ACTIVATE="${ASV_ACTIVATE:-true}"
command -v asv >/dev/null || [ "$ASV_ACTIVATE" != "true" ] || {
    echo "asv is not on PATH; activate your environment first, or set ASV_ACTIVATE" >&2
    exit 1
}

STAGE_SCRIPT="$REPO/benchmarks/hpc/stage.pbs"
# Passed through the environment rather than in ``-v``, whose value list is
# comma-separated and so cannot hold an activation command or a path with a
# comma in it.
export REPO SHARDS REV CONFIG ASV_MACHINE
export THREADS="${THREADS:-}"
export UXARRAY_BENCH_CACHE_DIR="$CACHE_DIR"

mkdir -p "$CACHE_DIR"

setup=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-setup \
    -l select=1:ncpus=128 -l walltime="$SETUP_WALLTIME" \
    -V -v "STAGE=setup" "$STAGE_SCRIPT")
echo "setup  $setup"

shards=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-shard \
    -J "0-$((SHARDS - 1))" \
    -l select=1:ncpus=128 -l walltime="$WALLTIME" \
    -W "depend=afterok:$setup" \
    -V -v "STAGE=shard" "$STAGE_SCRIPT")
echo "shards $shards  ($SHARDS of them)"

merge=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-merge \
    -l select=1:ncpus=1 -l walltime=00:30:00 \
    -W "depend=afteranyarray:$shards" \
    -V -v "STAGE=merge" "$STAGE_SCRIPT")
echo "merge  $merge"
echo
echo "watch with:  qstat -u $USER -t"
echo "results in:  $REPO/benchmarks/results/$ASV_MACHINE/"
