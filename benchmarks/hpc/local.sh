#!/usr/bin/env bash
# Runs the sharded suite on the node you are already sitting on -- an
# interactive PBS session, typically -- with the shards concurrent rather than
# queued as separate jobs.
#
# This only makes sense because a derecho CPU node has 128 cores and a shard at
# NUMBA_NUM_THREADS=8 wants nine of them. Each shard is pinned to its own slice
# so they cannot land on each other's cores; what they do still share is memory
# bandwidth and last-level cache, and for the grid operations in this suite that
# is not nothing. So: the BASE-vs-HEAD ratios ``asv compare`` reports stay
# usable, since both sides of a comparison run inside the same shard under the
# same contention, but absolute timings come out noisier than a run that had a
# node to itself. Take those from one-shard-per-node (``submit.sh``).
#
# Usage, from the repository root:
#
#   ./benchmarks/hpc/local.sh
#   SHARDS=8 THREADS=4 ./benchmarks/hpc/local.sh
#   REV=main^! ./benchmarks/hpc/local.sh
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHARDS="${SHARDS:-4}"
THREADS="${THREADS:-8}"
export REPO SHARDS THREADS
export REV="${REV:-HEAD^!}"
export CONFIG="${CONFIG:-asv.conf.hpc.json}"
# Left empty on purpose: stage.pbs derives it, so this works unchanged on
# derecho and casper both.
export ASV_MACHINE="${ASV_MACHINE:-}"
export ASV_ACTIVATE="${ASV_ACTIVATE:-true}"
# Also evaluated here, not just in the stages, so the core count below can be
# read with python rather than with a shell tool that lies about it.
eval "$ASV_ACTIVATE"

STAGE_SCRIPT="$REPO/benchmarks/hpc/stage.pbs"
LOGS="${LOGS:-$REPO/benchmarks/hpc/logs}"
mkdir -p "$LOGS"

# Neither PBS's NCPUS nor ``nproc`` can be trusted for this. NCPUS is the ncpus
# *requested per chunk*, 1 for a plain ``qsub -I``, and says nothing about the
# node. And GNU ``nproc`` honours OMP_NUM_THREADS and OMP_THREAD_LIMIT, so in a
# session that sets either it reports the OpenMP thread limit -- 1, or 2 -- and
# not the machine's cores at all.
#
# The affinity mask is the real answer: the CPUs this process may actually run
# on. It respects a cpuset the scheduler imposed and ignores OpenMP entirely.
# Override with CORES to hold the run to fewer than the mask allows.
CORES="${CORES:-$(python -c '
import os
try:
    print(len(os.sched_getaffinity(0)))
except AttributeError:   # not Linux
    print(os.cpu_count() or 1)
')}"
if ! [ "$CORES" -ge 1 ] 2>/dev/null; then
    echo "could not work out a core count (got ${CORES:-empty}); set CORES" >&2
    exit 1
fi
PER=$((CORES / SHARDS))
if [ "$PER" -lt "$((THREADS + 1))" ]; then
    echo "warning: $CORES cores over $SHARDS shards is $PER each, under the" >&2
    echo "         $THREADS threads a shard wants; they will oversubscribe" >&2
fi

PIN=""
if command -v taskset >/dev/null; then
    PIN="taskset"
else
    echo "warning: no taskset, shards will not be pinned and will drift across cores" >&2
fi

echo "== setup =="
STAGE=setup bash "$STAGE_SCRIPT" 2>&1 | tee "$LOGS/setup.log"

echo "== $SHARDS shards, $PER cores each, $THREADS threads each =="
pids=()
for S in $(seq 0 $((SHARDS - 1))); do
    lo=$((S * PER))
    hi=$((lo + PER - 1))
    if [ -n "$PIN" ]; then
        SHARD=$S STAGE=shard taskset -c "$lo-$hi" bash "$STAGE_SCRIPT" \
            >"$LOGS/shard$S.log" 2>&1 &
    else
        SHARD=$S STAGE=shard bash "$STAGE_SCRIPT" >"$LOGS/shard$S.log" 2>&1 &
    fi
    pid=$!
    pids+=("$pid")
    echo "  shard $S -> cores $lo-$hi, pid $pid, log $LOGS/shard$S.log"
done

# Every shard is waited on and its status reported, but a failure does not stop
# the merge: a tree missing one shard's rows is still worth having, same as the
# ``afteranyarray`` dependency the PBS path uses.
failed=0
for S in $(seq 0 $((SHARDS - 1))); do
    if wait "${pids[$S]}"; then
        echo "  shard $S ok"
    else
        echo "  shard $S FAILED (see $LOGS/shard$S.log)" >&2
        failed=$((failed + 1))
    fi
done

echo "== merge =="
STAGE=merge bash "$STAGE_SCRIPT" 2>&1 | tee "$LOGS/merge.log"
[ "$failed" -eq 0 ] || echo "$failed shard(s) failed; merged what landed" >&2
