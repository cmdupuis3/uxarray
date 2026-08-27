"""Splitting the suite into shards of roughly equal cost.

asv runs one benchmark at a time. ``--parallel`` builds environments in
parallel and nothing else -- its own help text is "Build (but don't benchmark)
in parallel" -- so cutting the wall clock means running several ``asv``
processes, and a ``time_*`` result is only worth having if nothing else is
competing for the machine while it is measured. That points at one shard per
runner rather than several per runner, and at this module, whose whole job is to
decide which benchmarks each of those runners should claim.

The split is by whole benchmark, not by parameter combination. asv matches
``--bench`` against the expanded ``name(param0, param1)`` for parameterized
benchmarks, so a finer cut is available -- but measured durations say it is not
needed. The suite is flat: the heaviest single benchmark is about 5% of the
total, and greedy longest-first packing lands within ~1% of a perfect split
even at eight shards. Splitting inside a benchmark would buy nothing and would
put every shard's results in the same row of the same results file, which then
has to be merged element-wise.

Weights come from the ``duration`` asv records per benchmark in the results file
it writes (``Results.save``), so a partition improves as results accumulate
rather than needing a cost model. Benchmarks with no recorded duration -- new
ones, mostly -- get the median of the ones that have, which is a better guess
than either zero or the mean of a long-tailed distribution.

Usage::

    python -m benchmarks.helpers._partition --shards 4
    python -m benchmarks.helpers._partition --shards 4 --shard 0 --bench-args
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

__all__ = ["load_benchmarks", "load_weights", "plan", "bench_regexes"]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]

# ``setup_cache`` groups whose members may be split across shards. Anything else
# is paid once per shard that holds any of its benchmarks, so those move
# together. ``CachedFixtures.setup_cache`` is just ``prime()``, a stat per file
# once the cache is warm (1.2s in CI), and ``None`` is no setup_cache at all.
SPLITTABLE_PREFIX = "helpers._fixtures:"

_SKIP_FILES = frozenset({"machine.json", "benchmarks.json"})


def _splittable(setup_cache_key):
    """Whether benchmarks sharing this ``setup_cache_key`` may land in different shards."""
    return setup_cache_key is None or str(setup_cache_key).startswith(SPLITTABLE_PREFIX)


def load_benchmarks(results_dir):
    """The discovered benchmarks, as asv wrote them to ``benchmarks.json``."""
    path = Path(results_dir) / "benchmarks.json"
    with open(path) as handle:
        discovered = json.load(handle)
    # asv stores its own format version alongside the benchmarks.
    return {name: value for name, value in discovered.items() if name != "version"}


def load_weights(results_dirs):
    """Mean recorded duration per benchmark, in seconds, over the files on disk.

    The mean rather than the latest: a benchmark's first run on a cold numba
    cache can cost hundreds of times its warm cost (a 9.6s bounds compile
    against 13ms warm, in one observed run), and a partition built from one
    such outlier sends a whole shard chasing work that is not there.
    """
    samples = {}
    for results_dir in results_dirs:
        root = Path(results_dir)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.json")):
            if path.name in _SKIP_FILES:
                continue
            try:
                with open(path) as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            columns = data.get("result_columns") or []
            if "duration" not in columns:
                continue
            index = columns.index("duration")
            for name, row in (data.get("results") or {}).items():
                if len(row) <= index or row[index] is None:
                    continue
                samples.setdefault(name, []).append(float(row[index]))
    return {name: statistics.fmean(values) for name, values in samples.items()}


def plan(benchmarks, n_shards, weights=None):
    """Partitions ``benchmarks`` into ``n_shards`` lists of names.

    Greedy longest-first onto the lightest shard so far -- the standard LPT
    heuristic, which on a distribution this flat is within about 1% of optimal
    and, unlike anything smarter, is obvious enough to debug from the report.

    Deterministic: equal weights are broken by name, so the same inputs always
    give the same shards and a shard can compute its own membership without
    being told.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be at least 1, got {n_shards}")
    weights = dict(weights or {})
    known = [value for value in weights.values() if value > 0]
    default = statistics.median(known) if known else 1.0

    # Group anything sharing an expensive setup_cache, so it is paid once.
    units = {}
    for name, benchmark in benchmarks.items():
        key = benchmark.get("setup_cache_key")
        unit = name if _splittable(key) else f"setup_cache:{key}"
        units.setdefault(unit, []).append(name)

    costs = {
        unit: sum(weights.get(name, default) for name in names)
        for unit, names in units.items()
    }

    shards = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for unit in sorted(units, key=lambda u: (-costs[u], u)):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        shards[target].extend(sorted(units[unit]))
        loads[target] += costs[unit]
    return shards


def bench_regexes(names):
    """``--bench`` patterns selecting exactly ``names`` and nothing else.

    asv filters a parameterized benchmark on ``name(param0, param1)`` and an
    unparameterized one on ``name`` (``Benchmarks.__init__``), so the trailing
    group has to admit both an open parenthesis and end-of-string. Without it
    ``^name$`` silently matches none of a parameterized benchmark's
    combinations, and the shard runs nothing.
    """
    return [f"^{re.escape(name)}($|\\()" for name in names]


def _report(benchmarks, shards, weights):
    known = [value for value in weights.values() if value > 0]
    default = statistics.median(known) if known else 1.0
    total = sum(weights.get(name, default) for name in benchmarks)
    print(
        f"{len(benchmarks)} benchmarks, {len(weights)} with recorded durations, "
        f"{total / 60:.1f} min of work; median fallback {default:.1f}s"
    )
    loads = [sum(weights.get(name, default) for name in shard) for shard in shards]
    ideal = total / len(shards) if shards else 0.0
    for index, (shard, load) in enumerate(zip(shards, loads)):
        drift = 100 * (load - ideal) / ideal if ideal else 0.0
        print(f"  shard {index}: {len(shard):3} benchmarks  {load / 60:5.1f} min  {drift:+5.1f}%")
    if loads and ideal:
        print(
            f"  slowest shard {max(loads) / 60:.1f} min against an ideal "
            f"{ideal / 60:.1f}; speedup {total / max(loads):.2f}x of a possible {len(shards)}x"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._partition",
        description="Split the benchmark suite into shards of roughly equal cost.",
    )
    parser.add_argument("--shards", type=int, default=4, help="Number of shards (default 4).")
    parser.add_argument(
        "--shard", type=int, default=None, help="Report only this shard, by index."
    )
    parser.add_argument(
        "--results",
        action="append",
        default=None,
        help="Results directory to read durations and benchmarks.json from. "
        "Repeatable; defaults to benchmarks/results.",
    )
    parser.add_argument(
        "--bench-args",
        action="store_true",
        help="Print the shard's --bench arguments for asv, rather than a report.",
    )
    args = parser.parse_args(argv)

    results_dirs = args.results or [str(BENCHMARK_DIR / "results")]
    benchmarks = load_benchmarks(results_dirs[0])
    weights = load_weights(results_dirs)
    shards = plan(benchmarks, args.shards, weights)

    if args.bench_args:
        if args.shard is None:
            parser.error("--bench-args needs --shard")
        for pattern in bench_regexes(shards[args.shard]):
            print("--bench", pattern)
        return 0

    if args.shard is None:
        _report(benchmarks, shards, weights)
    else:
        for name in shards[args.shard]:
            print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
