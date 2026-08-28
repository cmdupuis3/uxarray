"""Merging a sharded run's results back into one tree.

asv reads its results file once before running a benchmark set and writes it
once after (``Results.load_data`` then ``Results.save`` in ``commands/run.py``),
and the name it writes is ``results/<machine>/<commit>-<env>.json`` -- one file
per commit and environment, whatever subset of benchmarks the run measured. So
shards sharing a results directory each rewrite that whole file from what they
alone measured, and the last one to finish wins. Each shard therefore gets its
own directory (``_partition --config-out``) and they are combined here.

The combination is a union rather than an element-wise reconciliation, which is
what the by-whole-benchmark split buys: a row is keyed on the benchmark name and
carries its whole parameter sweep inside, so every row is owned by exactly one
shard. Splitting inside a benchmark would have put two shards in one row.

Order is restored rather than preserved. ``results`` is a JSON object, asv writes
it with ``compact=True`` -- which disables sorting, so key order is the order asv
appended to it -- and for an unsharded run that order is ``sorted(benchmarks)``
grouped by ``setup_cache_key`` (``runner.py``, ``iter_run_items``). Shards finish
in whatever order the queue hands back, so :func:`canonical_order` recovers the
order the same suite would have produced serially and every merged file is
written in it.

Idempotent, and indifferent to shards that have not landed: merging the three
directories that exist gives a valid tree, and merging again when the fourth
arrives puts its rows in their proper place. That is what makes it safe to run
from a polling loop as jobs come back rather than only after a barrier.

Usage::

    python -m benchmarks.helpers._merge --out results results.shard*
    python -m benchmarks.helpers._merge --out results --quiet results.shard*
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

__all__ = ["canonical_order", "merge", "merge_benchmarks", "merge_result_files"]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]

MACHINE_FILE = "machine.json"
BENCHMARKS_FILE = "benchmarks.json"
_SPECIAL_FILES = frozenset({MACHINE_FILE, BENCHMARKS_FILE})

# asv stores its own format version alongside the data in both files.
_VERSION_KEY = "version"


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _dump(path, data):
    """Writes ``data`` the way asv writes a results file.

    ``util.write_json(..., compact=True)`` disables both sorting and
    indentation; the sorting is the part that matters, because key order is the
    only place a results file records what ran when.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def canonical_order(benchmarks):
    """Benchmark names in the order an unsharded ``asv run`` would produce them.

    Mirrors ``runner.run_benchmarks``: it walks ``sorted(benchmarks.items())``
    building ``benchmark_order``, a dict keyed on ``setup_cache_key``, then runs
    each of those groups in turn. So the order is by name within a group, and
    groups in the order their first member is reached by name.
    """
    groups = {}
    for name in sorted(benchmarks):
        key = benchmarks[name].get("setup_cache_key")
        groups.setdefault(key, []).append(name)
    return [name for group in groups.values() for name in group]


def merge_benchmarks(shard_dirs):
    """Union of the shards' ``benchmarks.json``.

    A shard discovers under its own ``--bench`` patterns, so each file holds
    only that shard's benchmarks and the full set exists nowhere until here.
    ``_partition.load_benchmarks`` needs that full set to plan the next run.
    """
    merged, version = {}, None
    for shard_dir in shard_dirs:
        path = Path(shard_dir) / BENCHMARKS_FILE
        if not path.is_file():
            continue
        data = _load(path)
        version = data.get(_VERSION_KEY, version)
        for name, value in data.items():
            if name != _VERSION_KEY:
                merged[name] = value
    if version is not None:
        merged[_VERSION_KEY] = version
    return merged


def _pick(name, existing, candidate, report):
    """Which of two rows for one benchmark to keep.

    Only reachable when a name landed in more than one shard, which the
    partition does not do -- so it means the plan the shards ran was not the one
    that produced them. Preferring a row that has a result over one that does
    not, then the later ``started_at``, keeps a re-run over the run it replaced
    instead of picking on file order.
    """
    if existing == candidate:
        return existing

    def rank(row):
        return (row.get("result") is not None, row.get("started_at") or 0)

    keep, drop = (candidate, existing) if rank(candidate) > rank(existing) else (existing, candidate)
    report(
        f"{name}: found in more than one shard with different data; keeping the "
        f"row started at {keep.get('started_at')} over {drop.get('started_at')}"
    )
    return keep


def merge_result_files(datas, order, report):
    """One results file from several shards' versions of it.

    ``datas`` are the parsed files, in shard order; ``order`` is the name order
    to write. Every field outside ``results`` and ``durations`` describes the
    commit and environment rather than the run, and is identical across shards
    by construction, so the first shard's copy carries over untouched.
    """
    merged = dict(datas[0])
    columns = list(merged.get("result_columns") or [])

    rows, durations = {}, {}
    for data in datas:
        # Read each row against its own file's columns. Identical in practice --
        # one asv builds every shard -- but a row is a bare list, so aligning it
        # to the wrong header would silently shift every value.
        shard_columns = data.get("result_columns") or columns
        for name, row in (data.get("results") or {}).items():
            values = dict(zip(shard_columns, row))
            rows[name] = (
                _pick(name, rows[name], values, report) if name in rows else values
            )
        # ``durations`` holds only the ``<build>`` and ``<setup_cache ...>``
        # entries; a benchmark's own duration lives in its row. Every shard pays
        # both, so the max is the one a single run would have reported, and the
        # sum would describe work no single wall clock ever saw.
        for key, value in (data.get("durations") or {}).items():
            durations[key] = max(durations.get(key, 0.0), float(value))

    known = [name for name in order if name in rows]
    extra = sorted(name for name in rows if name not in set(order))
    if extra:
        report(f"{len(extra)} row(s) not in benchmarks.json, appended: {', '.join(extra[:3])}...")

    results = {}
    for name in known + extra:
        row = [rows[name].get(column) for column in columns]
        # asv drops trailing nulls when it writes a row; keeping that keeps the
        # merged file the same size as the one a serial run would have written.
        while row and row[-1] is None:
            row.pop()
        results[name] = row

    merged["results"] = results
    merged["durations"] = durations
    return merged


def merge(shard_dirs, out_dir, report=lambda message: None):
    """Merges ``shard_dirs`` into ``out_dir``. Returns a per-file row count."""
    shard_dirs = [Path(d) for d in shard_dirs]
    out_dir = Path(out_dir)
    resolved_out = out_dir.resolve()
    if any(d.resolve() == resolved_out for d in shard_dirs):
        raise ValueError(f"--out {out_dir} is also a shard directory; refusing to merge in place")

    present = [d for d in shard_dirs if d.is_dir()]
    for missing in [d for d in shard_dirs if not d.is_dir()]:
        report(f"{missing}: not there yet, skipped")
    if not present:
        raise ValueError("no shard directories to merge")

    benchmarks = merge_benchmarks(present)
    order = canonical_order({k: v for k, v in benchmarks.items() if k != _VERSION_KEY})
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(benchmarks) > (1 if _VERSION_KEY in benchmarks else 0):
        _dump(out_dir / BENCHMARKS_FILE, benchmarks)

    # One group per (machine, results file): a shard writes the same file name as
    # every other shard of its commit and environment, which is the collision
    # this module exists to undo.
    groups = {}
    for shard_dir in present:
        for path in sorted(shard_dir.glob("*/*.json")):
            if path.name in _SPECIAL_FILES:
                continue
            groups.setdefault((path.parent.name, path.name), []).append(path)
        for machine_path in sorted(shard_dir.glob(f"*/{MACHINE_FILE}")):
            target = out_dir / machine_path.parent.name / MACHINE_FILE
            if target.is_file() and _load(target) != _load(machine_path):
                report(f"{machine_path}: disagrees with the machine.json already merged")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(machine_path, target)

    counts = {}
    for (machine, filename), paths in sorted(groups.items()):
        merged = merge_result_files(
            [_load(p) for p in paths],
            order,
            lambda message, f=filename: report(f"{f}: {message}"),
        )
        _dump(out_dir / machine / filename, merged)
        counts[f"{machine}/{filename}"] = (len(merged["results"]), len(paths))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._merge",
        description="Merge a sharded run's results directories into one tree.",
    )
    parser.add_argument("shards", nargs="+", help="Shard results directories to merge.")
    parser.add_argument(
        "--out",
        default=str(BENCHMARK_DIR / "results"),
        help="Directory to write the merged tree to (default benchmarks/results).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file notes.")
    args = parser.parse_args(argv)

    def report(message):
        if not args.quiet:
            print(f"  {message}", file=sys.stderr)

    counts = merge(args.shards, args.out, report)
    for name, (rows, shards) in sorted(counts.items()):
        print(f"{name}: {rows} benchmarks from {shards} shard(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
