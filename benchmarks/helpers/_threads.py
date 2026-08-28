"""Choosing how many threads a benchmark run may use.

numba reads ``NUMBA_NUM_THREADS`` once, when it brings up its threading layer,
and treats it as a ceiling rather than a setting: ``set_num_threads`` can hold
the pool lower for a block -- which is what :func:`~benchmarks.helpers._peakmem.numba_threads`
does while tracing -- but asking for more than the ceiling raises
``ValueError``. So a run's thread count has to be decided before the benchmark
process imports numba, which means the environment, which is what this module
resolves.

asv copies ``os.environ`` into the processes it launches
(``Environment.run_executable``), so exporting the variable ahead of ``asv run``
is enough::

    export NUMBA_NUM_THREADS=$(python -m benchmarks.helpers._threads)
    asv run ...

One asymmetry to know about: asv layers the ``env_nobuild`` matrix *over* the
inherited environment, not under it, so a variable named in ``asv.conf.json``
overrides the shell. ``NUMBA_THREADING_LAYER`` is named there because
fork-safety is not negotiable. The thread count deliberately is not, so a node
with more cores than a CI runner can decide for itself.

The default is physical cores rather than ``os.cpu_count()``. These kernels are
floating-point and memory-bound, and a second hardware thread per core tends to
cost more in contention than it recovers in latency hiding: CI runners with 2
physical cores plus SMT measured about 1.28x slower per thread on the same
scalar kernels than runners with 4 real cores.

``UXARRAY_BENCH_THREADS`` overrides the default -- an integer, or ``physical``
or ``logical`` to name a rule rather than a number.
"""

import os
import subprocess
import sys

__all__ = ["logical_cores", "physical_cores", "resolve"]

_ENV_VAR = "UXARRAY_BENCH_THREADS"


def logical_cores():
    """Schedulable CPUs, honouring any affinity mask this process was given.

    ``os.cpu_count()`` reports the machine; ``os.sched_getaffinity`` reports
    what this process may actually use, which is the smaller and more useful
    number under a batch scheduler or a ``taskset``.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def physical_cores():
    """Cores rather than hardware threads, or the logical count if unknown.

    Deliberately shells out rather than adding a dependency on ``psutil``: the
    benchmark environment asv builds is defined by ``asv.conf.json``'s matrix,
    and a helper that has to run in it is not worth an entry there.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            return max(1, int(out.strip()))
        if sys.platform.startswith("linux"):
            # One line per logical CPU, "<cpu>,<core>,<socket>,..."; distinct
            # (socket, core) pairs are the physical cores. Counting distinct
            # core ids alone would collapse two sockets into one.
            out = subprocess.run(
                ["lscpu", "-p=core,socket"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            pairs = {
                line for line in (l.strip() for l in out.splitlines())
                if line and not line.startswith("#")
            }
            if pairs:
                return max(1, len(pairs))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return logical_cores()


def resolve(spec=None):
    """The thread count to run with.

    ``spec`` defaults to ``$UXARRAY_BENCH_THREADS``, and that to ``physical``.
    Anything unrecognised falls back to the physical core count rather than
    failing: this sits in front of a benchmark run, and refusing to start
    because a variable is misspelt costs more than quietly doing the sensible
    thing. The resolved number is echoed to stderr so it appears in the log.
    """
    if spec is None:
        spec = os.environ.get(_ENV_VAR, "").strip()
    spec = (spec or "physical").lower()

    if spec == "logical":
        return logical_cores()
    if spec != "physical":
        try:
            return max(1, int(spec))
        except ValueError:
            print(
                f"{_ENV_VAR}={spec!r} is not an integer, 'physical' or 'logical'; "
                "using the physical core count",
                file=sys.stderr,
            )
    # Never hand back more than this process may schedule on.
    return min(physical_cores(), logical_cores())


if __name__ == "__main__":
    print(resolve())
