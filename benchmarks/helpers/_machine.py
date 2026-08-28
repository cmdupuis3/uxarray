"""Pinning the machine name asv records results under.

asv keys results on a machine name and defaults it to the hostname
(``Machine.get_defaults``), which on a hosted runner is fresh for every job --
``runnervmgx7h7`` on one run, something else on the next. A name that never
repeats cannot be compared across runs, and once the suite is sharded it cannot
even be merged within one run: the file asv writes is
``results/<machine>/<commit>-<env>.json``, so every shard has to agree on
``<machine>`` or there is nothing for :mod:`_merge` to line up.

``asv machine --machine NAME`` will not do it on its own. That command stores
only the fields that differ from the ones it detected and then skips filling the
rest in (``commands/machine.py``), so naming the machine is precisely what drops
``cpu``, ``num_cpu`` and ``ram`` -- the fields that say what the timings were
measured on, and the ones the duration report prints. Detect first with ``asv
machine --yes``, rename after, which is what this does.

Idempotent, so a job that runs it twice, or a machine file that arrives already
pinned, is fine.

Usage::

    asv machine --yes
    python -m benchmarks.helpers._machine --name gh-Linux-X64
"""

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path

__all__ = ["pin"]

_VERSION_KEY = "version"


def default_name():
    """A machine name that survives landing on a different node next time.

    The scheduler puts you on ``derecho3`` one day and ``crhtc70`` the next, and
    asv keys results on ``platform.uname``'s node name, so left alone it records
    a new machine every login and the results scatter across all of them.

    ``NCAR_HOST`` is the reliable answer where it is set -- it names the cluster
    rather than the node, which is the granularity results want. Failing that,
    the node name with its trailing digits removed, which folds ``derecho3`` and
    ``derecho5`` together but *not* ``derecho3`` and ``crhtc70``: login and
    compute nodes of one cluster do not share a stem. Set ``ASV_MACHINE``
    yourself if you move between them without ``NCAR_HOST``.
    """
    for variable in ("ASV_MACHINE", "NCAR_HOST"):
        value = os.environ.get(variable)
        if value:
            return value, variable
    node = platform.node().split(".")[0]
    return re.sub(r"[-_]?\d+$", "", node) or node, None


def default_path():
    """Where asv keeps its machine file (``MachineCollection.get_machine_file_path``)."""
    return Path.home() / ".asv-machine.json"


def pin(name, path=None, hostname=None, sole=False):
    """Renames the machine file's freshly detected entry to ``name``.

    Returns its details. Entries for other machines are left alone -- a runner
    has only the one, but a login node that has recorded every compute node it
    ever landed on should not lose them to a benchmark run.

    The fresh entry is the one keyed by this host's name, since that is what
    ``asv machine --yes`` writes (``Machine.get_defaults`` takes it from
    ``platform.uname``). Renaming it is the whole point: several nodes of one
    cluster should file their results under one machine, or a sharded run has
    nothing to merge. Falls back to a lone entry whatever its name, for a runner
    whose hostname has already been renamed away by an earlier call.
    """
    path = Path(path) if path is not None else default_path()
    hostname = hostname if hostname is not None else platform.node()
    stored = json.loads(path.read_text())
    version = stored.pop(_VERSION_KEY, None)

    if hostname in stored:
        detected = stored.pop(hostname)
    elif name in stored:
        detected = stored[name]
    elif len(stored) == 1:
        (only,) = stored
        detected = stored.pop(only)
    else:
        raise ValueError(
            f"{path} holds {len(stored)} machines ({', '.join(sorted(stored))}), none of "
            f"them this host ({hostname!r}) and none of them {name!r}; cannot tell which "
            f"describes the machine this is running on. Run ``asv machine --yes`` first, "
            f"or pass --name one of the recorded machines"
        )

    detected["machine"] = name
    if sole:
        stored = {}
    stored[name] = detected
    if version is not None:
        stored[_VERSION_KEY] = version
    path.write_text(json.dumps(stored, indent=4))
    return detected


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._machine",
        description="Rename asv's detected machine entry to a fixed name.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Machine name to pin to. Defaults to $ASV_MACHINE, then $NCAR_HOST, "
        "then this host's name with trailing digits removed.",
    )
    parser.add_argument("--path", default=None, help="Machine file (default ~/.asv-machine.json).")
    parser.add_argument(
        "--hostname", default=None, help="Host whose entry to rename (default this one)."
    )
    parser.add_argument(
        "--sole",
        action="store_true",
        help="Drop every other machine from the file. asv falls back to a lone entry "
        "whatever the hostname, so this makes bare ``asv run``/``asv show`` work from "
        "any node without -m. Use it where you only ever benchmark one machine.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the pinned name, for capturing."
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the name that would be pinned and change nothing.",
    )
    args = parser.parse_args(argv)

    name, source = (args.name, "--name") if args.name else default_name()
    if args.print_only:
        print(name)
        return 0
    detected = pin(name, args.path, args.hostname, sole=args.sole)
    if args.quiet:
        print(name)
        return 0
    print(
        f"{name}: {detected.get('cpu', '?')} "
        f"({detected.get('num_cpu', '?')} cpu, {detected.get('os', '?')})"
    )
    if source is None:
        print(
            f"  note: {name!r} came from this host's name. A cluster's login and compute "
            f"nodes do not share a stem, so set ASV_MACHINE (or rely on NCAR_HOST) if you "
            f"benchmark from both, or the results will still split in two.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
