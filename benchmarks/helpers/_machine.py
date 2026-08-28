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
import sys
from pathlib import Path

__all__ = ["pin"]

_VERSION_KEY = "version"


def default_path():
    """Where asv keeps its machine file (``MachineCollection.get_machine_file_path``)."""
    return Path.home() / ".asv-machine.json"


def pin(name, path=None):
    """Renames the machine file's single entry to ``name``. Returns its details.

    Raises if there is more than one entry and none of them is ``name`` already:
    with several to choose from there is no way to tell which one describes the
    machine this is running on, and picking wrong would label the results with
    another machine's hardware.
    """
    path = Path(path) if path is not None else default_path()
    stored = json.loads(path.read_text())
    version = stored.pop(_VERSION_KEY, None)

    if name in stored:
        detected = stored[name]
    elif len(stored) == 1:
        (detected,) = stored.values()
    else:
        raise ValueError(
            f"{path} holds {len(stored)} machines ({', '.join(sorted(stored))}) and none "
            f"is {name!r}; cannot tell which describes this machine"
        )

    detected["machine"] = name
    merged = {name: detected}
    if version is not None:
        merged[_VERSION_KEY] = version
    path.write_text(json.dumps(merged, indent=4))
    return detected


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._machine",
        description="Rename asv's detected machine entry to a fixed name.",
    )
    parser.add_argument("--name", required=True, help="Machine name to pin to.")
    parser.add_argument("--path", default=None, help="Machine file (default ~/.asv-machine.json).")
    args = parser.parse_args(argv)

    detected = pin(args.name, args.path)
    print(
        f"{args.name}: {detected.get('cpu', '?')} "
        f"({detected.get('num_cpu', '?')} cpu, {detected.get('os', '?')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
