"""End-to-end ``ux.open_grid``, from a grid file on disk, across mesh sizes.

Reading the file is the subject here, so every benchmark opens the original
rather than the ``_io_cache`` copies the rest of the suite works from. Repeat
samples find the file in the OS page cache, so these measure a warm read, not
one off cold storage.

Each measurement comes in two flavors:

- ``open_grid``: what the call itself costs. The MPAS reader converts
  connectivity eagerly, but can leave other variables backed lazily by the file.
- ``open_grid_load``: the call plus ``Grid._ds.load()``, from file to a fully
  in-memory ``Grid``. Without it, a change that makes the reader lazier would
  read as a speedup when the cost has only moved downstream.
"""

import uxarray as ux

from .helpers._fixtures import ALL_RESOLUTIONS, GRIDS_BY_RESOLUTION, OQU_RESOLUTIONS
from .helpers._memsize import grid_nbytes
from .helpers._peakmem import peak_allocated
from .helpers._warmup import assert_numba_warm, warm_in_parent


def _open(resolution):
    return ux.open_grid(GRIDS_BY_RESOLUTION[resolution])


def _open_load(resolution):
    uxgrid = _open(resolution)
    uxgrid._ds.load()
    return uxgrid


class _OpenGridBenchmark:
    param_names = ["resolution"]
    params = [ALL_RESOLUTIONS]

    # asv's 60s default does not cover reading the DYAMOND grids.
    timeout = 1800


class OpenGrid(_OpenGridBenchmark):
    """Time to open a grid file."""

    # One call per sample, and no warmup calls: the parent has already warmed
    # the path, and at 3.75km a single call is the whole budget.
    number = 1
    warmup_time = 0
    repeat = (1, 10, 60.0)

    def time_open_grid(self, resolution):
        with assert_numba_warm():
            _open(resolution)

    def time_open_grid_load(self, resolution):
        with assert_numba_warm():
            _open_load(resolution)

    def track_nbytes_open_grid(self, resolution):
        """Footprint of the opened ``Grid``, lazily backed variables included,
        to read ``OpenGridTracemalloc`` against."""
        return grid_nbytes(_open(resolution))

    track_nbytes_open_grid.unit = "bytes"


class OpenGridTracemalloc(_OpenGridBenchmark):
    """High-water allocation of opening a grid file.

    tracemalloc sees numpy's allocations, which is where the grid lives, but not
    the netCDF/HDF5 libraries' own caches and conversion buffers. Those are
    capped by library configuration rather than mesh size for the oQU grids,
    which are netCDF-3 or uncompressed netCDF-4 and read straight into the
    destination array, so this undercounts by a roughly constant amount and
    still tracks how the cost scales. Revisit if a grid turns out compressed and
    chunked, where decompression buffers grow with the chunk size; the DYAMOND
    grids' layout has not been checked.
    """

    unit = "bytes"

    def track_peakmem_open_grid(self, resolution):
        with assert_numba_warm():
            return peak_allocated(lambda: _open(resolution))

    def track_peakmem_open_grid_load(self, resolution):
        with assert_numba_warm():
            return peak_allocated(lambda: _open_load(resolution))


def _warm_open_grid():
    """Runs the whole open-and-load path once on each oQU grid.

    No Numba kernel sits on that path today; this runs regardless, so one added
    later is compiled here and inherited by every fork, and
    :func:`assert_numba_warm` fails any benchmark that still compiles one. It
    also settles xarray's and uxarray's deferred imports. Both oQU grids, so the
    netCDF-3 and netCDF-4 read paths are each taken; the DYAMOND grids are too
    large to open in the parent. Nothing is kept, so forks inherit no grid.
    """
    for resolution in OQU_RESOLUTIONS:
        _open_load(resolution)


warm_in_parent(_warm_open_grid, "the open_grid path")
