import numpy as np
import xarray as xr
from numba import njit, prange

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.utils import (
    MIN_ADAPTIVE_SORT_SIZE,
    _adaptive_sort_bucket,
    _insertion_sort_bucket,
)


def close_face_nodes(face_node_connectivity, n_face, n_max_face_nodes):
    """Closes (``face_node_connectivity``) by inserting the first node index
    after the last non-fill-value node.

    Parameters
    ----------
    face_node_connectivity : np.ndarray
        Connectivity array for constructing a face from its nodes
    n_face : constant
        Number of faces
    n_max_face_nodes : constant
        Max number of nodes that compose a face

    Returns
    ----------
    closed : ndarray
        Closed (padded) face_node_connectivity

    Example
    ----------
    Given face nodes with shape [2 x 5]
        [0, 1, 2, 3, FILL_VALUE]
        [4, 5, 6, 7, 8]
    Pads them to the following with shape [2 x 6]
        [0, 1, 2, 3, 0, FILL_VALUE]
        [4, 5, 6, 7, 8, 4]
    """

    # padding to shape [n_face, n_max_face_nodes + 1]
    closed = np.ones((n_face, n_max_face_nodes + 1), dtype=INT_DTYPE) * INT_FILL_VALUE

    # set all non-paded values to original face nodee values
    closed[:, :-1] = face_node_connectivity.copy()

    # instance of first fill value
    first_fv_idx_2d = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # 2d to 1d index for np.put()
    first_fv_idx_1d = first_fv_idx_2d + ((n_max_face_nodes + 1) * np.arange(0, n_face))

    # column of first node values
    first_node_value = face_node_connectivity[:, 0].copy()

    # insert first node column at occurrence of first fill value
    np.put(closed.ravel(), first_fv_idx_1d, first_node_value)

    return closed


def _replace_fill_values(grid_var, original_fill, new_fill, new_dtype=None):
    """Replaces all instances of the current fill value (``original_fill``) in
    (``grid_var``) with (``new_fill``) and converts to the dtype defined by
    (``new_dtype``)

    Parameters
    ----------
    grid_var : xr.DataArray
        Grid variable to be modified
    original_fill : constant
        Original fill value used in (``grid_var``)
    new_fill : constant
        New fill value to be used in (``grid_var``)
    new_dtype : np.dtype, optional
        New data type to convert (``grid_var``) to

    Returns
    -------
    grid_var : xr.DataArray
        Modified DataArray with updated fill values and dtype
    """

    # Identify fill value locations
    if original_fill is not None and np.isnan(original_fill):
        # For NaN fill values
        fill_val_idx = grid_var.isnull()
        # Temporarily replace NaNs with a placeholder if dtype conversion is needed
        if new_dtype is not None and np.issubdtype(new_dtype, np.floating):
            grid_var = grid_var.fillna(0.0)
        else:
            # Choose an appropriate placeholder for non-floating types
            grid_var = grid_var.fillna(new_fill)
    else:
        # For non-NaN fill values
        fill_val_idx = grid_var == original_fill

    # Convert to the new data type if specified
    if new_dtype is not None and new_dtype != grid_var.dtype:
        grid_var = grid_var.astype(new_dtype)

    # Validate that the new_fill can be represented in the new_dtype
    if new_dtype is not None:
        if np.issubdtype(new_dtype, np.integer):
            int_min = np.iinfo(new_dtype).min
            int_max = np.iinfo(new_dtype).max
            if not (int_min <= new_fill <= int_max):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by integer dtype: {new_dtype}"
                )
        elif np.issubdtype(new_dtype, np.floating):
            if not (
                np.isnan(new_fill)
                or (np.finfo(new_dtype).min <= new_fill <= np.finfo(new_dtype).max)
            ):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by float dtype: {new_dtype}"
                )
        else:
            raise ValueError(f"Data type {new_dtype} not supported for grid variables")

    grid_var = grid_var.where(~fill_val_idx, new_fill)

    return grid_var


def _populate_n_nodes_per_face(grid):
    """Constructs the connectivity variable (``n_nodes_per_face``) and stores
    it within the internal (``Grid._ds``) and through the attribute
    (``Grid.n_nodes_per_face``)."""

    n_nodes_per_face = (
        (grid.face_node_connectivity != INT_FILL_VALUE).sum(axis=1).astype(INT_DTYPE)
    )

    if n_nodes_per_face.ndim == 0:
        # convert scalar value into a [1, 1] array
        n_nodes_per_face = np.expand_dims(n_nodes_per_face, 0)

    # add to internal dataset
    grid._ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face,
        dims=ugrid.N_NODES_PER_FACE_DIMS,
        attrs=ugrid.N_NODES_PER_FACE_ATTRS,
    )


def _populate_edge_node_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""

    # Check edge coordinates already exist, if they do this might cause issues

    if "n_edge" in grid.sizes:
        # TODO: raise a warning or exception?
        pass

    # HACK: this is lieu of an xarray equivalent to `da.compute(a, b)`
    computed = xr.Dataset(
        {
            "face_nodes": grid.face_node_connectivity.variable,
            "n_nodes_per_face": grid.n_nodes_per_face.variable,
        }
    ).compute()

    edge_nodes, face_edges = _build_edge_node_connectivity(
        computed.face_nodes.data, computed.n_nodes_per_face.data, grid.n_node
    )

    grid._ds["edge_node_connectivity"] = xr.DataArray(
        edge_nodes,
        dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
    )

    grid._ds["face_edge_connectivity"] = xr.DataArray(
        face_edges,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True, parallel=True)
def _build_edge_node_connectivity(face_node_connectivity, n_nodes_per_face, n_node):
    """Constructs the ``edge_node_connectivity`` variable, which represents the indices of the two nodes that make up
    each edge. Additionally, the ``face_edge_connectivity`` is derived during construction,  which represents the
    indices of the edges that make up each face.

    Each edge is stored as an ascending ``(node_a, node_b)`` pair, and the edges are numbered in lexicographic
    order of that pair.

    Parameters
    ----------
    face_node_connectivity : np.ndarray
        Face Node Connectivity
    n_nodes_per_face : np.ndarray
        Number of nodes/edges per face
    n_node : int
        Total number of nodes, used as the number of buckets for the counting sort

    Returns
    -------
    edge_node_connectivity : np.ndarray
        Edge Node Connectivity with shape (n_edge, 2)
    face_edge_connectivity : np.ndarray
        Face Edge Connectivity with shape (n_face, n_max_face_edges)

    """

    n_face, n_max_face_nodes = face_node_connectivity.shape

    # Keep track of face_edge_connectivity
    face_edge_connectivity = np.full_like(
        face_node_connectivity, INT_FILL_VALUE, dtype=INT_DTYPE
    )

    n_half_edge = np.sum(n_nodes_per_face)

    if n_half_edge == 0:
        return np.empty((0, 2), dtype=INT_DTYPE), face_edge_connectivity

    # Count how many half edges fall into each ``start_node`` bucket, then prefix sum so that
    # ``bucket_bounds[a]`` is where bucket ``a`` starts
    bucket_bounds = np.zeros(n_node + 1, dtype=INT_DTYPE)
    for face_idx in range(n_face):
        n_edges = n_nodes_per_face[face_idx]
        for current_node in range(n_edges):
            start_node = face_node_connectivity[face_idx, current_node]
            end_node = face_node_connectivity[face_idx, (current_node + 1) % n_edges]
            bucket_bounds[min(start_node, end_node) + 1] += 1
    for i in range(n_node):
        bucket_bounds[i + 1] += bucket_bounds[i]

    # Scatter the half edges into their buckets. This advances each entry of
    # ``bucket_bounds`` to the *end* of its bucket, so afterwards bucket ``a`` spans
    # ``bucket_bounds[a - 1]`` up to ``bucket_bounds[a]``, with bucket 0 starting at 0
    order = np.empty(n_half_edge, dtype=INT_DTYPE)
    end_node_keys = np.empty(n_half_edge, dtype=INT_DTYPE)
    for face_idx in range(n_face):
        n_edges = n_nodes_per_face[face_idx]
        for current_node in range(n_edges):
            start_node = face_node_connectivity[face_idx, current_node]
            end_node = face_node_connectivity[face_idx, (current_node + 1) % n_edges]

            if start_node > end_node:
                end_node, start_node = start_node, end_node

            slot = bucket_bounds[start_node]
            order[slot] = face_idx * n_max_face_nodes + current_node
            end_node_keys[slot] = end_node
            bucket_bounds[start_node] = slot + 1

    # Sort each bucket by ``node_b`` and count its unique edges while the bucket is in
    # cache. Buckets are disjoint, so this runs one bucket per thread.
    unique_per_bucket = np.empty(n_node, dtype=INT_DTYPE)
    for n in prange(n_node):
        bucket_start = bucket_bounds[n - 1] if n > 0 else 0
        bucket_end = bucket_bounds[n]

        size = bucket_end - bucket_start
        if size > MIN_ADAPTIVE_SORT_SIZE:
            # Large enough that a bad ordering would be worth catching, which only a
            # collapsed pole or a similarly degenerate node reaches
            _adaptive_sort_bucket(end_node_keys, order, bucket_start, size)
        elif size > 1:
            _insertion_sort_bucket(end_node_keys, order, bucket_start, size)

        n_unique = 0
        prev_b = INT_FILL_VALUE
        for i in range(bucket_start, bucket_end):
            if end_node_keys[i] != prev_b:
                n_unique += 1
                prev_b = end_node_keys[i]

        unique_per_bucket[n] = n_unique

    edge_offset = np.empty(n_node + 1, dtype=INT_DTYPE)
    n_edge = 0
    for n in range(n_node):
        edge_offset[n] = n_edge
        n_edge += unique_per_bucket[n]
    edge_offset[n_node] = n_edge

    # Duplicate half edges are now adjacent, so a single walk assigns each unique edge its
    # index and populates the face edge connectivity.
    edge_node_connectivity = np.empty((n_edge, 2), dtype=INT_DTYPE)

    for n in prange(n_node):
        bucket_start = bucket_bounds[n - 1] if n > 0 else 0
        bucket_end = bucket_bounds[n]

        edge_idx = edge_offset[n] - 1
        prev_b = INT_FILL_VALUE

        for i in range(bucket_start, bucket_end):
            flat_idx = order[i]
            end_node = end_node_keys[i]

            if end_node != prev_b:
                # Only store unique edges
                edge_idx += 1
                edge_node_connectivity[edge_idx, 0] = n
                edge_node_connectivity[edge_idx, 1] = end_node
                prev_b = end_node

            face_edge_connectivity[
                flat_idx // n_max_face_nodes, flat_idx % n_max_face_nodes
            ] = edge_idx

    return edge_node_connectivity, face_edge_connectivity


def _populate_edge_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""
    edge_faces = _build_edge_face_connectivity(
        grid.face_edge_connectivity.values, grid.n_nodes_per_face.values, grid.n_edge
    )

    grid._ds["edge_face_connectivity"] = xr.DataArray(
        data=edge_faces,
        dims=ugrid.EDGE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_FACE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True)
def _build_edge_face_connectivity(face_edges, n_nodes_per_face, n_edge):
    """Helper for (``edge_faces``) construction."""
    edge_faces = np.full((n_edge, 2), INT_FILL_VALUE, dtype=INT_DTYPE)

    for face_idx, (cur_face_edges, n_edges) in enumerate(
        zip(face_edges, n_nodes_per_face)
    ):
        # obtain all the edges that make up a face (excluding fill values)
        edges = cur_face_edges[:n_edges]
        for edge_idx in edges:
            if edge_faces[edge_idx, 0] == INT_FILL_VALUE:
                edge_faces[edge_idx, 0] = face_idx
            else:
                edge_faces[edge_idx, 1] = face_idx

    return edge_faces


def _populate_face_edge_connectivity(grid):
    """Constructs the UGRID connectivity variable (``face_edge_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.face_edge_connectivity``)."""

    # TODO: Check if "edge_edge_connectivity" is already present

    if "edge_node_connectivity" not in grid._ds:
        _populate_edge_node_connectivity(grid)


def _populate_node_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``node_face_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.node_face_connectivity``)."""

    node_faces, n_max_faces_per_node = _build_node_face_connectivity(
        grid.face_node_connectivity.values, grid.n_node
    )

    grid._ds["node_face_connectivity"] = xr.DataArray(
        node_faces,
        dims=ugrid.NODE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_FACE_CONNECTIVITY_ATTRS,
    )


def _build_node_face_connectivity(face_nodes, n_node):
    """Builds the `Grid.node_faces_connectivity`: integer DataArray of size
    (n_node, n_max_faces_per_node) (optional) A DataArray of indices indicating
    faces that are neighboring each node.

    This function converts the face-node connectivity data into a sparse matrix, and then constructs the node-face
    connectivity by iterating over each node in the mesh and retrieving the set of neighboring faces.

    Raises
    ------
    RuntimeError
        If the Mesh object does not contain a 'face_node_connectivity' variable.
    """

    node_face_conn = {node_i: [] for node_i in range(n_node)}
    for face_i, face_nodes in enumerate(face_nodes):
        for node_i in face_nodes:
            if node_i != INT_FILL_VALUE:
                node_face_conn[node_i].append(face_i)

    n_max_node_faces = -1
    for face_indicies in node_face_conn.values():
        if len(face_indicies) > n_max_node_faces:
            n_max_node_faces = len(face_indicies)

    node_face_connectivity = np.full(
        (n_node, n_max_node_faces), INT_FILL_VALUE, dtype=INT_DTYPE
    )

    for node_idx, face_indices in enumerate(node_face_conn.values()):
        n_faces = len(face_indices)
        node_face_connectivity[node_idx, 0:n_faces] = face_indices

    return node_face_connectivity, n_max_node_faces


def _face_nodes_to_sparse_matrix(dense_matrix: np.ndarray) -> tuple:
    """Converts a given dense matrix connectivity to a sparse matrix format
    where the locations of non fill-value entries are stored using COO
    (coordinate list) standard. It is represented by three arrays: row indices,
    column indices, and non-filled element flags.

    Parameters
    ----------
    dense_matrix : np.ndarray
        The dense matrix to be converted.
    Returns
    -------
    tuple
        A tuple containing three arrays:
        - face_indices : np.ndarray
            Array containing the face indices for each non fill-value element.
        - node_indices : np.ndarray
            Array containing the node indices for each non fill-value element.
        - non_filled_elements_flag : np.ndarray
            Array containing flags indicating if a non fill-value element is present in the corresponding row and column
            index.
    Example
    -------
    >>> face_nodes_conn = np.array(
    ...     [[3, 4, 5, INT_FILL_VALUE], [3, 0, 2, 5], [3, 4, 1, 0], [0, 1, 2, -999]]
    ... )
    >>> face_indices, nodes_indices, non_filled_flag = _face_nodes_to_sparse_matrix(
    ...     face_nodes_conn
    ... )
    >>> face_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    >>> nodes_indices = np.array([3, 4, 5, 3, 0, 2, 5, 3, 4, 1, 0, 0, 1, 2])
    >>> non_filled_flag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    n_rows, n_cols = dense_matrix.shape
    flattened_matrix = dense_matrix.ravel()
    valid_node_mask = flattened_matrix != INT_FILL_VALUE
    face_indices = np.repeat(np.arange(n_rows), n_cols)[valid_node_mask]
    node_indices = flattened_matrix[valid_node_mask]
    non_filled_element_flags = np.ones(len(node_indices))
    return face_indices, node_indices, non_filled_element_flags


def get_face_node_partitions(n_nodes_per_face):
    """Returns the indices of how to partition `face_node_connectivity` by
    element size."""

    # sort number of nodes per face in ascending order
    n_nodes_per_face_sorted_ind = np.argsort(n_nodes_per_face)

    # unique element sizes and their respective counts
    element_sizes, size_counts = np.unique(n_nodes_per_face, return_counts=True)
    element_sizes_sorted_ind = np.argsort(element_sizes)

    # sort elements by their size
    element_sizes = element_sizes[element_sizes_sorted_ind]
    size_counts = size_counts[element_sizes_sorted_ind]

    # find the index at the point where the geometry changes from one shape to another
    change_ind = np.cumsum(size_counts)
    change_ind = np.concatenate((np.array([0]), change_ind))

    return change_ind, n_nodes_per_face_sorted_ind, element_sizes, size_counts


def _populate_face_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``face_face_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.face_face_connectivity``)."""
    # ``.data`` rather than ``.values``: the latter computes a dask-backed
    # variable in full, which is the one allocation this routine is trying to
    # avoid. ``_build_face_face_connectivity`` consumes either kind.
    face_face = _build_face_face_connectivity(
        grid.edge_face_connectivity.data, grid.n_face, grid.n_max_face_nodes
    )

    grid._ds["face_face_connectivity"] = xr.DataArray(
        data=face_face,
        dims=ugrid.FACE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_FACE_CONNECTIVITY_ATTRS,
    )


def _build_face_face_connectivity(edge_face_connectivity, n_face, n_max_face_nodes):
    """Constructs the ``face_face_connectivity`` variable, which represents the
    indices of the faces that neighbour each face.

    Mirrors the backing of ``edge_face_connectivity``: a dask input yields a dask
    output, so a grid that was chunked stays chunked through the connectivity
    chain rather than silently reverting to NumPy, and a NumPy input yields
    NumPy exactly as before.

    Each edge is scattered into the output independently, so the input is
    consumed a block at a time. The output is indexed by face rather than by
    edge, and the scatter has no locality to exploit, so an output block can
    only be produced by passing over the whole input -- output blocks are
    therefore independent of one another, and run concurrently.
    """
    import dask.array as da

    if not isinstance(edge_face_connectivity, da.Array):
        return _build_face_face_block(
            edge_face_connectivity, n_max_face_nodes, 0, n_face
        )

    import dask

    # Sized by dask's own ``array.chunk-size`` budget, and left whole along the
    # neighbour axis, which is only ``n_max_face_nodes`` wide.
    face_chunks, _ = da.core.normalize_chunks(
        ("auto", -1), shape=(n_face, n_max_face_nodes), dtype=INT_DTYPE
    )

    # One graph node per input block. Calling ``.compute()`` per block from
    # inside a task instead re-enters the scheduler each time, which measured as
    # ~97% of the runtime and serialised everything behind those nested calls.
    edge_face_blocks = list(edge_face_connectivity.to_delayed().ravel())

    blocks = []
    face_start = 0
    for face_chunk in face_chunks:
        face_stop = face_start + face_chunk
        # A linear fold: each accumulator has exactly one consumer, so
        # accumulating in place is safe, and only one input block per output
        # block is live at a time.
        accumulator = dask.delayed(_new_face_face_accumulator, pure=False)(
            face_chunk, n_max_face_nodes
        )
        for edge_face_block in edge_face_blocks:
            accumulator = dask.delayed(_accumulate_face_face_block, pure=False)(
                accumulator, edge_face_block, face_start, face_stop
            )
        blocks.append(
            da.from_delayed(
                dask.delayed(_face_face_result, pure=False)(accumulator),
                shape=(face_chunk, n_max_face_nodes),
                dtype=INT_DTYPE,
            )
        )
        face_start = face_stop

    if len(blocks) == 1:
        # ``concatenate`` would hold both its inputs and its result, costing a
        # second copy of the output for no reason when there is one block.
        return blocks[0]

    return da.concatenate(blocks, axis=0)


def _new_face_face_accumulator(n_rows, n_max_face_nodes):
    """Allocates the output rows for one face block, plus their fill counters."""
    return (
        np.full((n_rows, n_max_face_nodes), INT_FILL_VALUE, INT_DTYPE),
        np.zeros(n_rows, dtype=INT_DTYPE),
    )


def _accumulate_face_face_block(accumulator, edge_face_block, face_start, face_stop):
    """Folds one block of ``edge_face_connectivity`` into an accumulator."""
    face_face_connectivity, face_index_position = accumulator
    _accumulate_face_face_connectivity(
        # Free when the block already is contiguous, and keeps the kernel to a
        # single compiled signature when it is not.
        np.ascontiguousarray(edge_face_block),
        face_face_connectivity,
        face_index_position,
        face_start,
        face_stop,
    )
    return accumulator


def _face_face_result(accumulator):
    """Drops the fill counters, leaving the output rows."""
    return accumulator[0]


def _build_face_face_block(
    edge_face_connectivity, n_max_face_nodes, face_start, face_stop
):
    """Builds the rows of ``face_face_connectivity`` for faces
    ``[face_start, face_stop)`` from a single concrete array."""
    accumulator = _new_face_face_accumulator(
        face_stop - face_start, n_max_face_nodes
    )
    _accumulate_face_face_block(
        accumulator, edge_face_connectivity, face_start, face_stop
    )
    return _face_face_result(accumulator)


# ``nogil`` so that output blocks genuinely run concurrently: each one owns its
# accumulator arrays and only reads the shared input, so there is nothing to
# race on. Holding the GIL here pinned the threaded scheduler to ~1.0x no matter
# how many workers it was given.
@njit(cache=True, nogil=True)
def _accumulate_face_face_connectivity(
    edge_face_block, face_face_connectivity, face_index_position, face_start, face_stop
):
    """Scatters one block of ``edge_face_connectivity`` into the rows of the
    output covering ``[face_start, face_stop)``.

    ``njit`` sits at this level, rather than around the whole build, so that it
    only ever sees a single concrete block -- nopython mode cannot consume a
    dask array, and wrapping the outer loop would force the caller to
    materialize the entire input to satisfy it.
    """
    for i in range(edge_face_block.shape[0]):
        face_a = edge_face_block[i, 0]
        face_b = edge_face_block[i, 1]
        if face_a != INT_FILL_VALUE and face_b != INT_FILL_VALUE:
            if face_start <= face_a < face_stop:
                row = face_a - face_start
                face_face_connectivity[row, face_index_position[row]] = face_b
                face_index_position[row] += 1

            if face_start <= face_b < face_stop:
                row = face_b - face_start
                face_face_connectivity[row, face_index_position[row]] = face_a
                face_index_position[row] += 1


def _populate_node_edge_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""
    node_edge_connectivity = _build_node_edge_connectivity(
        grid.edge_node_connectivity.values, grid.n_node
    )

    grid._ds["node_edge_connectivity"] = xr.DataArray(
        data=node_edge_connectivity,
        dims=ugrid.NODE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_EDGE_CONNECTIVITY_ATTRS,
    )


@njit
def _build_node_edge_connectivity(edge_nodes, n_node):
    """Constructs the Node Edge Connectivity, which stores the indices of the edges that are shared by each node."""
    n_edge, nodes_per_edge = edge_nodes.shape

    # count how many edges touch each node
    counts = np.zeros(n_node, dtype=INT_DTYPE)
    for e in range(n_edge):
        for j in range(nodes_per_edge):
            node = edge_nodes[e, j]
            if node != INT_FILL_VALUE:
                counts[node] += 1

    # find the maximum
    max_edges = 0
    for i in range(n_node):
        if counts[i] > max_edges:
            max_edges = counts[i]

    # allocate output, pad with fill
    node_edge = np.full((n_node, max_edges), INT_FILL_VALUE, dtype=INT_DTYPE)

    ptr = np.zeros(n_node, dtype=INT_DTYPE)

    # fill in
    for e in range(n_edge):
        for j in range(nodes_per_edge):
            node = edge_nodes[e, j]
            if node != INT_FILL_VALUE:
                idx = ptr[node]
                node_edge[node, idx] = e
                ptr[node] += 1

    return node_edge
