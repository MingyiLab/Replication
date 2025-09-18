
import numpy as np
import heapq
from math import sqrt
from typing import Tuple, Iterable, Union

def graydist_quasi_euclidean(
    image: np.ndarray,
    seeds: Union[np.ndarray, Iterable[Tuple[int,int]]],
    method: str = "quasi-euclidean",
    flatten: bool = True,
) -> np.ndarray:
    """
    Compute gray-weighted distance transform and optionally return as a 1-D row.

    Parameters
    ----------
    image : 2D array-like
        Grayscale image (numeric). In this case, it's the grid map with longitude and latitude
    seeds : boolean mask (same shape) OR iterable of (row, col) seed tuples
    method : {'quasi-euclidean', 'cityblock', 'chessboard'}
    flatten : bool
        If True, return a 1-D numpy array of length rows*cols in row-major order
        where index 0 = (0,0), index N = (1,0) for N=cols, etc.
        If False, return the 2-D distance map.

    Returns
    -------
    np.ndarray
        1-D array (if flatten=True) or 2-D distance map (if flatten=False).
    """
    I = np.asarray(image, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("Only 2D images supported.")
    rows, cols = I.shape

    # Accept either boolean mask or list of coords
    if isinstance(seeds, np.ndarray) and seeds.dtype == bool:
        if seeds.shape != I.shape:
            raise ValueError("Seed mask must have the same shape as image.")
        seed_coords = np.argwhere(seeds)
    else:
        seed_coords = np.array(list(seeds), dtype=int)
        if seed_coords.size == 0:
            seed_coords = seed_coords.reshape((0,2))

    # Neighbor offsets + weights
    if method == "quasi-euclidean":
        neighs = [(-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                  (-1,-1, sqrt(2)), (-1, 1, sqrt(2)), (1, -1, sqrt(2)), (1, 1, sqrt(2))]
    elif method == "cityblock":
        neighs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0)]
    elif method == "chessboard":
        neighs = [(-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                  (-1,-1, 1.0), (-1, 1, 1.0), (1, -1, 1.0), (1, 1, 1.0)]
    else:
        raise ValueError("Unknown method. Choose 'quasi-euclidean', 'cityblock' or 'chessboard'.")

    dist = np.full_like(I, np.inf, dtype=np.float64)
    heap = []

    if seed_coords.size == 0:
        raise ValueError("No seeds provided.")
    for sc in seed_coords:
        r, c = int(sc[0]), int(sc[1])
        if 0 <= r < rows and 0 <= c < cols:
            if dist[r, c] > 0.0:
                dist[r, c] = 0.0
                heapq.heappush(heap, (0.0, r, c))

    # Dijkstra propagation
    while heap:
        dcur, r, c = heapq.heappop(heap)
        if dcur > dist[r, c]:
            continue
        Icur = I[r, c]
        for dr, dc, w in neighs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cost = 0.5 * (Icur + I[nr, nc]) * w
            nd = dcur + cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))

    if flatten:
        # row-major flattening: index = row*cols + col
        return dist.ravel(order='C')  # shape (rows*cols,)
    else:
        return dist
