from __future__ import annotations
import numpy as np
from typing import Optional, Literal

# ----------------------------------------------------------------------
#  HNSW-initialised embedder
# ----------------------------------------------------------------------
def embed(
    csvfile: str,
    *,
    outfile: str | None = ...,
    delim: str = ",",
    dim: int = 2,
    batch: int = 20,
    nbsample: int = 10,
    hierarchy: int = 0,
    scale: float = 1.0,
    quality_sampling: Optional[float] = ...,
    distance: Literal[
        "DistL1", "DistL2", "DistCosine", "DistJeffreys", "DistJensenShannon"
    ] = "DistL2",
    nbconn: int = 64,
    ef: int = 512,
    knbn: int = 10,
    scale_modification: float = 1.0,
) -> np.ndarray: ...
"""
    Embed vectors in *csvfile* using the annembed gradient-descent routine
    initialised by an HNSW k-NN graph.

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, ``dim``); dtype ``float64``.
    """

# ----------------------------------------------------------------------
#  Diffusion-Maps embedder
# ----------------------------------------------------------------------
def dmap_embed(
    csvfile: str,
    *,
    outfile: str | None = ...,
    delim: str = ",",
    dim: int = 2,
    alfa: float = 1.0,
    beta: float = 0.0,
    time: float = 5.0,
    hierarchy: int = 0,
    quality_sampling: Optional[float] = ...,
    distance: Literal[
        "DistL1", "DistL2", "DistCosine", "DistJeffreys", "DistJensenShannon"
    ] = "DistL2",
    nbconn: int = 64,
    ef: int = 512,
    knbn: int = 10,
    scale_modification: float = 1.0,
) -> np.ndarray: ...
"""
    Diffusion-Maps embedding with optional α, β, diffusion time, and
    hierarchical projection layer.

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, ``dim``); dtype ``float64``.
    """

