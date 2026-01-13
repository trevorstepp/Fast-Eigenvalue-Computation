import time
import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from algorithm import eig_KxK_diagblocks

class TimingResult(NamedTuple):
    eigenvalues: npt.NDArray
    eigenvectors: npt.NDArray
    time: float

def time_block_method(K: int, n: int, M: npt.NDArray) -> TimingResult:
    """
    Measure runtime for.
    
    Parameters
    ----

    Returns
    ----
    """
    t0 = time.perf_counter()
    eigs, vecs = eig_KxK_diagblocks(K, n, M)
    t1 = time.perf_counter()
    total_time = t1 - t0
    return TimingResult(
        eigenvalues=eigs,
        eigenvectors=vecs,
        time=total_time
    )

def time_numpy(M: npt.NDArray) -> TimingResult:
    """
    Docstring for time_numpy.
    
    Parameters
    ----

    Returns
    ----
    """
    t0 = time.perf_counter()
    eigs, vecs = np.linalg.eig(M)
    t1 = time.perf_counter()
    total_time = t1 - t0
    return TimingResult(
        eigenvalues=eigs,
        eigenvectors=vecs,
        time=total_time
    )