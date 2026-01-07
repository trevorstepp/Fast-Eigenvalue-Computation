import time
import numpy as np
import numpy.typing as npt

from algorithm import eig_KxK_diagblocks

def time_block_method(K: int, n: int, M: npt.NDArray):
    """
    Measure runtime for.
    
    Parameters
    ----

    Returns
    ----
    """
    t0 = time.perf_counter()
    eig_KxK_diagblocks(K, n, M)
    t1 = time.perf_counter()
    return t1 - t0

def time_numpy(M: npt.NDArray):
    """
    Docstring for time_numpy.
    
    Parameters
    ----

    Returns
    ----
    """
    t0 = time.perf_counter()
    np.linalg.eig(M)
    t1 = time.perf_counter()
    return t1 - t0