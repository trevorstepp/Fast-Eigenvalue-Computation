import time
import numpy as np
import numpy.typing as npt

from algorithm import eig_KxK_diagblocks, compare_results

def time_block_method(K: int, n: int, M: npt.NDArray):
    """
    Docstring for time_block_method.
    
    :param K: Description
    :type K: int
    :param n: Description
    :type n: int
    :param M: Description
    :type M: npt.NDArray
    """
    t0 = time.perf_counter()
    eig_KxK_diagblocks(K, n, M)
    t1 = time.perf_counter()
    return t1 - t0

def time_numpy(K: int, n: int, M: npt.NDArray):
    """
    Docstring for time_numpy
    
    :param K: Description
    :type K: int
    :param n: Description
    :type n: int
    :param M: Description
    :type M: npt.NDArray
    """
    t0 = time.perf_counter()
    np.linalg.eig(M)
    t1 = time.perf_counter()
    return t1 - t0