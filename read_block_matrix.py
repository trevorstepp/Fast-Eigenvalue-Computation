from pathlib import Path
#from scipy.sp
import scipy.io as sio
import numpy as np
import numpy.typing as npt
#import time

def load_block_matrix() -> tuple[int, int, int, npt.NDArray]:
    """Loads the block matrix from the .mat file.

    Parameters:
        None.
    Returns:
        tuple[int, int, int, npt.NDArray]: The sizes of the entire matrix, 
            the matrix containing the block matrices, and each block matrix, and the matrix itself
    """
    # get to location of .mat file with block matrix
    repo_root = Path(__file__).parent.parent
    mat_file = repo_root / "fast_Jnlin_eigs" / "3case.mat"

    # load file
    mat_contents = sio.loadmat(mat_file)
    matrix = mat_contents["Blocks"]
    Kn = matrix.shape[0]
    K = 3
    n = Kn // K
    return Kn, K, n, matrix

"""
def scipy_version(matrix):
    # check time it takes to build matrix
    start = time.time()
    dense_

    end = time.time()
    print(f"NumPy matrix shape: {}")
"""