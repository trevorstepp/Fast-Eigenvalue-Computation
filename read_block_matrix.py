from pathlib import Path
import scipy.io as sio
import numpy as np
import numpy.typing as npt

def load_block_matrix(file: str) -> tuple[int, int, int, npt.NDArray]:
    """
    Load a block matrix from a .mat file.

    Parameters
    ----
    file : str
        Name of the .mat file containing the matrix.

    Returns
    ----
    Kn : int
        Total number of rows/columns of the full matrix.
    K : int
        Number of blocks per row/column.
    n : int
        Size of each block (n x n).
    matrix : ndarray
        The full block matrix loaded from the .mat file.
    """
    # get to location of .mat file with block matrix
    repo_root = Path(__file__).parent.parent
    mat_file = repo_root / "fast_Jnlin_eigs" / file

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