import numpy as np
import numpy.typing as npt

from algorithm import eig_KxK_diagblocks

def verify_results(K: int, n: int, matrix: npt.NDArray, atol: float=1e-8) -> None:
    """
    Docstring for compare_results
    
    :param K: Number of block matrices per row/col (K^2 total blocks).
    :type K: int
    :param n: Number of rows/cols in each block matrix (each block matrix is n x n).
    :type n: int
    :param matrix: The matrix containing all block matrices (Kn x Kn).
    :type matrix: npt.NDArray
    """
    # KxK algorithm results
    alg_eigs, alg_vecs = eig_KxK_diagblocks(K, n, matrix)
    # regular method (eig on entire matrix) results
    reg_eigs, reg_vecs = np.linalg.eig(matrix)

    # sort eigenvalue results (could be different orderings) before comparing
    alg_eigs_sorted = np.sort_complex(alg_eigs)
    reg_eigs_sorted = np.sort_complex(reg_eigs)
    eig_check = np.allclose(alg_eigs_sorted, reg_eigs_sorted, atol=atol)

    # eigenvector residual check
    max_res = 0.0
    for i in range(len(alg_eigs)):
        pass

    return eig_check, max_res