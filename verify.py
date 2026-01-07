import numpy as np
import numpy.typing as npt

from algorithm import eig_KxK_diagblocks

def verify_results(K: int, n: int, matrix: npt.NDArray, atol: float=1e-8) -> None:
    """
    Docstring for compare_results.
    
    Parameters
    ----
    K : int
        Number of block matrices per row/col (K^2 total blocks).
    n : int
        Number of rows/cols in each block matrix (each block matrix is n x n).
    matrix : ndarray
        The full matrix containing all block matrices (Kn x Kn).
    atol : float
        The absolute tolerance for
    
    Returns
    ----
    eig_check : bool
        T
    max_res : float
        The maximum residual (error) among all eigenvectors.
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
    res_sum = 0.0
    for i in range(len(alg_eigs)):
        eigval = alg_eigs[i]
        eigvec = alg_vecs[:, i]
        res = np.linalg.norm(matrix @ eigvec - eigval * eigvec)
        res_sum += res
        max_res = max(max_res, res)

    return {
        "eigenvalues_match": eig_check,
        "max_residual": max_res,
        "mean_residual": res_sum / len(alg_eigs)
    }