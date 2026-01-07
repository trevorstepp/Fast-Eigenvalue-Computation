import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from algorithm import eig_KxK_diagblocks

class VerificationResult(NamedTuple):
    eigenvalues_match: bool
    max_residual: float
    mean_residual: float

def verify_results(K: int, n: int, matrix: npt.NDArray, atol: float=1e-8) -> VerificationResult:
    """
    Verify correctness of the block-diagonal eigendecomposition algorithm.

    Compare eigenvalues produced by the K x K block-diagonal algorithm against Numpy's eig
    function and evaluate the accuracy of the computed eigenvectors using residual norms.
    
    Parameters
    ----
    K : int
        Number of block matrices per row/col (K^2 total blocks).
    n : int
        Number of rows/cols in each block matrix (each block matrix is n x n).
    matrix : ndarray
        Full block-structured matrix (Kn x Kn).
    atol : float
        Absolute tolerance used to determine equality of sorted eigenvalues between
        the block algorithm and NumPy's eig results.
    
    Returns
    ---- 
    A named tuple with the following attributes:

    eigenvalues_match : bool 
        Indicates whether the sorted eigenvalues match NumPy's eig results.
    max_residual : float 
        Maximum eigenpair residual ``||matrix @ v - lambda * v||``.
    mean_residual : float 
        Mean eigenpair residual over all eigenvectors.
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

    mean_res = res_sum / len(alg_eigs)

    return VerificationResult(
        eigenvalues_match=eig_check,
        max_residual=max_res,
        mean_residual=mean_res
    )