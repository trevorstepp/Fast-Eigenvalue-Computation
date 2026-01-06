import numpy as np
import numpy.typing as npt

def eig_KxK_diagblocks(K: int, n: int, matrix: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the eigenvalues and eigenvectors of a block-diagonal matrix.
    
    Parameters
    ----
    K : int
        Number of block matrices per row/col (K^2 total blocks).
    N : int
        Number of rows/cols in each block matrix (each block matrix is n x n).
    matrix : ndarray
        The full matrix containing all block matrices (Kn x Kn).

    Returns
    ----
    eigs : ndarray
        Eigenvalues of the full matrix.
    vecs : ndarray
        Corresponding eigenvectors of the full matrix.
    """
    # hold eigenvalues
    eigs = np.zeros(K * n, dtype=complex)
    # hold eigenvectors
    vecs = np.zeros(shape=(K * n, K * n), dtype=complex)
    # column counter
    col = 0

    # for each l = 1, ..., n
    for l in range(n):
        M_l = np.zeros(shape=(K, K), dtype=complex)
        for i in range(K):
            for j in range(K):
                M_l[i, j] = matrix[i * n + l, j * n + l]
        # get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(M_l)

        eigs[col:col + K] = eigvals
        V = np.zeros(shape=(K * n, K), dtype=complex)
        for i in range(K):
            for j in range(K):
                V[i * n + l, j] = eigvecs[i, j]
        
        vecs[:, col:col + K] = V
        col += K
    return eigs, vecs

def compare_results(K: int, n: int, matrix: npt.NDArray) -> None:
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

    # must sort both eigenvalue results (could be different orderings)
    alg_eigs_sorted = np.sort_complex(alg_eigs)
    reg_eigs_sorted = np.sort_complex(reg_eigs)

    print(f"Eigenvalues match: {np.allclose(alg_eigs_sorted, reg_eigs_sorted, atol=1e-8)}")

if __name__ == "__main__":
    A = np.array([
        [1, 0, 3, 0],
        [0, 2, 0, 4],
        [5, 0, 7, 0],
        [0, 6, 0, 8]
    ])
    eigs, vecs = eig_KxK_diagblocks(K=2, n=2, matrix=A)
    print("****KxK function results****\n")
    print(f"eigenvalues = {eigs}")
    print(f"\neigenvectors = {vecs}\n\n")

    print("****Regular NumPy****\n")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"eigenvalues = {eigenvalues}")
    print(f"\neigenvectors = {eigenvectors}")

    compare_results(K=2, n=2, matrix=A)