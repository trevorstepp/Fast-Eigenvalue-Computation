import numpy as np
import numpy.typing as npt

def eig_KxK_diagblocks(K: int, n: int, matrix: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Docstring for eig_KxK_diagblocks.

    Params:
        K (int): number of block matrices
        n (int): number of rows/cols of each block matrix
        matrix (NDArray): the entire matrix
    Returns:
        tuple[NDArray, NDArray]: .
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

        # store in eigs and vecs
        for i in range(K):
            v = np.zeros(K * n, dtype=complex)
            for j in range(K):
                v[j * n + l] = eigvecs[j, i]
            # update vecs
            vecs[:, col] = v
            eigs[col] = eigvals[i]
            col += 1

    return vecs, eigs

if __name__ == "__main__":
    A = np.array([
        [1, 0, 3, 0],
        [0, 2, 0, 4],
        [5, 0, 7, 0],
        [0, 6, 0, 8]
    ])
    vecs, eigs = eig_KxK_diagblocks(K=2, n=2, matrix=A)
    print("****KxK function results****\n")
    print(f"eigenvalues = {eigs}")
    print(f"\neigenvectors = {vecs}\n\n")

    print("****Regular NumPy****\n")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"eigenvalues = {eigenvalues}")
    print(f"\neigenvectors = {eigenvectors}")