import numpy as np

from read_block_matrix import load_block_matrix
from algorithm import compare_results, eig_KxK_diagblocks

if __name__ == '__main__':
    Kn, K, n, blocks = load_block_matrix()
    vecs, eigs = eig_KxK_diagblocks(K=K, n=n, matrix=blocks)
    print("****KxK function results****\n")
    print(f"eigenvalues = {eigs}")
    print(f"\neigenvectors = {vecs}\n\n")

    print("****Regular NumPy****\n")
    eigenvalues, eigenvectors = np.linalg.eig(blocks)
    print(f"eigenvalues = {eigenvalues}")
    print(f"\neigenvectors = {eigenvectors}")