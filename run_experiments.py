import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from benchmark import time_block_method, time_numpy
from verify import verify_results
from build import build_block_matrix

N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000]

def measure_runtime_and_verify() -> None:
    """
    Docstring for measure_runtime.
    
    Parameters
    ----

    Returns
    ----
    """
    # lists to hold runtimes for block method and NumPy's eig method
    block_time = []
    eig_time = []

    for n in N_VALUES:
        print(f"\nRunning n = {n}")

        # create the block-diagonal matrix
        M = build_block_matrix(K=3, n=n, seed=0)

        # measure time for both methods
        block_result = time_block_method(K=3, n=n, M=M)
        numpy_result = time_numpy(M=M)

        # store times for plotting
        block_time.append(block_result.time)
        eig_time.append(numpy_result.time)

        # correctness check
        correct_check = verify_results(matrix=M, alg_eigs=block_result.eigenvalues, alg_vecs=block_result.eigenvectors,
                                       reg_eigs=numpy_result.eigenvalues, reg_vecs=numpy_result.eigenvectors)
        print(f"Eigenvalues match: {correct_check.eigenvalues_match}")
        print(f"Maximum residual: {correct_check.max_residual:.2e}")
        print(f"Mean residual: {correct_check.mean_residual:.2e}")
    
    return block_time, eig_time

def plot_runtime_comparison(t_block: list[float], t_eig: list[float]) -> None:
    """
    Docstring for plot_runtime_comparison.
    """
    plt.figure(figsize=(8,6))
    plt.semilogy(N_VALUES, t_block, 'o-', label='Our approach (all)')
    plt.semilogy(N_VALUES, t_eig, 's-', label='NumPy eig (full)')
    plt.xlabel('discretization size (n)')
    plt.ylabel('run time (s)')
    plt.title('All eigenpairs runtime (K=3)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Docstring for main.
    """
    block_time, eig_time = measure_runtime_and_verify()
    plot_runtime_comparison(block_time, eig_time)

if __name__ == '__main__':
    main()