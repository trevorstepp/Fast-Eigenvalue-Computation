import pandas as pd

from benchmark import time_block_method, time_numpy
from verify import verify_results
from build import build_block_matrix

N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000]
K_VAL = 3

def measure_runtime_and_verify() -> tuple[list[float], list[float]]:
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
        M = build_block_matrix(K=K_VAL, n=n, seed=0)

        # measure time for both methods
        block_result = time_block_method(K=K_VAL, n=n, M=M)
        numpy_result = time_numpy(M=M)

        # store times for plotting
        block_time.append(block_result.time)
        eig_time.append(numpy_result.time)

        # correctness check
        correct_check = verify_results(matrix=M, alg_eigs=block_result.eigenvalues, alg_vecs=block_result.eigenvectors,
                                       reg_eigs=numpy_result.eigenvalues, reg_vecs=numpy_result.eigenvectors)
        
        print(f"Eigenvalues match (top {correct_check.num_eigenvalues_compared}): {correct_check.eigenvalues_match}")
        print(f"Maximum residual: {correct_check.max_residual:.2e}")
        print(f"Mean residual: {correct_check.mean_residual:.2e}")
    
    return block_time, eig_time

def main():
    """
    Docstring for main.
    """
    block_time, eig_time = measure_runtime_and_verify()
    # save times in CSV file for future reference
    df = pd.DataFrame({
        "K": K_VAL,
        "n": N_VALUES,
        "block_time": block_time,
        "dense_time": eig_time
    })
    df.to_csv("timings.csv", index=False)

if __name__ == '__main__':
    main()