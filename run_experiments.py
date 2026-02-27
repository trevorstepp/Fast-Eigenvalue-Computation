import pandas as pd
import numpy as np

from benchmark import time_block_method, time_numpy
from verify import verify_results
from build import build_block_matrix

N_VALUES = [100, 250, 500, 750, 1000, 1500, 2000]
K_VAL = 3
NUM_RUNS = 5

def measure_runtime_and_verify() -> tuple[list[float], list[float], list[float]]:
    """
    Docstring for measure_runtime.
    
    Parameters
    ----

    Returns
    ----
    """
    # lists to hold runtimes for block method and NumPy's eig method and max residuals for block method
    block_time = []
    eig_time = []
    block_max_residuals = []

    for n in N_VALUES:
        # create the block-diagonal matrix
        M = build_block_matrix(K=K_VAL, n=n, seed=0)

        # list to hold run times, use to take average run time
        run_times_block = []
        run_times_eig = []

        # track largest residual across the NUM_RUNS iterations
        max_res_block = -np.inf

        for i in range(NUM_RUNS):
            print(f"\nRunning n = {n}, iteration {i}")

            # measure time for both methods
            block_result = time_block_method(K=K_VAL, n=n, M=M)
            numpy_result = time_numpy(M=M)

            # store times
            run_times_block.append(block_result.time)
            run_times_eig.append(numpy_result.time)

            # correctness check
            correct_check = verify_results(matrix=M, alg_eigs=block_result.eigenvalues, alg_vecs=block_result.eigenvectors,
                                           reg_eigs=numpy_result.eigenvalues, reg_vecs=numpy_result.eigenvectors)
            # check for max residual
            max_res_block = max(max_res_block, correct_check.max_residual)

            print(f"Eigenvalues match (top {correct_check.num_eigenvalues_compared}): {correct_check.eigenvalues_match}")
            print(f"Maximum residual: {correct_check.max_residual:.2e}")
            print(f"Mean residual: {correct_check.mean_residual:.2e}")
        
        # store average times for plotting
        avg_block = np.mean(run_times_block)
        avg_eig = np.mean(run_times_eig)

        block_time.append(avg_block)
        eig_time.append(avg_eig)
        # store max residual
        block_max_residuals.append(max_res_block)
    
    return block_time, eig_time, block_max_residuals

def main():
    """
    Docstring for main.
    """
    block_time, eig_time, block_residuals = measure_runtime_and_verify()
    # save times in CSV file for future reference
    df = pd.DataFrame({
        "K": K_VAL,
        "n": N_VALUES,
        "block_time": block_time,
        "dense_time": eig_time,
        "max_residual": block_residuals
    })
    df.to_csv("python_timings.csv", index=False)

if __name__ == '__main__':
    main()