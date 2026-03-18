import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calc_slopes(language: str, df_name: str):
    df = pd.read_csv(df_name)
    n_vals = df.n.to_numpy()
    block_t = df.block_time.to_numpy()
    dense_t = df.dense_time.to_numpy()

    block_slopes = np.diff(np.log(block_t)) / np.diff(np.log(n_vals))
    dense_slopes = np.diff(np.log(dense_t)) / np.diff(np.log(n_vals))

    print(f"{language} block solver slopes between n values: {block_slopes}")
    print(f"{language} dense solver slopes between n values: {dense_slopes}")

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    calc_slopes("Python", "python_timings.csv")

    julia_csv = base_dir / "Fast-Eigenvalue-Computation-Julia" / "julia_timings.csv"
    calc_slopes("Julia", julia_csv)

    matlab_csv = base_dir / "fast_Jnlin_eigs" / "matlab_timings.csv"
    calc_slopes("MATLAB", matlab_csv)