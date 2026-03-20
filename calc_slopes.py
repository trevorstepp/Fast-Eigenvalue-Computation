import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.polynomial.polynomial import polyfit

def calc_slopes(language: str, df_name: str):
    df = pd.read_csv(df_name)
    df_large_n = df[df["n"] >= 500]

    # want log-log slope
    log_n = np.log(df_large_n["n"])
    log_block = np.log(df_large_n["block_time"])
    log_dense = np.log(df_large_n["dense_time"])

    intercept_block, slope_block = polyfit(log_n, log_block, 1)
    intercept_dense, slope_dense = polyfit(log_n, log_dense, 1)

    print(f"{language} block slope: {slope_block}")
    print(f"{language} dense slope: {slope_dense}")

    # compute fitted line
    fit_block = intercept_block + slope_block * log_n
    fit_dense = intercept_dense + slope_dense * log_n

    # plot for sanity check
    plt.scatter(log_n, log_block, label=f"{language} Block")
    plt.scatter(log_n, log_dense, label=f"{language} Dense")
    plt.plot(log_n, fit_block, linestyle="--")
    plt.plot(log_n, fit_dense, linestyle="--")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    calc_slopes("Python", "python_timings.csv")

    julia_csv = base_dir / "Fast-Eigenvalue-Computation-Julia" / "julia_timings.csv"
    calc_slopes("Julia", julia_csv)

    matlab_csv = base_dir / "fast_Jnlin_eigs" / "matlab_timings.csv"
    calc_slopes("MATLAB", matlab_csv)

    #calc_slopes("Python", "test_timings.csv")