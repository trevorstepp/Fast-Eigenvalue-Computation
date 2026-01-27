import pandas as pd
import matplotlib.pyplot as plt

def plot_runtime_comparison() -> None:
    """
    Docstring for plot_runtime_comparison.
    """
    df = pd.read_csv("timings.csv")
    K = df['K'][0]

    plt.figure(figsize=(8,6))
    plt.semilogy(df.n, df.block_time, 'o-', label='Our approach (all)')
    plt.semilogy(df.n, df.dense_time, 's-', label='NumPy eig (full)')
    plt.xlabel('discretization size (n)')
    plt.ylabel('run time (s)')
    plt.title(f'All eigenpairs runtime (K={K})')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"K={K}.png")
    plt.show()

if __name__ == '__main__':
    plot_runtime_comparison()