import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("python_timings.csv")
n_vals = df.n 
block = df.block_time
dense = df.dense_time

# compute slope using log-log regression
slope_block = np.cov(np.log(n_vals), np.log(block))[0, 1] / np.var(np.log(n_vals))
slope_dense = np.cov(np.log(n_vals), np.log(dense))[0, 1] / np.var(np.log(n_vals))

print(f"Block slope = {slope_block}")
print(f"Dense slope = {slope_dense}")

# scatter plot with log-log scales
plt.figure(figsize=(8,6))
plt.scatter(n_vals, block, label="Block", marker='o')
plt.scatter(n_vals, dense, label="Dense", marker='D')

# reference lines
plt.plot(n_vals, n_vals * block[0]/n_vals[0], linestyle='--', label="O(n)")
plt.plot(n_vals, n_vals**3 * dense[0]/n_vals[0]**3, linestyle='--', label="O(n^3)")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("n")
plt.ylabel("Time (s)")
plt.title("Scaling of Block vs Dense Eigensolvers")
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend()

# save figure
plt.savefig("scaling_plot.png", dpi=300)
plt.show()