import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams.update(
#     {
#         "font.size": 6,  # Default text size
#         "axes.titlesize": 14,  # Title size
#         "axes.labelsize": 10,  # X/Y label size
#         "xtick.labelsize": 8,  # Tick label sizes
#         "ytick.labelsize": 8,
#         "legend.fontsize": 8,
#     }
# )

funcs = ["orig", "numba", "bulk", "numba_bulk", "numba_bulk_par"]
avg_times = np.fromfile("perf/avg_times_new.np").reshape(len(funcs), -1)
nions = np.array([50, 100, 500, 1000, 10000, 100000, 500000, int(1e6)])
assert nions.size == avg_times.shape[1]

plt.style.use("dark_background")
plt.title("Overview of simulation times")
for fname, times in zip(funcs, avg_times):
    valid_times = times[times > 1e-4]
    plt.plot(nions[: valid_times.size], valid_times, marker="o", label=fname)
    for nion, time in zip(nions[: valid_times.size], valid_times):
        text = f"{float(time):.2f}"
        plt.annotate(
            text,
            (nion, time),
            textcoords="offset pixels",
            xytext=(-(len(text) * 10), -5),
        )
plt.legend()
plt.tight_layout()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number ions simulated")
plt.ylabel("Time (s)")
# plt.savefig("media/sim_perf_par.png", dpi=300, transparent=True)
plt.show()
