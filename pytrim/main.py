import time

import bulk.pytrim_bulk as pytrim_bulk
import matplotlib.pyplot as plt
import numba_bulk.pytrim_bulk as pytrim_numba_bulk  # noqa: E402
import numba_local.pytrim as pytrim_numba  # noqa: E402
import numpy as np
from tqdm import tqdm

import pytrim

nion = 1000
iter_count = 5

if __name__ == "__main__":
    times = []
    data = []
    test_funcs = [
        # {
        #     "func": pytrim.simulate,
        #     "label": "orig",
        #     "compilable": False,
        #     "iter_override": 1,
        # },
        {"func": pytrim_numba.simulate, "label": "numba", "compilable": True},
        {"func": pytrim_bulk.simulate, "label": "bulk", "compilable": False},
        {"func": pytrim_numba_bulk.simulate, "label": "numba_bulk", "compilable": True},
    ]
    plt.style.use("dark_background")
    plt.title("Comparison of different simulations")
    for i in tqdm(range(iter_count), desc="Current iteration"):
        times.append([])
        for func_def in tqdm(test_funcs, desc="Progress", leave=False):
            if "iter_override" in func_def.keys():
                it_ovr = func_def["iter_override"]
                if i + 1 > it_ovr:
                    times[i].append(it_ovr - 1)  # TODO Should preserve mean???
                    continue
            if func_def["compilable"]:
                func_def["func"](nion)  # Dry-run to compile first
                # Prevent re-compilation on next iteration
                func_def["compilable"] = False
            start = time.time()
            pos, dir, e, is_inside, *other = func_def["func"](nion)
            times[i].append(time.time() - start)
            if i == iter_count - 1:
                if isinstance(pos, list):  # Assuming everything else is list as well
                    pos = np.array(pos)
                    dir = np.array(dir)
                    e = np.array(e)
                    is_inside = np.array(is_inside)
                data.append(pos[is_inside, 2])

    avg_times = [0.0 for _ in range(len(test_funcs))]
    for j in range(len(test_funcs)):
        avg_times[j] = sum(times[i][j] for i in range(iter_count)) / iter_count
    print(dict(zip([f["label"] for f in test_funcs], avg_times)))
    plt.boxplot(data)
    plt.xticks(
        [i + 1 for i in range(len(test_funcs))], [f["label"] for f in test_funcs]
    )
    plt.show()
