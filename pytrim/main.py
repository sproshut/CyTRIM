import time

import bulk.pytrim_bulk as pytrim_bulk
import matplotlib.pyplot as plt
import numba_bulk.pytrim_bulk as pytrim_numba_bulk  # noqa: E402
import numba_local.pytrim as pytrim_numba  # noqa: E402
import numpy as np
from tqdm import tqdm

import pytrim

test_nions = [100, 1000, 10000, 100000]
iter_count = 3
test_funcs = [
    # {
    #     "func": pytrim.simulate,
    #     "label": "orig",
    #     "compilable": False,
    #     # "iter_override": 1,
    #     # "min_nion_test": True,
    # },
    {"func": pytrim_numba.simulate, "label": "numba", "compilable": True},
    {"func": pytrim_bulk.simulate, "label": "bulk", "compilable": False},
    {"func": pytrim_numba_bulk.simulate, "label": "numba_bulk", "compilable": True},
]

if __name__ == "__main__":
    avg_times = np.zeros((len(test_funcs), len(test_nions)))
    z_pos = []
    for i in tqdm(range(iter_count), desc="Iterations"):
        for n_i, nion in enumerate(nion_pbar := tqdm(test_nions, leave=False)):
            nion_pbar.set_description_str(f"{nion} ions")
            for f_i, func_def in enumerate(func_pbar := tqdm(test_funcs, leave=False)):
                func_pbar.set_description_str(f"in {func_def['label']}")
                if func_def.get("min_nion_test") and n_i != 0:
                    continue
                if "iter_override" in func_def.keys():
                    it_ovr = func_def["iter_override"]
                    if i + 1 > it_ovr:
                        continue
                if func_def["compilable"]:
                    func_def["func"](nion)  # Dry-run to compile first
                    # Prevent re-compilation on next iteration / ion count
                    func_def["compilable"] = False

                start = time.time()
                pos, dir, e, is_inside, *other = func_def["func"](nion)
                avg_times[f_i, n_i] += time.time() - start
                if n_i == 0 and i == iter_count - 1:
                    if isinstance(pos, list):
                        # Assuming everything else is a list as well
                        pos = np.array(pos)
                        is_inside = np.array(is_inside)
                    z_pos.append(pos[is_inside, 2])
            (avg_times / (i + 1)).tofile("avg_times_3.np")

    avg_times /= iter_count
    print(avg_times)
    plt.style.use("dark_background")
    plt.title("Z depth across different simulations")
    plt.boxplot(z_pos)
    plt.xticks(
        [i + 1 for i in range(len(test_funcs))], [f["label"] for f in test_funcs]
    )
    plt.show()
