"""PyTRIM aims to be a Python implementation of TRIM.

TRIM (Transport of Ions in Matter) is a widely used software package
for simulating the interaction of ions with matter, particularly for
ion implantation in semiconductors. It comes as part of SRIM, see
www.srim.org.

PyTRIM seeks to replicate the core functionalities of TRIM using
Python, making it more accessible and easier to integrate with other
Python-based tools and workflows.

Currently, the input parameters are hardcoded in this script, but future
versions may include a more user-friendly interface for specifying
simulation parameters. Also, recoils are not yet followed, and only the
mean and the straggling of the penetration depth of the primary ions are
recorded.
"""

import time

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"
import numpy as np
from numba import jit, prange

if __package__ and __package__.endswith("numba_bulk_par"):
    from . import estop_bulk as estop
    from . import geometry_bulk as geometry
    from . import scatter_bulk as scatter
    from . import select_recoil_bulk as select_recoil
    from . import trajectory_bulk as trajectory
else:
    import estop_bulk as estop
    import geometry_bulk as geometry
    import scatter_bulk as scatter
    import select_recoil_bulk as select_recoil
    import trajectory_bulk as trajectory

nion = 1000  # number of projectiles to simulate

zmin = 0.0  # minimum z coordinate of the target (A)
zmax = 4000.0  # maximum z coordinate of the target (A)
z1 = 5  # atomic number of projectile
m1 = 11.009  # mass of projectile (amu)
z2 = 14  # atomic number of target
m2 = 28.086  # mass of target atom (amu)
density = 0.04994  # target density (atoms/A^3)
corr_lindhard = 1.5  # Correction factor to Lindhard stopping power

select_recoil.setup(density)
scatter.setup(z1, m1, z2, m2)
estop.setup(corr_lindhard, z1, m1, z2, density)
geometry.setup(zmin, zmax)
trajectory.setup()


@jit(fastmath=True, parallel=True, nogil=True, cache=True)
def simulate(nion: int, nthreads: int = 16):
    # Initial conditions of the projectile
    threads = 1
    col_cnt = nion
    for i in range(nthreads, 0, -1):
        if nion % i == 0:
            threads = i
            col_cnt = int(nion / i)
            break
    e = np.full((threads, col_cnt), 50000.0)  # energy (eV)
    pos = np.zeros((threads, col_cnt, 3))  # position (A)
    dir = np.zeros((threads, col_cnt, 3))
    dir[:, :, 2] = 1.0  # direction (unit vector)
    is_inside = np.full((threads, col_cnt), True)

    for i in prange(threads):
        trajectory.trajectories(pos[i], dir[i], e[i], is_inside[i])

    pos = pos.reshape((nion, 3))
    dir = dir.reshape((nion, 3))
    e = e.flatten()
    is_inside = is_inside.flatten()
    count_inside = np.count_nonzero(is_inside)
    mean_z = pos[is_inside, 2]
    std_z = np.std(mean_z)
    mean_z = np.mean(mean_z)

    return pos, dir, e, is_inside, mean_z, std_z, count_inside


if __name__ == "__main__":
    # Pre-compile
    simulate(8)
    start_time = time.time()
    pos, dir, e, is_inside, mean_z, std_z, count_inside = simulate(nion)
    end_time = time.time()
    print(f"Number of ions stopped inside the target: {count_inside} / {nion}")
    print(f"Mean penetration depth of ions stopped inside the target: {mean_z:.2f} A")
    print(f"Standard deviation of penetration depth: {std_z:.2f} A")
    print(f"Simulation time: {end_time - start_time:.2f} seconds")
    print(f"Seconds per ion: {(end_time - start_time) / nion}")

    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.boxplot(pos[is_inside, 2])
    plt.title("Numba bulk")
    plt.show()
