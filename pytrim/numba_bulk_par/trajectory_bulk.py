"""Simulate projectile trajectories.

Available functions:
    setup: setup module variables.
    trajectory: simulate one trajectory."""

import numpy as np
from numba import jit

if __package__ and __package__.endswith("numba_bulk_par"):
    from .estop_bulk import eloss
    from .geometry_bulk import is_inside_target
    from .scatter_bulk import scatter
    from .select_recoil_bulk import get_recoil_position
else:
    from estop_bulk import eloss
    from geometry_bulk import is_inside_target
    from scatter_bulk import scatter
    from select_recoil_bulk import get_recoil_position


def setup():
    """Setup module variables.

    Parameters:
        None

    Returns:
        None
    """
    global EMIN

    EMIN = 5.0  # eV


@jit(fastmath=True)
def trajectories(
    pos: np.ndarray, dir: np.ndarray, e: np.ndarray, is_inside: np.ndarray
):
    """Simulate one trajectory.

    Parameters:
        pos_init (ndarray): initial position of the projectile (size 3)
        dir_init (ndarray): initial direction of the projectile (size 3)
        e_init (float): initial energy of the projectile (eV)

    Returns:
        ndarray: final position of the projectile (size 3)
        ndarray: final direction of the projectile (size 3)
        float: final energy of the projectile (eV)
        bool: True if projectile is stopped inside the target,
            False otherwise
    """

    while True:
        valid_ions = np.argwhere((e > EMIN) & (is_inside)).flatten()
        if not valid_ions.any():
            break
        # Current values reference values inside the full list for which the condition is met
        e_current = e[valid_ions]
        pos_current = pos[valid_ions]
        dir_current = dir[valid_ions]
        inside_current = is_inside[valid_ions]

        free_path, p, dirp, _ = get_recoil_position(pos_current, dir_current)
        dee = eloss(e_current, free_path)
        e_current -= dee
        pos_current += free_path * dir_current
        inside_current = is_inside_target(pos_current)
        dir_current, e_current, _, _ = scatter(e_current, dir_current, p, dirp)

        # Write updated values back
        e[valid_ions] = e_current
        pos[valid_ions] = pos_current
        dir[valid_ions] = dir_current
        is_inside[valid_ions] = inside_current
