"""Simulate projectile trajectories.

Available functions:
    setup: setup module variables.
    trajectory: simulate one trajectory."""

from select_recoil_bulk import get_recoil_position
from scatter_bulk import scatter
from estop_bulk import eloss
from geometry import is_inside_target
import numpy as np


def setup():
    """Setup module variables.

    Parameters:
        None

    Returns:
        None
    """
    global EMIN

    EMIN = 5.0  # eV


def trajectories(pos_init: np.ndarray, dir_init: np.ndarray, e_init: np.ndarray):
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
    pos = pos_init.copy()
    dir = dir_init.copy()
    e = e_init.copy()
    is_inside = np.array([True for _ in range(e.size)])

    while True:
        high_e_indices = np.argwhere(e > EMIN).flatten()
        full_condition = np.argwhere(is_inside[high_e_indices] == True)
        if not full_condition.any():
            break
        # Current values reference values inside the full list for which the condition is met
        full_condition = full_condition.flatten()
        e_current = e[full_condition]
        pos_current = pos[full_condition]
        dir_current = dir[full_condition]
        inside_current = is_inside[full_condition]

        free_path, p, dirp, _ = get_recoil_position(pos_current, dir_current)
        dee = eloss(e_current, free_path)
        e_current -= dee
        pos_current += free_path * dir_current
        inside_current = is_inside_target(pos_current)
        # The execution will continue for outside particles as well
        # TODO fix?
        dir_current, e_current, _, _ = scatter(e_current, dir_current, p, dirp)

    return pos, dir, e, is_inside