"""Create the recoil position for the next collision.

Currently, only amorphous targets are supported. The free path length to
the next collision is assumed to be constant and equal to the atomic
density to the power -1/3.

Available functions:
    setup: setup module variables.
    get_recoil_position: get the recoil position.
"""
from math import sqrt, sin, cos
import numpy as np


def setup(density):
    """Setup module variables depending on target density.

    Parameters:
        density (float): target density (atoms/A^3)

    Returns:
        None    
    """
    global PMAX, MEAN_FREE_PATH

    MEAN_FREE_PATH = density**(-1/3)
    PMAX = MEAN_FREE_PATH / sqrt(np.pi)


def get_recoil_position(pos, dir):
    """Get the recoil position based on the projectile position and direction.

    Parameters:
        pos (ndarray): position of the projectile (size 3)
        dir (ndarray): direction vector of the projectile (size 3)

    Returns:
        float: free path length to the next collision (A)
        float: impact parameter = distance between collision point and 
            recoil (A)
        ndarray: direction vector from collision point to recoil (size 3)
        ndarray: position of the recoil (size 3)
    """
    free_path = MEAN_FREE_PATH
    pos_collision = pos + free_path * dir

    p = PMAX * np.sqrt(np.random.rand(pos.shape[0]))
    # Azimuthal angle fi
    fi = 2 * np.pi * np.random.rand(pos.shape[0])
    cos_fi = np.cos(fi)
    sin_fi = np.sin(fi)

    # Convert direction vectors to polar angles
    k = np.argmin(np.abs(dir), axis=1)  # make k point to the smallest dir so sin_alpha > sqrt(2/3)
    # TODO modulo operator is very compute intensive
    i = (k + 1) % 3
    j = (i + 1) % 3
    
    rows = np.arange(dir.shape[0])
    cos_alpha = dir[rows, k]
    sin_alpha = np.sqrt(dir[rows, i]**2 + dir[rows, j]**2)
    cos_phi = dir[rows, i] / sin_alpha
    sin_phi = dir[rows, j] / sin_alpha

    # direction vector from collision point to recoil
    dirp = np.zeros((pos.shape[0], 3))
    dirp[rows, i] = cos_fi * cos_alpha * cos_phi - sin_fi * sin_phi
    dirp[rows, j] = cos_fi * cos_alpha * sin_phi + sin_fi * cos_phi
    dirp[rows, k] = -cos_fi * sin_alpha
    # TODO what's that?
    norm = np.linalg.norm(dirp, axis=1, keepdims=True)
    dirp /= norm
    # position of the recoil
    # TODO
    pos_recoil = pos_collision[:] + p * dirp[:]

    return free_path, p, dirp, pos_recoil
