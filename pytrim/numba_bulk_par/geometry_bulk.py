"""Target-geometry related operations.

Currently, only a planar target geometry is supported.

Available functions:
    setup: setup module variables.
    is_inside_target: check if a given position is inside the target
"""

from numba import jit


def setup(zmin, zmax):
    """Define the geometry of the target.

    Parameters:
        zmin (int): minimum z coordinate of the target (A)
        zmax (int): maximum z coordinate of the target (A)

    Returns:
        None
    """
    global ZMIN, ZMAX

    ZMIN = zmin
    ZMAX = zmax


@jit(fastmath=True)
def is_inside_target(pos):
    """Check if a given position is inside the target.

    Parameters:
        pos (ndarray): position to check (size 3)

    Returns:
        bool: True if position is inside the target, False otherwise
    """
    return (ZMIN <= pos[:, 2]) & (pos[:, 2] <= ZMAX)
