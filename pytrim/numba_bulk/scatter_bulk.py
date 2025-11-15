"""Treat the scattering of a projectile on a target atom.

Currently, only the ZBL potential (Ziegler, Biersack, Littmark,
The Stopping and Range of Ions in Matter, Pergamon Press, 1985) is
implemented, along with Biersack's "magic formula" for the scattering
angle.

Available functions:
    setup: setup module variables.
    scatter: treat a scattering event.
"""

import numpy as np
from numba import jit


def setup(z1, m1, z2, m2):
    """Setup module variables depending on projectile and target species.

    Parameters:
        z1 (int): atomic number of projectile
        m1 (float): mass of projectile (amu)
        z2 (int): atomic number of target
        m2 (float): mass of target (amu)

    Returns:
        None
    """
    global ENORM, RNORM, DIRFAC, DENFAC

    m1_m2 = m1 / m2
    RNORM = 0.4685 / (z1**0.23 + z2**0.23)  # A
    ENORM = 14.39979 * z1 * z2 / RNORM * (1 + m1_m2)  # eV
    DIRFAC = 2 / (1 + m1_m2)
    DENFAC = 4 * m1_m2 / (1 + m1_m2) ** 2


# Constants for ZBL screening function
A1 = 0.18175
A2 = 0.50986
A3 = 0.28022
A4 = 0.02817

B1 = 3.1998
B2 = 0.94229
B3 = 0.4029
B4 = 0.20162

A1B1 = A1 * B1
A2B2 = A2 * B2
A3B3 = A3 * B3
A4B4 = A4 * B4


@jit(fastmath=False)
def ZBLscreen(r: np.ndarray):
    """Calculate the ZBL screening function and its derivative.

    Parameters:
        r (float): Distance (RNORM)

    Returns:
        float: ZBL potential at distance r (ENORM)
        float: derivative of ZBL potential at distance r (ENORM/RNORM)
    """
    exp1 = np.exp(-B1 * r)
    exp2 = np.exp(-B2 * r)
    exp3 = np.exp(-B3 * r)
    exp4 = np.exp(-B4 * r)
    screen = A1 * exp1 + A2 * exp2 + A3 * exp3 + A4 * exp4
    dscreen = -(A1B1 * exp1 + A2B2 * exp2 + A3B3 * exp3 + A4B4 * exp4)

    return screen, dscreen


# Constants for apsis estimation for the ZBL potential
K2 = 0.38  # factor of the 1/R part
K3 = 7.2  # factor of the 1/R^3 part
K1 = 1 / (4 * K2)
R12sq = (2 * K2) ** 2
R23sq = K3 / K2
NITER = 1  # number of Newton-Raphson iterations


@jit(fastmath=False)
def estimate_apsis(e: np.ndarray, p: np.ndarray):
    """Estimate the distance of closest approach (apsis) in a colllision.

    Parameters:
        e (float): energy of projectile before the collision (ENORM)
        p (float): impact parameter (RNORM)

    Returns:
        float: Estimated apsis of the collision (RNORM)
    """
    psq = p**2
    r0sq = 0.5 * (psq + np.sqrt(psq**2 + 4 * K3 / e))

    r0sq = np.where(r0sq < R23sq, psq + K2 / e, r0sq)
    r0 = np.where(
        r0sq < R12sq,
        (1 + np.sqrt(1 + 4 * e * (e + K1) * psq)) / (2 * (e + K1)),
        np.sqrt(r0sq),
    )

    # Do Newton-Raphson iterations to improve the estimate
    for _ in range(NITER):
        screen, dscreen = ZBLscreen(r0)
        numerator = r0 * (r0 - screen / e) - psq
        denominator = 2 * r0 - (screen + r0 * dscreen) / e
        r0 -= numerator / denominator

        residuum = 1 - screen / (e * r0) - psq / r0**2
        # NOTE Values below threshold may be filtered out
        # NOTE in the future to speed up processing with large number of iterations
        if np.all(np.abs(residuum) < 1e-4):
            break

    return r0


C1 = 0.99229
C2 = 0.011615
C3 = 0.007122
C4 = 14.813
C5 = 9.3066


@jit(fastmath=False)
def magic(e: np.ndarray, p_init: np.ndarray):
    """Calculate CM scattering angle using Biersack's magic formula.

    Parameters:
        e (float): energy of projectile before the collision (ENORM)
        p (float): impact parameter (RNORM)

    Returns:
        float: cosine of half the scattering angle in the center-of-mass
            system
    """
    p = p_init.copy().flatten()
    r0 = estimate_apsis(e, p)
    screen, dscreen = ZBLscreen(r0)

    rho = 2 * (e * r0 - screen) / (screen / r0 - dscreen)
    sqrte = np.sqrt(e)
    alpha = 1 + C1 / sqrte
    beta = (C2 + sqrte) / (C3 + sqrte)
    gamma = (C4 + e) / (C5 + e)
    a = 2 * alpha * e * p**beta
    g = gamma / (np.sqrt(1 + a**2) - a)
    delta = a * (r0 - p) / (1 + g)

    cos_half_theta = (p + rho + delta) / (r0 + rho)
    if np.any(cos_half_theta > 1):
        print("Warning: cos_half_theta > 1:", cos_half_theta)
        print("  e =", e, "p =", p, "r0 =", r0, "rho =", rho, "delta =", delta)

    return cos_half_theta


@jit(fastmath=False)
def scatter(e: np.ndarray, dir: np.ndarray, p: np.ndarray, dirp: np.ndarray):
    """Treat a scattering event.

    The atomic numbers and masses of the projectile and target enter the
    calculation via the module variables ENORM, PNORM, DIRFAC, and DENFAC.

    The direction vectors dir and dirp are assumed to be normalized to
    unit length.

    Parameters:
        e (float): energy of the projectile before the collision (eV)
        dir (ndarray): direction vector of the projectile before
            the collision (size 3)
        p (float): impact parameter (A)
        dirp (ndarray): direction vector of the impact parameter
            (= from the collision point to the recoil position before
            the collision) (size 3)

    Returns:
        ndarray: direction vector of the projectile after the collision
            (size 3)
        float: energy of the projectile after the collision (eV)
        ndarray: direction vector of the recoil after the collision
            (size 3)
        float: energy of the projectile after the collision
    """
    # scattering angle theta in the center-of-mass system
    cos_half_theta = magic(e / ENORM, p / RNORM)
    cos_half_theta = cos_half_theta[:, np.newaxis]

    # directions of the recoil and the projectile after the collision
    sin_psi = cos_half_theta
    cos_psi = np.sqrt(1 - sin_psi**2)
    dir_recoil = DIRFAC * cos_psi * (cos_psi * dir + sin_psi * dirp)

    dir_new = dir - dir_recoil
    norm = np.sqrt(np.sum(dir_new**2, axis=1)).reshape(-1, 1)
    dir_new /= norm

    norm = np.sqrt(np.sum(dir_recoil**2, axis=1)).reshape(-1, 1)
    dir_recoil /= norm

    # energy after scattering
    e_recoil = DENFAC * e * (1 - cos_half_theta.flatten() ** 2)
    e -= e_recoil

    return dir_new, e, dir_recoil, e_recoil
