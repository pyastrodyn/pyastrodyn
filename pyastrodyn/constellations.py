"""Constellations of satellites: Walker patterns and Keplerian propagation.

A constellation is described by the classical orbital elements of its
satellites, one value per satellite, in the order and with the conventions
of pyastrodyn-edu: ``a, e, i, raan, argp, f`` (true anomaly), radians.

Model of this version: KEPLERIAN motion (the ideal two-body problem). Each
satellite moves on a fixed ellipse; only its anomaly changes. There are no
perturbations: in reality the oblateness of the Earth turns the planes of a
constellation (for Galileo, by about 9.5 degrees per year, westward).

The Earth is turned at a constant rate about its axis (``dcm_EN``), starting
from a Greenwich angle given by the user: there are no time scales in this
version, so the results are not tied to a calendar date.

Arrays: for n satellites and m times, positions and velocities have the shape
(n, m, 3), angles the shape (n, m).
"""

import numpy as np

from ._edu import OMEGA_EARTH, R_EARTH, M2f, change_basis, coe2rv, dcm_EN, f2M, mean_motion


def walker_delta(t, p, f, a, i, raan0=0.0, arglat0=0.0):
    """Orbital elements of the satellites of a Walker delta constellation.

    In Walker's notation, i: t/p/f is a constellation of ``t`` satellites on
    circular orbits of the same radius and inclination ``i``, in ``p`` planes
    with their nodes equally spaced around the equator, ``t/p`` satellites
    equally spaced in each plane. The phasing ``f`` (an integer from 0 to
    p - 1) shifts the satellites of each plane forward, with respect to those
    of the plane before it, by f times 360/t degrees.

    Parameters
    ----------
    t : int
        Total number of satellites.
    p : int
        Number of planes; ``t`` must be a multiple of ``p``.
    f : int
        Phasing, 0 <= f <= p - 1.
    a : float
        Semi-major axis (the radius of the circular orbits).
    i : float
        Inclination, radians.
    raan0 : float
        Right ascension of the ascending node of the first plane, radians.
    arglat0 : float
        Argument of latitude of the first satellite of the first plane, radians.

    Returns
    -------
    a, e, i, raan, argp, f : ndarray, shape (t,)
        The elements of every satellite, plane by plane: satellite k is in
        plane k // (t/p). The orbits are circular, so, following the
        convention of pyastrodyn-edu, ``argp`` is zero and ``f`` is the
        argument of latitude.

    References
    ----------
    Walker, "Satellite constellations", *Journal of the British
    Interplanetary Society* 37 (1984); Wertz and Larson (eds.), *Space
    Mission Analysis and Design*.

    Notes
    -----
    Galileo, for example, is the constellation 56 deg: 24/3/1.
    """
    if any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in (t, p, f)):
        raise ValueError("t, p and f must be integers.")
    t, p, f = int(t), int(p), int(f)
    if not a > 0.0:
        raise ValueError("The semi-major axis a must be positive.")
    if t <= 0 or p <= 0 or t % p != 0:
        raise ValueError("t and p must be positive, and t a multiple of p.")
    if not 0 <= f < p:
        raise ValueError("The phasing f must be an integer from 0 to p - 1.")
    s = t // p                                           # satellites per plane
    # Step 1: plane index and position in the plane of every satellite.
    plane = np.repeat(np.arange(p), s)                   # 0, 0, ..., 1, 1, ...
    slot = np.tile(np.arange(s), p)                      # 0, 1, ..., 0, 1, ...
    # Step 2: nodes equally spaced; in each plane the satellites equally
    #         spaced, and shifted by f 2 pi / t from one plane to the next.
    raan = np.mod(raan0 + 2.0 * np.pi * plane / p, 2.0 * np.pi)
    arglat = np.mod(arglat0 + 2.0 * np.pi * slot / s + 2.0 * np.pi * f * plane / t, 2.0 * np.pi)
    ones = np.ones(t)
    return a * ones, 0.0 * ones, i * ones, raan, 0.0 * ones, arglat


def propagate_constellation(a, e, i, raan, argp, f, times, mu):
    """Positions and velocities of all the satellites at all the times (Keplerian motion).

    Parameters
    ----------
    a, e, i, raan, argp, f : float or ndarray, shape (n,)
        Elements of the n satellites at time 0 (elliptic orbits).
    times : ndarray, shape (m,)
        Times after time 0.
    mu : float
        Gravitational parameter.

    Returns
    -------
    r_N, v_N : ndarray, shape (n, m, 3)
        Components in the inertial basis N.

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*; Curtis,
    *Orbital Mechanics for Engineering Students*.

    Notes
    -----
    In Keplerian motion five elements are constant and the mean anomaly
    grows at the mean motion: M(t) = M0 + n t. The true anomaly follows from
    Kepler's equation (``M2f``), and the state from the elements
    (``coe2rv``). Exact for the model, to the tolerance of Kepler's equation.
    All the satellites and times are computed in one call, with no loop.
    """
    a, e, i, raan, argp, f = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (a, e, i, raan, argp, f))
    a, e, i, raan, argp, f = np.broadcast_arrays(a, e, i, raan, argp, f)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    # Step 1: one row per satellite, one column per time.
    column = (slice(None), None)
    M = f2M(f, e)[column] + mean_motion(a, mu)[column] * times[None, :]     # shape (n, m)
    # Step 2: the true anomaly at every time, then the states.
    f_t = M2f(np.mod(M, 2.0 * np.pi), e[column])
    return coe2rv(a[column], e[column], i[column], raan[column], argp[column], f_t, mu)


def to_earth_fixed(r_N, times, theta_g0=0.0, omega_earth=OMEGA_EARTH):
    """Positions in the Earth-fixed basis E, for an Earth turning at a constant rate.

    Parameters
    ----------
    r_N : ndarray, shape (n, m, 3)
        Positions in the inertial basis N (from ``propagate_constellation``).
    times : ndarray, shape (m,)
    theta_g0 : float
        Greenwich angle at time 0, radians.
    omega_earth : float
        Rotation rate of the Earth, rad/s.

    Returns
    -------
    r_E : ndarray, shape (n, m, 3)

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*.

    Notes
    -----
    The Greenwich angle grows uniformly: theta_g = theta_g0 + omega t.
    Precession, nutation and polar motion are not included (see ``dcm_EN``).
    """
    C_EN = dcm_EN(theta_g0 + omega_earth * np.asarray(times, dtype=float))     # shape (m, 3, 3)
    return change_basis(C_EN[None], np.asarray(r_N, dtype=float))


def ground_tracks(r_N, times, theta_g0=0.0, omega_earth=OMEGA_EARTH):
    """Latitude and longitude of the points below the satellites, on a spherical Earth.

    Parameters
    ----------
    r_N, times, theta_g0, omega_earth : as in ``to_earth_fixed``

    Returns
    -------
    latitude, longitude : ndarray, shape (n, m)
        Geocentric latitude in [-pi/2, pi/2] and longitude in (-pi, pi],
        radians.

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*; Curtis,
    *Orbital Mechanics for Engineering Students*.
    """
    r_E = to_earth_fixed(r_N, times, theta_g0, omega_earth)
    x, y, z = r_E[..., 0], r_E[..., 1], r_E[..., 2]
    return np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)


def elevation_from_station(r_N, times, latitude, longitude, theta_g0=0.0, omega_earth=OMEGA_EARTH,
                           radius=R_EARTH):
    """Elevation of every satellite above the local horizon of a ground station.

    Parameters
    ----------
    r_N, times, theta_g0, omega_earth : as in ``to_earth_fixed``
    latitude, longitude : float
        Geocentric latitude and longitude of the station, radians.
    radius : float
        Distance of the station from the centre of the Earth (spherical
        Earth); by default the equatorial radius.

    Returns
    -------
    elevation : ndarray, shape (n, m)
        Radians, from -pi/2 to pi/2. A satellite is above the horizon where
        the elevation is positive.

    References
    ----------
    Vallado, *Fundamentals of Astrodynamics and Applications*; Montenbruck
    and Gill, *Satellite Orbits*.

    Notes
    -----
    The local vertical is the direction from the centre of the Earth to the
    station. The elevation is the angle between the line of sight and the
    horizontal plane: sin(elevation) = (rho . up) / |rho|.
    """
    r_E = to_earth_fixed(r_N, times, theta_g0, omega_earth)
    # Step 1: the station and its local vertical, in the Earth-fixed basis.
    up_E = np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
    # Step 2: the line of sight from the station to every satellite.
    rho_E = r_E - radius * up_E
    rho = np.linalg.norm(rho_E, axis=-1)
    return np.arcsin(np.clip(rho_E @ up_E / rho, -1.0, 1.0))


def closest_approaches(r_N):
    """Smallest distance between any two satellites, at every time and overall.

    Parameters
    ----------
    r_N : ndarray, shape (n, m, 3)
        Positions of n >= 2 satellites at m times, in any one basis.

    Returns
    -------
    distance_min : float
        Smallest distance between two satellites over all the times.
    pair : tuple of int
        The two satellites (k, l), with k < l.
    time_index : int
        Index of the time at which it occurs.
    distance_by_time : ndarray, shape (m,)
        Smallest distance between two satellites at each time.

    References
    ----------
    Wertz and Larson (eds.), *Space Mission Analysis and Design*.

    Notes
    -----
    Only the sampled times are examined: two satellites may come closer
    between two samples. Use a time step small compared with the time two
    satellites take to cross each other's path, or refine around the result.
    The memory used grows as n^2 per time, and the times are processed one
    at a time.
    """
    r_N = np.asarray(r_N, dtype=float)
    if r_N.ndim != 3 or r_N.shape[2] != 3:
        raise ValueError("r_N must have the shape (n, m, 3).")
    n, m = r_N.shape[:2]
    if n < 2:
        raise ValueError("At least two satellites are needed.")
    if m < 1:
        raise ValueError("At least one time is needed.")
    if not np.all(np.isfinite(r_N)):
        raise ValueError("r_N contains values that are not finite.")
    upper = np.triu_indices(n, k=1)                      # every pair once
    distance_by_time = np.empty(m)
    best = (np.inf, (0, 1), 0)
    for k in range(m):
        # Distances between all the pairs at this time, shape (n, n).
        difference = r_N[:, None, k, :] - r_N[None, :, k, :]
        d = np.linalg.norm(difference, axis=-1)[upper]
        j = int(np.argmin(d))
        distance_by_time[k] = d[j]
        if d[j] < best[0]:
            best = (float(d[j]), (int(upper[0][j]), int(upper[1][j])), k)
    return best[0], best[1], best[2], distance_by_time


__all__ = ["walker_delta", "propagate_constellation", "to_earth_fixed", "ground_tracks",
           "elevation_from_station", "closest_approaches"]
