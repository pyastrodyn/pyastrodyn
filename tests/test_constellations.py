"""Tests of pyastrodyn: the pass-through of pyastrodyn-edu and the constellations."""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import pyastrodyn as pa
import pyastrodyn_edu as edu

MU = edu.MU_EARTH
DAY = 86400.0


# --- The functions of pyastrodyn-edu --------------------------------------------


def test_edu_functions_are_passed_through_unchanged():
    for name in pa.EDU_NAMES:
        assert hasattr(edu, name), f"{name} is not in the installed pyastrodyn-edu"
        assert getattr(pa, name) is getattr(edu, name), f"{name} is not the pyastrodyn-edu object"


def test_only_functions_of_the_part_1_release_are_used():
    # The public release 0.0.9 of pyastrodyn-edu covers Lectures 1 to 5 of the course. Every name
    # of pyastrodyn-edu used by the library, the tests and the examples must be in this list.
    # Limitation: the test cannot see which release is installed beyond its version number.
    part_1 = {"MU_EARTH", "R_EARTH", "OMEGA_EARTH", "period", "mean_motion", "M2f", "f2M", "coe2rv",
              "rv2coe", "dcm_EN", "change_basis", "propagate_kepler_rv"}
    assert set(pa.EDU_NAMES) <= part_1
    root = Path(__file__).resolve().parent.parent
    used = set()
    for path in [*root.glob("pyastrodyn/*.py"), *root.glob("tests/*.py"), *root.glob("examples/*.py")]:
        text = path.read_text()
        used |= set(re.findall(r"\b(?:edu|_edu)\.([A-Za-z_]\w*)", text))
        used |= set(re.findall(r"\bpa\.([A-Za-z_]\w*)", text)) & set(dir(edu))
    used -= {"__version__"}
    assert used <= part_1, used - part_1
    major, minor, patch = (int(x) for x in re.findall(r"\d+", pa.EDU_VERSION)[:3])
    assert (major, minor) == (0, 0) and patch >= 9


def test_every_function_documents_its_references():
    for name in pa.constellations.__all__:
        doc = inspect.getdoc(getattr(pa, name))
        assert "References\n----------" in doc and "Parameters\n----------" in doc, name


# --- Walker constellations -----------------------------------------------------


def test_walker_delta_geometry():
    t, p, f, i = 24, 3, 1, np.radians(56.0)
    a, e, inc, raan, argp, arglat = pa.walker_delta(t, p, f, 29600.0, i)
    assert a.shape == (t,) and np.all(a == 29600.0) and np.all(e == 0.0) and np.all(inc == i) and np.all(argp == 0.0)
    s = t // p
    planes = raan.reshape(p, s)
    assert np.allclose(planes, planes[:, :1])                                        # one node per plane
    assert np.allclose(np.diff(planes[:, 0]), 2 * np.pi / p)                         # nodes 120 deg apart
    u = arglat.reshape(p, s)
    assert np.allclose(np.mod(np.diff(u, axis=1), 2 * np.pi), 2 * np.pi / s)         # 45 deg apart in a plane
    assert np.allclose(np.mod(u[1:, 0] - u[:-1, 0], 2 * np.pi), 2 * np.pi * f / t)   # 15 deg between planes


def test_walker_delta_rejects_impossible_patterns():
    with pytest.raises(ValueError):
        pa.walker_delta(25, 3, 1, 7000.0, 1.0)
    with pytest.raises(ValueError):
        pa.walker_delta(24, 3, 3, 7000.0, 1.0)


# --- Propagation -------------------------------------------------------------------


def test_propagation_matches_one_satellite_at_a_time():
    rng = np.random.default_rng(3)
    n = 7
    a = rng.uniform(7000.0, 30000.0, n)
    e = rng.uniform(0.0, 0.6, n)
    a = np.maximum(a, 6700.0 / (1 - e))                                               # perigee above the Earth
    i, raan, argp, f = (rng.uniform(0.0, 3.0, n) for _ in range(4))
    times = np.linspace(0.0, 2.0 * DAY, 37)
    r_N, v_N = pa.propagate_constellation(a, e, i, raan, argp, f, times, MU)
    assert r_N.shape == (n, times.size, 3)
    for k in range(n):
        r0_N, v0_N = edu.coe2rv(a[k], e[k], i[k], raan[k], argp[k], f[k], MU)
        for j, t in enumerate(times):
            r_ref, v_ref = edu.propagate_kepler_rv(r0_N, v0_N, t, MU)
            assert np.allclose(r_N[k, j], r_ref, rtol=0, atol=1e-6)
            assert np.allclose(v_N[k, j], v_ref, rtol=0, atol=1e-9)


def test_propagation_matches_a_numerical_integration():
    # Independent of the Kepler solver: integrate r'' = -mu r / |r|^3 with a fixed-step RK4.
    a, e, i, raan, argp, f = 26560.0, 0.7, 1.1, 0.4, 2.0, 0.3
    duration = 30000.0
    r_N, v_N = pa.propagate_constellation(a, e, i, raan, argp, f, [0.0, duration], MU)
    y = np.concatenate([r_N[0, 0], v_N[0, 0]])

    def rate(y):
        r = y[:3]
        return np.concatenate([y[3:], -MU * r / np.linalg.norm(r) ** 3])

    steps = 60000
    h = duration / steps
    for _ in range(steps):
        k1 = rate(y); k2 = rate(y + h / 2 * k1); k3 = rate(y + h / 2 * k2); k4 = rate(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(y[:3], r_N[0, 1], rtol=0, atol=1e-5)
    assert np.allclose(y[3:], v_N[0, 1], rtol=0, atol=1e-8)


def test_input_errors():
    with pytest.raises(ValueError):
        pa.walker_delta(24.7, 3, 1, 29600.0, 1.0)
    with pytest.raises(ValueError):
        pa.walker_delta(24, 3, 1, -7000.0, 1.0)
    with pytest.raises(ValueError):
        pa.closest_approaches(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        pa.closest_approaches(np.zeros((4, 0, 3)))
    with pytest.raises(ValueError):
        pa.closest_approaches(np.full((4, 2, 3), np.nan))


def test_propagation_conserves_energy_and_angular_momentum():
    el = pa.walker_delta(12, 4, 2, 7178.0, np.radians(53.0))
    times = np.linspace(0.0, DAY, 500)
    r_N, v_N = pa.propagate_constellation(*el, times, MU)
    energy = 0.5 * np.sum(v_N**2, axis=-1) - MU / np.linalg.norm(r_N, axis=-1)
    h_N = np.cross(r_N, v_N)
    assert np.allclose(energy, -MU / (2 * 7178.0), rtol=1e-12)
    assert np.allclose(h_N, h_N[:, :1], rtol=1e-10, atol=1e-6)
    assert np.allclose(np.linalg.norm(r_N, axis=-1), 7178.0, rtol=1e-12)              # circular orbits


def test_one_satellite_and_one_time_give_the_expected_shapes():
    r_N, v_N = pa.propagate_constellation(7000.0, 0.01, 1.0, 0.0, 0.0, 0.0, 600.0, MU)
    assert r_N.shape == (1, 1, 3) and v_N.shape == (1, 1, 3)


# --- Ground tracks and visibility --------------------------------------------------


def test_ground_track_reaches_the_inclination_and_drifts_west():
    i = np.radians(56.0)
    el = pa.walker_delta(1, 1, 0, 29600.0, i)
    T = edu.period(29600.0, MU)
    times = np.linspace(0.0, T, 20001)
    r_N, _ = pa.propagate_constellation(*el, times, MU)
    latitude, longitude = pa.ground_tracks(r_N, times)
    assert np.isclose(latitude.max(), i, atol=1e-6) and np.isclose(latitude.min(), -i, atol=1e-6)
    # After one period the satellite is back at the node; the Earth has turned by omega T underneath.
    drift = np.mod(longitude[0, -1] - longitude[0, 0] + np.pi, 2 * np.pi) - np.pi
    expected = np.mod(-edu.OMEGA_EARTH * T + np.pi, 2 * np.pi) - np.pi
    assert np.isclose(drift, expected, atol=1e-9)


def test_elevation_overhead_and_below_the_horizon():
    # A satellite on the x axis at time 0, with the Greenwich angle 0: it is over latitude 0, longitude 0.
    r_N = np.array([[[42164.0, 0.0, 0.0]]])
    assert np.isclose(pa.elevation_from_station(r_N, [0.0], 0.0, 0.0)[0, 0], np.pi / 2)
    assert pa.elevation_from_station(r_N, [0.0], 0.0, np.pi)[0, 0] < -np.pi / 2 + 0.2  # the other side
    # On the horizon: the line of sight is tangent to the sphere.
    R = edu.R_EARTH
    r_tangent = np.array([[[R, 30000.0, 0.0]]])
    assert np.isclose(pa.elevation_from_station(r_tangent, [0.0], 0.0, 0.0)[0, 0], 0.0, atol=1e-12)


def test_earth_fixed_rotation():
    r_N = np.array([[[7000.0, 0.0, 0.0]]])
    quarter_turn = np.pi / 2 / edu.OMEGA_EARTH
    r_E = pa.to_earth_fixed(r_N, [quarter_turn])
    assert np.allclose(r_E[0, 0], [0.0, -7000.0, 0.0], atol=1e-9)                     # the Earth turned east


# --- Closest approaches -----------------------------------------------------------------


def test_closest_approaches_against_a_direct_search():
    rng = np.random.default_rng(5)
    r_N = rng.normal(size=(9, 13, 3)) * 1000.0
    distance, pair, time_index, by_time = pa.closest_approaches(r_N)
    best = np.inf
    for k in range(9):
        for l in range(k + 1, 9):
            for j in range(13):
                d = np.linalg.norm(r_N[k, j] - r_N[l, j])
                if d < best:
                    best, best_pair, best_time = d, (k, l), j
    assert np.isclose(distance, best) and pair == best_pair and time_index == best_time
    assert np.isclose(by_time.min(), best) and by_time.shape == (13,)
    with pytest.raises(ValueError):
        pa.closest_approaches(r_N[:1])


def test_galileo_like_constellation_results():
    # The numbers of the worked example 1, with its original data.
    el = pa.walker_delta(24, 3, 1, edu.R_EARTH + 23222.0, np.radians(56.0))
    times = np.arange(0.0, DAY + 1.0, 60.0)
    r_N, _ = pa.propagate_constellation(*el, times, MU)
    distance, pair, time_index, _ = pa.closest_approaches(r_N)
    assert 4000.0 < distance < 5500.0
    elevation = pa.elevation_from_station(r_N, times, np.radians(48.14), np.radians(11.58))
    visible = np.sum(elevation > np.radians(10.0), axis=0)
    assert visible.min() >= 4                          # at least four: enough for a position fix
