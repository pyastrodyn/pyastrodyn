# %% [markdown]
# # Worked example 1: A navigation constellation
#
# **pyastrodyn 0.0.1, constellations in Keplerian motion.** Inspired by the
# Galileo constellation; the statement was written for this library. The
# nominal Galileo values (Walker 56 deg: 24/3/1, altitude 23222 km) are those
# published by ESA and by the European GNSS Service Centre.
#
# **Problem.** A navigation constellation is the Walker pattern
# 56 deg: 24/3/1 (24 satellites in 3 planes inclined at 56 deg) at an
# altitude of 23222 km, the nominal values of Galileo. Over one day:
#
# 1. build the constellation and draw it in three dimensions;
# 2. propagate it and draw the ground tracks;
# 3. find the smallest distance between two satellites, and when it occurs;
# 4. count how many satellites a receiver in Munich sees above 10 deg of
#    elevation.
#
# This example exists as a notebook and as a script with the same content.
#
# Model: Keplerian motion, no perturbations; a spherical Earth turning at a
# constant rate, with the Greenwich angle zero at the start. Units: km, s;
# angles in radians inside the computation, degrees in the printouts.

# %%
import matplotlib.pyplot as plt
import numpy as np

import pyastrodyn as pa

# %% [markdown]
# ## 1. Given data
#
# These are the numbers to play with. To return to the original problem,
# type the original values back (they are given in the comments) and run
# everything again.

# %%
t_total = 24             # number of satellites, at least 2 (original: 24)
planes = 3               # number of planes (original: 3)
phasing = 1              # Walker phasing, from 0 to planes - 1 (original: 1)
altitude = 23222.0       # altitude of the circular orbits, km (original: 23222.0)
inclination_deg = 56.0   # inclination, degrees (original: 56.0)
station_lat_deg = 48.14  # latitude of the receiver, degrees (Munich; original: 48.14)
station_lon_deg = 11.58  # longitude of the receiver, degrees (Munich; original: 11.58)
min_elevation_deg = 10.0 # smallest useful elevation, degrees (original: 10.0)
duration = 86400.0       # time span, s (original: 86400.0, one day)
step = 60.0              # time step, s (original: 60.0)
mu = pa.MU_EARTH         # gravitational parameter of the Earth, km^3/s^2 (original: pa.MU_EARTH)
R = pa.R_EARTH           # equatorial radius of the Earth, km (original: pa.R_EARTH)

# %% [markdown]
# ## 2. The constellation
#
# `walker_delta` returns the elements of every satellite, plane by plane.
# The orbits are circular, so the argument of periapsis is zero and the
# "true anomaly" is the argument of latitude (the convention of
# pyastrodyn-edu).

# %%
a = R + altitude
elements = pa.walker_delta(t_total, planes, phasing, a, np.radians(inclination_deg))
a_k, e_k, i_k, raan_k, argp_k, u_k = elements
T = pa.period(a, mu)
print(f"semi-major axis {a:.1f} km, period {T / 3600.0:.3f} h")
print(" sat  plane   RAAN deg   arg. of latitude deg")
for k in range(t_total):
    print(f"{k:4d} {k // (t_total // planes):6d} {np.degrees(raan_k[k]):10.1f} {np.degrees(u_k[k]):12.1f}")

# %% [markdown]
# ## 3. Propagation and three-dimensional view
#
# `propagate_constellation` returns all the satellites at all the times in
# one call: positions of shape (satellites, times, 3).

# %%
times = np.arange(0.0, duration + step / 2.0, step)
r_N, v_N = pa.propagate_constellation(*elements, times, mu)
print("shape of the positions:", r_N.shape)

fig = plt.figure(figsize=(6.5, 6.0))
ax = fig.add_subplot(projection="3d")
colors = plt.cm.tab10(np.arange(planes))
orbit = times <= T                                       # one revolution is enough to draw the orbits
for k in range(t_total):
    color = colors[k // (t_total // planes)]
    ax.plot(*r_N[k, orbit].T, color=color, lw=0.6)
    ax.scatter(*r_N[k, 0], color=color, s=14)
sphere_u, sphere_v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
ax.plot_surface(R * np.cos(sphere_u) * np.sin(sphere_v), R * np.sin(sphere_u) * np.sin(sphere_v),
                R * np.cos(sphere_v), color="lightsteelblue", alpha=0.5, lw=0)
ax.set_box_aspect((1, 1, 1))
limit = 1.05 * a
ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
ax.set_xlabel("x, km"); ax.set_ylabel("y, km"); ax.set_zlabel("z, km")
ax.set_title("Satellites at time 0 and their orbits")
plt.show()

# %% [markdown]
# ## 4. Ground tracks
#
# The points below the satellites, on a spherical Earth that turns under the
# orbits. Each satellite makes about 1.7 revolutions in a day.

# %%
latitude, longitude = pa.ground_tracks(r_N, times)
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for k in range(t_total):
    lon = np.degrees(longitude[k])
    lon[np.abs(np.diff(lon, prepend=lon[0])) > 180.0] = np.nan    # do not join across the date line
    ax.plot(lon, np.degrees(latitude[k]), color=colors[k // (t_total // planes)], lw=0.7)
ax.plot(station_lon_deg, station_lat_deg, "k*", ms=11, label="Munich")
ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
ax.set_xlabel("longitude, deg"); ax.set_ylabel("latitude, deg")
ax.grid(True, alpha=0.4); ax.legend(loc="lower left")
ax.set_title("Ground tracks over one day")
plt.show()

# %% [markdown]
# ## 5. Closest approach between two satellites
#
# `closest_approaches` compares every pair of satellites at every time.

# %%
distance_min, pair, time_index, distance_by_time = pa.closest_approaches(r_N)
print(f"smallest distance {distance_min:.1f} km, between satellites {pair[0]} and {pair[1]}, "
      f"at t = {times[time_index] / 3600.0:.3f} h")
print(f"planes of the two satellites: {pair[0] // (t_total // planes)} and {pair[1] // (t_total // planes)}")

fig, ax = plt.subplots(figsize=(8.0, 3.2))
ax.plot(times / 3600.0, distance_by_time, color="tab:blue")
ax.set_xlabel("time, h"); ax.set_ylabel("smallest distance, km"); ax.grid(True, alpha=0.4)
plt.show()

# %% [markdown]
# ## 6. Satellites visible from Munich
#
# `elevation_from_station` gives the elevation of every satellite above the
# local horizon. A receiver needs at least four satellites for a position.

# %%
elevation = pa.elevation_from_station(r_N, times, np.radians(station_lat_deg), np.radians(station_lon_deg))
visible = np.sum(elevation > np.radians(min_elevation_deg), axis=0)
print(f"satellites above {min_elevation_deg:.0f} deg: from {visible.min()} to {visible.max()}, "
      f"on average {visible.mean():.1f}")

fig, ax = plt.subplots(figsize=(8.0, 3.2))
ax.step(times / 3600.0, visible, where="post", color="tab:green")
ax.axhline(4, color="0.5", ls=":", lw=1.0)
ax.set_xlabel("time, h"); ax.set_ylabel("visible satellites"); ax.grid(True, alpha=0.4)
plt.show()

# %% [markdown]
# ## 7. Independent checks
#
# 1. Walker geometry: nodes 360/p deg apart, satellites 360 p/t deg apart in
#    each plane, and a shift of 360 f/t deg from one plane to the next.
# 2. Consistency with pyastrodyn-edu: a satellite propagated alone with
#    `propagate_kepler_rv` is at the same position. (This uses the same
#    Kepler solver; the tests of the library also compare with a numerical
#    integration of the equation of motion.)
# 3. Closest approach: the same pair, propagated again around the time found
#    on a grid 100 times finer, comes at most a little closer (the sampling
#    can only miss a closer point between two samples).
# 4. Visibility: the elevation computed directly, with explicit vectors,
#    agrees with `elevation_from_station`.

# %%
import pyastrodyn_edu as edu  # noqa: E402  (used only for the independent checks)

s = t_total // planes
assert np.allclose(np.mod(np.diff(raan_k.reshape(planes, s)[:, 0]), 2 * np.pi), 2 * np.pi / planes)
assert np.allclose(np.mod(np.diff(u_k.reshape(planes, s), axis=1), 2 * np.pi), 2 * np.pi / s)
assert np.allclose(np.mod(u_k.reshape(planes, s)[1:, 0] - u_k.reshape(planes, s)[:-1, 0], 2 * np.pi),
                   2 * np.pi * phasing / t_total)

for k in (0, t_total // 2, t_total - 1):
    r0_N, v0_N = edu.coe2rv(a_k[k], e_k[k], i_k[k], raan_k[k], argp_k[k], u_k[k], mu)
    r_ref, _ = edu.propagate_kepler_rv(r0_N, v0_N, times[-1], mu)
    assert np.allclose(r_N[k, -1], r_ref, atol=1e-6)

fine = np.linspace(max(times[time_index] - step, 0.0), times[time_index] + step, 201)
r_fine, _ = pa.propagate_constellation(*(x[list(pair)] for x in elements), fine, mu)
fine_min = np.linalg.norm(r_fine[0] - r_fine[1], axis=-1).min()
print(f"closest approach on a finer grid: {fine_min:.1f} km (sampled: {distance_min:.1f} km)")
assert fine_min <= distance_min + 1e-9 and distance_min - fine_min < 0.01 * a

lat0, lon0 = np.radians(station_lat_deg), np.radians(station_lon_deg)
k, j = pair[0], len(times) // 3
theta = pa.OMEGA_EARTH * times[j]                           # Greenwich angle at that time
station_N = R * np.array([np.cos(lat0) * np.cos(lon0 + theta), np.cos(lat0) * np.sin(lon0 + theta), np.sin(lat0)])
line_N = r_N[k, j] - station_N
direct = np.arcsin(line_N @ station_N / (np.linalg.norm(line_N) * R))
assert np.isclose(direct, elevation[k, j])
print("The checks agree.")

# %% [markdown]
# ## 8. What to observe
#
# - The three planes are 120 deg apart, and in each plane the satellites are
#   45 deg apart. The phasing 1 shifts each plane by 15 deg with respect to
#   the previous one.
# - The ground tracks reach 56 deg of latitude, the inclination.
# - The smallest distance occurs between satellites of different planes,
#   near the points where the planes cross. By the symmetry of the pattern,
#   other pairs come equally close at other times: the pair printed is the
#   first one found.
# - From Munich at least four satellites are above 10 deg at all times, as
#   needed for a position fix.
# - In this model the constellation never changes. In reality the oblateness
#   of the Earth turns the planes, by about 9.5 deg per year, westward, for Galileo.
#
# ## Now try
#
# 1. Set `phasing = 0` and `phasing = 2`. How does the closest approach change?
# 2. Set `planes = 6` (24 satellites in 6 planes). How many satellites does
#    Munich see now?
# 3. Set `station_lat_deg = 80.0`. Is the inclination of 56 deg enough near
#    the pole?
# 4. Compare the period with 10 sidereal days (861641 s): Galileo makes 17
#    revolutions in that time, so its ground tracks repeat every ten days.
