# pyastrodyn

> **Provided as is.** This is research software, made available without any
> warranty. Installation and use are not supported.

pyastrodyn is a research library in astrodynamics, the companion of the
educational library [pyastrodyn-edu](https://pypi.org/project/pyastrodyn-edu/).
It adds capabilities that are not part of the course library.

Version 0.0.1 contains one capability: **constellations in Keplerian motion**.

- `walker_delta`: the orbital elements of a Walker delta constellation i: t/p/f.
- `propagate_constellation`: all the satellites at all the times in one call.
- `to_earth_fixed`, `ground_tracks`: positions in the Earth-fixed basis and ground tracks.
- `elevation_from_station`: elevation of every satellite seen from a ground station.
- `closest_approaches`: smallest distance between two satellites over a time span.

## Model and limits

- Two-body Keplerian motion: no oblateness, drag or third bodies. The planes
  of a real constellation turn under the oblateness of the Earth; here they do not.
- Spherical Earth turning at a constant rate about the third axis; geocentric latitude.
- No time scales: times are seconds from an epoch at which the Greenwich angle
  is given (zero by default).

## Same names and conventions as pyastrodyn-edu

pyastrodyn uses the names, units and conventions of pyastrodyn-edu: plain
NumPy functions, one or many states per call, explicit `mu`, radians, element
order `a, e, i, raan, argp, f` (for circular orbits `argp = 0` and `f` is the
argument of latitude), basis suffixes such as `r_N` and `C_EN`.

The basic functions (`coe2rv`, `rv2coe`, `M2f`, `f2M`, `period`,
`mean_motion`, `dcm_EN`, `change_basis` and the Earth constants) are those of
pyastrodyn-edu, passed through unchanged: `pyastrodyn.coe2rv` *is*
`pyastrodyn_edu.coe2rv`. A later version may replace one of them with a faster
implementation under the same name and with the same contract, tested against
pyastrodyn-edu as reference. In this version no function is replaced.

Version 0.0.1 uses only the functions of the public release of pyastrodyn-edu
for Lectures 1 to 5 (version 0.0.9).

## Installation

Python 3.11 or later.

```
pip install pyastrodyn            # when published on PyPI
pip install -e ".[plot,dev]"      # from a copy of the repository
```

Repository: <https://github.com/pyastrodyn/pyastrodyn>

## Worked example

`examples/example_1_navigation_constellation.py` and the notebook with the same
name: a Galileo-like constellation (Walker 56°: 24/3/1, altitude 23222 km) over
one day, with a three-dimensional view, ground tracks, the closest approach
between satellites and the satellites visible from Munich. The notebook is
stored without results; run it to see them.

The notebook is written from the script:

```
python tools/make_notebooks.py
```

## Tests

```
python -m pytest
```

## How the code was written

The code was written through a code-by-intent process: the author defined what
each function does, its conventions and its tests, and directed and reviewed
the work; an AI assistant (Claude, Anthropic) wrote the code and the
documentation. The results are checked by automatic tests. The author decided
and reviewed all of its content; as stated in the license, it is provided as
is, without warranty or liability. It may still contain mistakes; please
report them to the author.

## References

- J. G. Walker, "Satellite constellations", *Journal of the British Interplanetary Society*, 1984.
- Wertz and Larson, *Space Mission Analysis and Design*.

## Licence and citation

MIT License, see `LICENSE`. To cite pyastrodyn, see `CITATION.cff`.
