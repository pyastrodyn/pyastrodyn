"""pyastrodyn: astrodynamics in Python, built on the conventions of pyastrodyn-edu.

This first version adds constellations of satellites in Keplerian motion
(``constellations``). The basic functions (orbital elements, anomalies,
rotations, constants) are those of pyastrodyn-edu, passed through unchanged,
with the same names and conventions (``_edu``).

Copyright (c) 2026 Dr. Marcello Romano. MIT License.
"""

__version__ = "0.0.2"

from . import constellations  # noqa: E402
from ._edu import *  # noqa: F401,F403,E402
from ._edu import EDU_NAMES, EDU_VERSION  # noqa: F401,E402
from .constellations import *  # noqa: F401,F403,E402
