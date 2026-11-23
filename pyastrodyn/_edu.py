"""The functions of pyastrodyn-edu that pyastrodyn uses and passes through.

pyastrodyn keeps the names and the conventions of pyastrodyn-edu: the same
argument order, units, array shapes and basis suffixes. A function listed
here is taken from pyastrodyn-edu unchanged, and is available as
``pyastrodyn.<name>``.

Where speed is needed, pyastrodyn may later provide its own version of one of
these functions, under the same name and with the same contract; the tests
then check it against the pyastrodyn-edu version, which stays the reference.
No function is replaced in this version.

Only functions of the public release of pyastrodyn-edu 0.0.9 (Lectures 1 to
5 of the course) are used.
"""

import pyastrodyn_edu as _edu

#: Names taken from pyastrodyn-edu. A test checks that each one exists in the
#: release 0.0.9 and that pyastrodyn passes it through unchanged.
EDU_NAMES = (
    # constants
    "MU_EARTH", "R_EARTH", "OMEGA_EARTH",
    # conic sections and anomalies
    "period", "mean_motion", "M2f", "f2M",
    # orbital elements and rotations
    "coe2rv", "rv2coe", "dcm_EN", "change_basis",
)

for _name in EDU_NAMES:
    globals()[_name] = getattr(_edu, _name)

EDU_VERSION = _edu.__version__

__all__ = list(EDU_NAMES)
