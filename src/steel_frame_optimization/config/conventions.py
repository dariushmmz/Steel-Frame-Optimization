"""Declare coordinate and sign conventions at the data boundary.

These conventions specify storage and reporting orientation only. They do not resolve DEC-018's
engineering treatment of axial force in hinge-capacity formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Axis(StrEnum):
    """Name the right-handed global coordinate axes."""

    X = "x"
    Y = "y"
    Z = "z"


class AxialSign(StrEnum):
    """Name the stored axial-force sign convention."""

    TENSION_POSITIVE = "tension_positive"


class RotationSign(StrEnum):
    """Name the stored planar rotation sign convention."""

    COUNTERCLOCKWISE_POSITIVE = "counterclockwise_positive_about_positive_z"


@dataclass(frozen=True, slots=True)
class CoordinateConvention:
    """Describe the immutable global and member-orientation convention."""

    horizontal_axis: Axis
    vertical_axis: Axis
    out_of_plane_axis: Axis
    gravity_direction: str
    member_local_x: str
    member_local_y: str
    axial_sign: AxialSign
    rotation_sign: RotationSign


CANONICAL_COORDINATE_CONVENTION: Final = CoordinateConvention(
    horizontal_axis=Axis.X,
    vertical_axis=Axis.Y,
    out_of_plane_axis=Axis.Z,
    gravity_direction="negative_y",
    member_local_x="from_i_node_to_j_node",
    member_local_y="positive_90_degrees_from_local_x_in_global_xy_plane",
    axial_sign=AxialSign.TENSION_POSITIVE,
    rotation_sign=RotationSign.COUNTERCLOCKWISE_POSITIVE,
)
