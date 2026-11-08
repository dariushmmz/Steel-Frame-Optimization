"""Independent layer-2 beam, portal, and generalized-eigenproblem benchmarks."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import openseespy.opensees as ops
import pytest

pytestmark = pytest.mark.analytical

RELATIVE_TOLERANCE = 1e-6


def _run_linear_static() -> None:
    """Execute one deterministic linear step for a closed-form fixture."""

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0


def test_simply_supported_beam_deflection_reactions_and_end_moments() -> None:
    """Match 5wL^4/(384EI), equal reactions, and zero simple-support end moments."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    area = 0.02
    length = 4.0
    line_load = 5_000.0
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, length / 2.0, 0.0)
        ops.node(3, length, 0.0)
        ops.fix(1, 1, 1, 0)
        ops.fix(3, 0, 1, 0)
        ops.geomTransf("Linear", 1)
        ops.element("elasticBeamColumn", 1, 1, 2, area, elastic_modulus, inertia, 1)
        ops.element("elasticBeamColumn", 2, 2, 3, area, elastic_modulus, inertia, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.eleLoad("-ele", 1, 2, "-type", "-beamUniform", -line_load)
        _run_linear_static()
        ops.reactions()
        expected_deflection = -5.0 * line_load * length**4 / (384.0 * elastic_modulus * inertia)
        assert float(ops.nodeDisp(2, 2)) == pytest.approx(
            expected_deflection, rel=RELATIVE_TOLERANCE
        )
        assert float(ops.nodeReaction(1, 2)) == pytest.approx(line_load * length / 2.0)
        assert float(ops.nodeReaction(3, 2)) == pytest.approx(line_load * length / 2.0)
        assert float(ops.eleResponse(1, "localForce")[2]) == pytest.approx(0.0, abs=1e-8)
        assert float(ops.eleResponse(2, "localForce")[5]) == pytest.approx(0.0, abs=1e-8)
    finally:
        ops.wipe()


def _frame_element_stiffness(
    elastic_modulus: float,
    area: float,
    inertia: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Return one independently assembled global Euler-Bernoulli frame stiffness matrix."""

    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    cosine = delta_x / length
    sine = delta_y / length
    axial = elastic_modulus * area / length
    flexural = elastic_modulus * inertia
    local = np.array(
        [
            [axial, 0, 0, -axial, 0, 0],
            [
                0,
                12 * flexural / length**3,
                6 * flexural / length**2,
                0,
                -12 * flexural / length**3,
                6 * flexural / length**2,
            ],
            [
                0,
                6 * flexural / length**2,
                4 * flexural / length,
                0,
                -6 * flexural / length**2,
                2 * flexural / length,
            ],
            [-axial, 0, 0, axial, 0, 0],
            [
                0,
                -12 * flexural / length**3,
                -6 * flexural / length**2,
                0,
                12 * flexural / length**3,
                -6 * flexural / length**2,
            ],
            [
                0,
                6 * flexural / length**2,
                2 * flexural / length,
                0,
                -6 * flexural / length**2,
                4 * flexural / length,
            ],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [cosine, sine, 0, 0, 0, 0],
            [-sine, cosine, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cosine, sine, 0],
            [0, 0, 0, -sine, cosine, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rotation.T @ local @ rotation


def test_portal_displacement_matches_independent_matrix_assembly() -> None:
    """Compare a fixed-base one-bay portal against an independently assembled 6-DOF solve."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    area = 0.02
    width = 6.0
    height = 4.0
    lateral_force = 20_000.0
    coordinates = ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height))
    connectivity = ((0, 2), (1, 3), (2, 3))
    stiffness = np.zeros((12, 12), dtype=np.float64)
    for i_node, j_node in connectivity:
        indices = (
            3 * i_node,
            3 * i_node + 1,
            3 * i_node + 2,
            3 * j_node,
            3 * j_node + 1,
            3 * j_node + 2,
        )
        element = _frame_element_stiffness(
            elastic_modulus, area, inertia, coordinates[i_node], coordinates[j_node]
        )
        stiffness[np.ix_(indices, indices)] += element
    free = np.arange(6, 12)
    loads = np.zeros(6, dtype=np.float64)
    loads[0] = lateral_force / 2.0
    loads[3] = lateral_force / 2.0
    expected = np.linalg.solve(stiffness[np.ix_(free, free)], loads)

    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, (x_coordinate, y_coordinate) in enumerate(coordinates, start=1):
            ops.node(tag, x_coordinate, y_coordinate)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        for tag, (i_node, j_node) in enumerate(connectivity, start=1):
            ops.element(
                "elasticBeamColumn",
                tag,
                i_node + 1,
                j_node + 1,
                area,
                elastic_modulus,
                inertia,
                1,
            )
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(3, lateral_force / 2.0, 0.0, 0.0)
        ops.load(4, lateral_force / 2.0, 0.0, 0.0)
        _run_linear_static()
        actual = np.array((*ops.nodeDisp(3), *ops.nodeDisp(4)), dtype=np.float64)
        assert actual == pytest.approx(expected, rel=RELATIVE_TOLERANCE, abs=1e-12)
    finally:
        ops.wipe()


def test_two_degree_generalized_eigenproblem_matches_numpy() -> None:
    """Compare OpenSees eigenvalues with an independent two-mass shear-system solution."""

    lower_stiffness = 8.0e6
    upper_stiffness = 5.0e6
    lower_mass = 2_000.0
    upper_mass = 1_500.0
    stiffness = np.array(
        [
            [lower_stiffness + upper_stiffness, -upper_stiffness],
            [-upper_stiffness, upper_stiffness],
        ],
        dtype=np.float64,
    )
    mass = np.diag(np.array([lower_mass, upper_mass], dtype=np.float64))
    expected = np.sort(np.linalg.eigvals(np.linalg.solve(mass, stiffness)).real)
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        ops.node(1, 0.0)
        ops.node(2, 0.0)
        ops.node(3, 0.0)
        ops.fix(1, 1)
        ops.mass(2, lower_mass)
        ops.mass(3, upper_mass)
        ops.uniaxialMaterial("Elastic", 1, lower_stiffness)
        ops.uniaxialMaterial("Elastic", 2, upper_stiffness)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
        ops.element("zeroLength", 2, 2, 3, "-mat", 2, "-dir", 1)
        actual = np.sort(np.array(ops.eigen("-fullGenLapack", 2), dtype=np.float64))
        assert actual == pytest.approx(expected, rel=RELATIVE_TOLERANCE)
    finally:
        ops.wipe()
