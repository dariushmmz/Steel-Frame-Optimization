"""Layer 1/2 tests for spectrum formulas and the Phase 4 elastic OpenSees kernel."""

from __future__ import annotations

import inspect
import itertools
import json
import math
from pathlib import Path

import openseespy.opensees as ops
import pytest

from steel_frame_optimization.analysis.elastic import (
    LoadFactors,
    run_modal_analysis,
    run_static_analysis,
)
from steel_frame_optimization.analysis.modal import (
    distribute_story_forces_n,
    elastic_base_shear_n,
    periods_from_eigenvalues,
    spectral_acceleration_g,
)
from steel_frame_optimization.models.frame import FrameDefinition, GeneratedFrame, generate_frame
from steel_frame_optimization.sections.records import SectionRecord
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.analytical

RELATIVE_TOLERANCE = 1e-6
EQUILIBRIUM_TOLERANCE = 1e-8
EXACT_BEAM_COLUMN_RELATIVE_TOLERANCE = 1e-4
REFINEMENT_SEQUENCE = (1, 2, 4, 8, 16, 32)
MINIMUM_SUPERPOSITION_ERROR = 0.05


@pytest.fixture
def analysis_inputs(
    project_root: Path,
) -> tuple[GeneratedFrame, dict[str, SectionRecord], dict[str, object]]:
    """Load the source-traceable frame, published Case 2 GWO design, and spectrum config."""

    definition = FrameDefinition.from_json(
        project_root / "configs" / "frames" / "three_story_frame.json"
    )
    generated = generate_frame(definition)
    repository = SectionRepository.from_json(
        project_root / "data" / "processed" / "aisc_v14_1_w_shapes.json"
    )
    case = json.loads(
        (project_root / "configs" / "cases" / "three_story_case2_gwo.json").read_text(
            encoding="utf-8"
        )
    )
    sections = {
        group_id: repository.get(section_name)
        for group_id, section_name in case["sections_by_group"].items()
    }
    return generated, sections, case


def _case_story_forces(
    generated: GeneratedFrame, case: dict[str, object], fundamental_period_s: float
) -> tuple[float, tuple[float, ...]]:
    """Compute the configured Case 2 base shear and parabolic floor forces."""

    spectrum = case["spectrum"]
    assert isinstance(spectrum, dict)
    acceleration = spectral_acceleration_g(
        fundamental_period_s,
        short_period_g=float(spectrum["short_period_g"]),
        one_second_g=float(spectrum["one_second_g"]),
        fa=float(spectrum["fa"]),
        fv=float(spectrum["fv"]),
    )
    weights = [floor.seismic_weight_n for floor in generated.definition.floors]
    base_shear = elastic_base_shear_n(sum(weights), acceleration, float(spectrum["r_factor"]))
    elevation = 0.0
    elevations = []
    for height in generated.definition.story_heights_m:
        elevation += height
        elevations.append(elevation)
    forces = distribute_story_forces_n(
        base_shear,
        weights,
        elevations,
        exponent=float(spectrum["distribution_exponent"]),
    )
    return base_shear, forces


def test_spectrum_boundaries_and_force_distribution() -> None:
    """Check both spectrum joins and exact base-shear closure before observing frame response."""

    transition = 1.94 * 0.23 / (1.4 * 0.5)
    assert spectral_acceleration_g(
        0.0, short_period_g=0.5, one_second_g=0.23, fa=1.4, fv=1.94
    ) == pytest.approx(0.28)
    for period in (0.2 * transition, transition):
        assert spectral_acceleration_g(
            period, short_period_g=0.5, one_second_g=0.23, fa=1.4, fv=1.94
        ) == pytest.approx(0.7)
    base_shear = elastic_base_shear_n(14_447_000.0, 0.7, 8.0)
    forces = distribute_story_forces_n(
        base_shear, (4_688_000.0, 4_688_000.0, 5_071_000.0), (3.9624, 7.9248, 11.8872)
    )
    assert sum(forces) == base_shear
    assert forces == tuple(sorted(forces))


def test_spectrum_and_modal_converters_reject_invalid_values() -> None:
    """Reject negative periods, mismatched story arrays, and nonpositive eigenvalues."""

    with pytest.raises(ValueError, match="period"):
        spectral_acceleration_g(-1.0, short_period_g=0.5, one_second_g=0.23, fa=1.4, fv=1.94)
    with pytest.raises(ValueError, match="equal"):
        distribute_story_forces_n(1.0, (1.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="positive"):
        periods_from_eigenvalues((0.0,))


def test_single_degree_of_freedom_period_matches_closed_form() -> None:
    """Compare the OpenSees generalized eigenvalue with T=2*pi*sqrt(m/k)."""

    mass_kg = 2_000.0
    stiffness_n_per_m = 8_000_000.0
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        ops.node(1, 0.0)
        ops.node(2, 0.0)
        ops.fix(1, 1)
        ops.mass(2, mass_kg)
        ops.uniaxialMaterial("Elastic", 1, stiffness_n_per_m)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
        eigenvalue = float(ops.eigen("-fullGenLapack", 1)[0])
        actual = periods_from_eigenvalues((eigenvalue,))[0]
        expected = 2.0 * math.pi * math.sqrt(mass_kg / stiffness_n_per_m)
        assert actual == pytest.approx(expected, rel=RELATIVE_TOLERANCE)
    finally:
        ops.wipe()


def test_cantilever_and_fixed_beam_displacements_match_closed_form() -> None:
    """Check a cantilever tip and fixed-fixed midspan displacement independently."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    area = 0.02
    length = 4.0
    force = 10_000.0
    line_load = 5_000.0

    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, length, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        ops.element("elasticBeamColumn", 1, 1, 2, area, elastic_modulus, inertia, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 0.0, -force, 0.0)
        _run_one_static_step()
        assert float(ops.nodeDisp(2, 2)) == pytest.approx(
            -force * length**3 / (3.0 * elastic_modulus * inertia), rel=RELATIVE_TOLERANCE
        )
    finally:
        ops.wipe()

    ops.model("basic", "-ndm", 2, "-ndf", 3)
    try:
        ops.node(1, 0.0, 0.0)
        ops.node(2, length / 2.0, 0.0)
        ops.node(3, length, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(3, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        ops.element("elasticBeamColumn", 1, 1, 2, area, elastic_modulus, inertia, 1)
        ops.element("elasticBeamColumn", 2, 2, 3, area, elastic_modulus, inertia, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.eleLoad("-ele", 1, 2, "-type", "-beamUniform", -line_load)
        _run_one_static_step()
        assert float(ops.nodeDisp(2, 2)) == pytest.approx(
            -line_load * length**4 / (384.0 * elastic_modulus * inertia),
            rel=RELATIVE_TOLERANCE,
        )
    finally:
        ops.wipe()


def _run_one_static_step() -> None:
    """Run the shared deterministic solver sequence used by closed-form fixtures."""

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0


def _cantilever_tip_displacement(
    *,
    element_count: int,
    compression_force: float,
    horizontal_force: float,
    staged: bool,
) -> float:
    """Solve the fixed P-Delta fixture; one-step mode exists only as a test negative control."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    area = 0.02
    length = 4.0
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for index in range(element_count + 1):
            ops.node(index + 1, 0.0, length * index / element_count)
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("PDelta", 1)
        for index in range(element_count):
            ops.element(
                "elasticBeamColumn",
                index + 1,
                index + 1,
                index + 2,
                area,
                elastic_modulus,
                inertia,
                1,
            )
        tip_tag = element_count + 1
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        if staged:
            ops.load(tip_tag, 0.0, -compression_force, 0.0)
            _run_newton_static_step()
            ops.loadConst("-time", 0.0)
            ops.wipeAnalysis()
            ops.timeSeries("Linear", 2)
            ops.pattern("Plain", 2, 2)
            ops.load(tip_tag, horizontal_force, 0.0, 0.0)
            _run_newton_static_step()
        else:
            ops.load(tip_tag, horizontal_force, -compression_force, 0.0)
            _run_one_static_step()
        return float(ops.nodeDisp(tip_tag, 1))
    finally:
        ops.wipe()


def _run_newton_static_step() -> None:
    """Execute one tangent-reforming load increment for the P-Delta fixtures."""

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 50)
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Newton")
    ops.analysis("Static")
    assert ops.analyze(1) == 0


def test_pdelta_one_element_staged_reference_and_negative_control() -> None:
    """Keep the unchanged P-Delta-only reference and prove the old one-step defect."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    length = 4.0
    horizontal_force = 10_000.0
    compression_force = 200_000.0
    first_order = horizontal_force * length**3 / (3.0 * elastic_modulus * inertia)
    p_delta_only = horizontal_force / (
        3.0 * elastic_modulus * inertia / length**3 - compression_force / length
    )
    corrected = _cantilever_tip_displacement(
        element_count=1,
        compression_force=compression_force,
        horizontal_force=horizontal_force,
        staged=True,
    )
    old_one_step = _cantilever_tip_displacement(
        element_count=1,
        compression_force=compression_force,
        horizontal_force=horizontal_force,
        staged=False,
    )
    assert corrected == pytest.approx(p_delta_only, rel=RELATIVE_TOLERANCE)
    assert corrected / first_order == pytest.approx(1.0714285714285714, rel=RELATIVE_TOLERANCE)
    assert old_one_step == pytest.approx(first_order, rel=RELATIVE_TOLERANCE)
    assert "staged" not in inspect.signature(run_static_analysis).parameters
    assert "one_step" not in inspect.signature(run_static_analysis).parameters


def test_pdelta_mesh_refines_to_exact_beam_column_solution() -> None:
    """Refine the fixed sequence to the exact beam-column solution without changing its gate."""

    elastic_modulus = 200.0e9
    inertia = 8.0e-5
    length = 4.0
    horizontal_force = 10_000.0
    compression_force = 200_000.0
    wave_number = math.sqrt(compression_force / (elastic_modulus * inertia))
    exact = (
        horizontal_force
        / compression_force
        * (math.tan(wave_number * length) / wave_number - length)
    )
    displacements = tuple(
        _cantilever_tip_displacement(
            element_count=element_count,
            compression_force=compression_force,
            horizontal_force=horizontal_force,
            staged=True,
        )
        for element_count in REFINEMENT_SEQUENCE
    )
    errors = tuple(abs(value - exact) / exact for value in displacements)
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    assert displacements[-1] == pytest.approx(exact, rel=EXACT_BEAM_COLUMN_RELATIVE_TOLERANCE)


def test_pdelta_direct_combination_is_not_superposition() -> None:
    """Show that separately summed gravity/lateral results differ from direct staged response."""

    direct = _cantilever_tip_displacement(
        element_count=1,
        compression_force=200_000.0,
        horizontal_force=10_000.0,
        staged=True,
    )
    lateral_only = _cantilever_tip_displacement(
        element_count=1,
        compression_force=0.0,
        horizontal_force=10_000.0,
        staged=True,
    )
    gravity_only = _cantilever_tip_displacement(
        element_count=1,
        compression_force=200_000.0,
        horizontal_force=0.0,
        staged=True,
    )
    superposed = lateral_only + gravity_only
    assert abs(direct - superposed) / direct > MINIMUM_SUPERPOSITION_ERROR


def test_published_frame_modal_static_equilibrium_and_repeatability(
    analysis_inputs: tuple[GeneratedFrame, dict[str, SectionRecord], dict[str, object]],
) -> None:
    """Run the published design twice and verify deterministic response and global equilibrium."""

    generated, sections, case = analysis_inputs
    first_modal = run_modal_analysis(generated, sections)
    second_modal = run_modal_analysis(generated, sections)
    assert first_modal == second_modal
    assert len(first_modal.periods_s) == 3
    assert first_modal.periods_s == tuple(sorted(first_modal.periods_s, reverse=True))
    assert all(mode[-1] == pytest.approx(1.0) for mode in first_modal.normalized_story_modes)

    base_shear, forces = _case_story_forces(generated, case, first_modal.periods_s[0])
    assert sum(forces) == pytest.approx(base_shear)
    factors = LoadFactors(dead=1.2, live=1.0, seismic=1.0)
    first = run_static_analysis(generated, sections, factors, forces, p_delta=True)
    second = run_static_analysis(generated, sections, factors, forces, p_delta=True)
    assert first == second
    assert first.equilibrium_relative_residual <= EQUILIBRIUM_TOLERANCE
    assert first.model_node_count == 32
    assert first.model_element_count == 39
    assert first.load_stages == ("gravity_converged_and_held", "lateral_converged")
    assert first.combination_evaluated_directly is True
    assert len(first.element_forces) == 39
    assert all(len(force.local_end_forces_n_nm) == 6 for force in first.element_forces)

    negative = run_static_analysis(
        generated, sections, factors, tuple(-force for force in forces), p_delta=True
    )
    assert negative.roof_displacement_m == pytest.approx(
        -first.roof_displacement_m, rel=RELATIVE_TOLERANCE
    )


def test_invalid_section_map_and_story_force_count_fail_clearly(
    analysis_inputs: tuple[GeneratedFrame, dict[str, SectionRecord], dict[str, object]],
) -> None:
    """Reject incomplete group maps and lateral-force vectors before returning a result."""

    generated, sections, _ = analysis_inputs
    incomplete = dict(sections)
    del incomplete["5"]
    with pytest.raises(ValueError, match="every design group"):
        run_modal_analysis(generated, incomplete)
    with pytest.raises(ValueError, match="per story"):
        run_static_analysis(
            generated,
            sections,
            LoadFactors(dead=0.0, live=0.0, seismic=1.0),
            (1.0,),
            p_delta=False,
        )
