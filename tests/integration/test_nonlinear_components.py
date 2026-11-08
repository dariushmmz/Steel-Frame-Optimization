"""OpenSees integration checks for Phase 6 hinges and nonlinear convergence handling."""

from __future__ import annotations

import math
from pathlib import Path

import openseespy.opensees as ops
import pytest

from steel_frame_optimization.analysis.convergence import (
    NonlinearSolveResult,
    NonlinearSolverPolicy,
    run_static_load_path,
)
from steel_frame_optimization.analysis.hinges import (
    ColumnHingeMode,
    RotationalHingeParameters,
    add_hinged_member_2d,
    build_rotational_hinge_parameters,
    create_rotational_hinge_material,
    lower_bound_column_compression_strength_n,
)
from steel_frame_optimization.models.failures import FailureCategory
from steel_frame_optimization.sections.repository import SectionRepository

pytestmark = pytest.mark.integration


def _policy(project_root: Path) -> NonlinearSolverPolicy:
    """Load the frozen Phase 6 nonlinear solver policy."""

    import json

    content = json.loads(
        (project_root / "configs/analysis/phase6_nonlinear_policy.json").read_text(encoding="utf-8")
    )
    return NonlinearSolverPolicy.model_validate(content["solver_policy"])


def _beam_parameters(project_root: Path) -> RotationalHingeParameters:
    """Return representative published-design beam hinge parameters."""

    repository = SectionRepository.from_json(
        project_root / "data/processed/aisc_v14_1_w_shapes.json"
    )
    return build_rotational_hinge_parameters(
        repository.get("W24X55"),
        member_type="beam",
        member_length_m=9.144,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=248_000_000.0,
        expected_yield_strength_pa=339_000_000.0,
    )


def _build_single_hinge(parameters: RotationalHingeParameters, target_moment_nm: float) -> None:
    """Build one zeroLength rotational spring under a unit load-factor target."""

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 0)
    create_rotational_hinge_material(1, parameters)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 3)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, 0.0, target_moment_nm)


def test_monotonic_steel01_hinge_matches_bilinear_closed_form(project_root: Path) -> None:
    """Compare OpenSees rotation below and above yield with an independent bilinear solution."""

    parameters = _beam_parameters(project_root)
    target_moment = 1.5 * parameters.expected_plastic_moment_nm
    _build_single_hinge(parameters, target_moment)
    try:
        result = run_static_load_path(
            target_load_increment=1.0, nominal_steps=16, policy=_policy(project_root)
        )
        expected = parameters.spring_yield_rotation_rad + (
            target_moment - parameters.expected_plastic_moment_nm
        ) / (parameters.spring_hardening_ratio * parameters.spring_initial_stiffness_nm_per_rad)
        assert result.converged
        assert ops.nodeDisp(2, 3) == pytest.approx(expected, rel=1e-8)
    finally:
        ops.wipe()


def test_series_springs_recover_member_stiffness_in_opensees(project_root: Path) -> None:
    """Verify the corrected elastic/spring series response with actual OpenSees elements."""

    parameters = _beam_parameters(project_root)
    elastic_element_stiffness = (
        (parameters.stiffness_multiplier_n + 1.0)
        / parameters.stiffness_multiplier_n
        * parameters.member_stiffness_nm_per_rad
    )
    target_moment = 0.25 * parameters.expected_plastic_moment_nm
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag in (1, 2, 3):
            ops.node(tag, 0.0, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 0)
        ops.fix(3, 1, 1, 0)
        ops.uniaxialMaterial("Elastic", 1, elastic_element_stiffness)
        create_rotational_hinge_material(2, parameters)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 3)
        ops.element("zeroLength", 2, 2, 3, "-mat", 2, "-dir", 3)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(3, 0.0, 0.0, target_moment)
        result = run_static_load_path(
            target_load_increment=1.0, nominal_steps=4, policy=_policy(project_root)
        )
        assert result.converged
        measured = target_moment / float(ops.nodeDisp(3, 3))
        assert measured == pytest.approx(parameters.member_stiffness_nm_per_rad, rel=1e-8)
    finally:
        ops.wipe()


def test_refinement_sequence_is_repeatable_and_step_insensitive(project_root: Path) -> None:
    """Freeze 4/8/16/32-step results and require the same monotonic endpoint."""

    parameters = _beam_parameters(project_root)
    target_moment = 1.2 * parameters.expected_plastic_moment_nm
    rotations: list[float] = []
    for steps in (4, 8, 16, 32):
        _build_single_hinge(parameters, target_moment)
        try:
            result = run_static_load_path(
                target_load_increment=1.0,
                nominal_steps=steps,
                policy=_policy(project_root),
            )
            assert result.converged
            rotations.append(float(ops.nodeDisp(2, 3)))
        finally:
            ops.wipe()
    assert max(rotations) - min(rotations) <= 1e-12 * max(abs(value) for value in rotations)


def _run_controlled_failure(policy: NonlinearSolverPolicy) -> NonlinearSolveResult:
    """Create a zero-hardening spring loaded beyond its finite moment capacity."""

    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, 0.0, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 0)
        ops.uniaxialMaterial("ElasticPP", 1, 1_000.0, 0.01)
        ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 3)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 0.0, 0.0, 20.0)
        return run_static_load_path(target_load_increment=1.0, nominal_steps=1, policy=policy)
    finally:
        ops.wipe()


def test_controlled_nonconvergence_is_typed_and_repeatable(project_root: Path) -> None:
    """Exhaust the bounded ladder twice and require deterministic explicit failure."""

    policy = _policy(project_root)
    first = _run_controlled_failure(policy)
    second = _run_controlled_failure(policy)
    assert first == second
    assert first.converged is False
    assert first.failure is not None
    assert first.failure.category is FailureCategory.NUMERICAL_NONCONVERGENCE
    assert first.achieved_load_factor < first.target_load_factor
    assert {attempt.algorithm for attempt in first.attempts} == {
        "Newton",
        "NewtonLineSearch",
        "KrylovNewton",
    }
    assert {attempt.refinement_divisor for attempt in first.attempts} == {1, 2, 4, 8, 16}


def test_reusable_hinged_member_builder_creates_expected_topology(project_root: Path) -> None:
    """Build one two-ended member without invoking a full-frame pushover."""

    parameters = _beam_parameters(project_root)
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, parameters.member_length_m, 0.0)
        ops.geomTransf("PDelta", 1)
        create_rotational_hinge_material(1, parameters)
        create_rotational_hinge_material(2, parameters)
        tags = add_hinged_member_2d(
            outer_i_node_tag=1,
            outer_j_node_tag=2,
            i_internal_node_tag=3,
            j_internal_node_tag=4,
            i_spring_element_tag=1,
            j_spring_element_tag=2,
            member_element_tag=3,
            i_material_tag=1,
            j_material_tag=2,
            area_m2=0.01,
            elastic_modulus_pa=parameters.elastic_modulus_pa,
            adjusted_inertia_m4=parameters.adjusted_member_inertia_m4,
            transformation_tag=1,
        )
        assert tags.member_element_tag == 3
        assert ops.getNodeTags() == [1, 2, 3, 4]
        assert ops.getEleTags() == [1, 2, 3]
        assert math.isclose(float(ops.nodeCoord(3, 1)), 0.0)
        assert math.isclose(float(ops.nodeCoord(4, 1)), parameters.member_length_m)
    finally:
        ops.wipe()


def test_force_controlled_column_spring_remains_elastic(project_root: Path) -> None:
    """Verify that a P/Pcl=0.5 column does not acquire unapproved plastic rotation."""

    repository = SectionRepository.from_json(
        project_root / "data/processed/aisc_v14_1_w_shapes.json"
    )
    section = repository.get("W16X57")
    length = 3.9624
    pcl = lower_bound_column_compression_strength_n(
        section,
        effective_length_factor=1.0,
        length_m=length,
        nominal_yield_strength_pa=345_000_000.0,
        elastic_modulus_pa=200_000_000_000.0,
    )
    parameters = build_rotational_hinge_parameters(
        section,
        member_type="column",
        member_length_m=length,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=345_000_000.0,
        expected_yield_strength_pa=397_000_000.0,
        column_mode=ColumnHingeMode.FINAL_AXIAL_REDUCED,
        compression_force_n=0.5 * pcl,
    )
    target_moment = 2.0 * parameters.expected_plastic_moment_nm
    _build_single_hinge(parameters, target_moment)
    try:
        result = run_static_load_path(
            target_load_increment=1.0, nominal_steps=4, policy=_policy(project_root)
        )
        assert result.converged
        assert ops.nodeDisp(2, 3) == pytest.approx(
            target_moment / parameters.spring_initial_stiffness_nm_per_rad,
            rel=1e-8,
        )
    finally:
        ops.wipe()
