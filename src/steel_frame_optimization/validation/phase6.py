"""Generate deterministic Phase 6 component and nonlinear-solver evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import openseespy.opensees as ops
from pydantic import Field

from steel_frame_optimization.analysis.convergence import (
    NonlinearSolveResult,
    NonlinearSolverPolicy,
    run_static_load_path,
)
from steel_frame_optimization.analysis.hinges import (
    ColumnHingeMode,
    RotationalHingeParameters,
    build_rotational_hinge_parameters,
    corrected_spring_hardening_ratio,
    create_rotational_hinge_material,
    lower_bound_column_compression_strength_n,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import canonical_json_dumps


class MonotonicPoint(StrictConfigModel):
    """Compare one OpenSees hinge endpoint with the closed-form Steel01 response."""

    load_ratio_m_over_mp: float = Field(gt=0.0, allow_inf_nan=False)
    nominal_steps: int = Field(gt=0)
    moment_nm: float = Field(gt=0.0, allow_inf_nan=False)
    calculated_rotation_rad: float = Field(allow_inf_nan=False)
    reference_rotation_rad: float = Field(allow_inf_nan=False)
    relative_error: float = Field(ge=0.0, allow_inf_nan=False)
    solver_result: NonlinearSolveResult


class Phase6Artifact(StrictConfigModel):
    """Persist the complete deterministic Phase 6 component evidence set."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1)
    beam_section: str = Field(min_length=1)
    column_section: str = Field(min_length=1)
    beam_parameters: RotationalHingeParameters
    preliminary_column_parameters: RotationalHingeParameters
    final_deformation_column_parameters: RotationalHingeParameters
    final_force_column_parameters: RotationalHingeParameters
    series_initial_stiffness_nm_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    series_postyield_stiffness_nm_per_rad: float = Field(gt=0.0, allow_inf_nan=False)
    monotonic_points: tuple[MonotonicPoint, ...]
    refinement_endpoint_rotations_rad: dict[str, float]
    maximum_refinement_spread_rad: float = Field(ge=0.0, allow_inf_nan=False)
    controlled_failure: NonlinearSolveResult
    sources: tuple[str, ...]
    limitations: tuple[str, ...]


def _load_policy(path: Path) -> tuple[dict[str, object], NonlinearSolverPolicy, tuple[int, ...]]:
    """Load hinge metadata, strict solver policy, and fixed verification refinements."""

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Phase 6 policy root must be an object")
    hinge = content.get("hinge_policy")
    solver = content.get("solver_policy")
    verification = content.get("verification")
    if (
        not isinstance(hinge, dict)
        or not isinstance(solver, dict)
        or not isinstance(verification, dict)
    ):
        raise ValueError("Phase 6 policy requires hinge, solver, and verification objects")
    raw_steps = verification.get("refinement_step_counts")
    if not isinstance(raw_steps, list) or not all(isinstance(value, int) for value in raw_steps):
        raise ValueError("refinement_step_counts must be a list of integers")
    return hinge, NonlinearSolverPolicy.model_validate(solver), tuple(raw_steps)


def _required_float(values: dict[str, object], key: str) -> float:
    """Read one required finite numeric policy value without accepting booleans."""

    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _build_single_hinge(parameters: RotationalHingeParameters, target_moment_nm: float) -> None:
    """Build one isolated OpenSees rotational spring with a unit-factor moment pattern."""

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


def _closed_form_rotation(parameters: RotationalHingeParameters, moment_nm: float) -> float:
    """Return the independent monotonic Steel01 rotation for positive moment."""

    if moment_nm <= parameters.expected_plastic_moment_nm:
        return moment_nm / parameters.spring_initial_stiffness_nm_per_rad
    return parameters.spring_yield_rotation_rad + (
        moment_nm - parameters.expected_plastic_moment_nm
    ) / (parameters.spring_hardening_ratio * parameters.spring_initial_stiffness_nm_per_rad)


def _run_monotonic_point(
    parameters: RotationalHingeParameters,
    *,
    ratio: float,
    nominal_steps: int,
    solver_policy: NonlinearSolverPolicy,
) -> MonotonicPoint:
    """Run and compare one isolated monotonic hinge endpoint."""

    moment = ratio * parameters.expected_plastic_moment_nm
    _build_single_hinge(parameters, moment)
    try:
        result = run_static_load_path(
            target_load_increment=1.0,
            nominal_steps=nominal_steps,
            policy=solver_policy,
        )
        if not result.converged:
            raise RuntimeError("verification hinge failed to converge")
        actual = float(ops.nodeDisp(2, 3))
    finally:
        ops.wipe()
    reference = _closed_form_rotation(parameters, moment)
    return MonotonicPoint(
        load_ratio_m_over_mp=ratio,
        nominal_steps=nominal_steps,
        moment_nm=moment,
        calculated_rotation_rad=actual,
        reference_rotation_rad=reference,
        relative_error=abs(actual - reference) / max(abs(reference), 1e-30),
        solver_result=result,
    )


def _run_controlled_failure(policy: NonlinearSolverPolicy) -> NonlinearSolveResult:
    """Exhaust the solver ladder on a load above a zero-hardening spring's capacity."""

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


def build_phase6_artifact(*, sections_path: Path, policy_path: Path) -> Phase6Artifact:
    """Build representative beam/column hinges and deterministic numerical evidence."""

    repository = SectionRepository.from_json(sections_path)
    hinge_policy, solver_policy, refinement_steps = _load_policy(policy_path)
    n = _required_float(hinge_policy, "stiffness_multiplier_n")
    hardening = _required_float(hinge_policy, "member_hardening_ratio")
    beam_section = repository.get("W24X55")
    column_section = repository.get("W16X57")
    beam = build_rotational_hinge_parameters(
        beam_section,
        member_type="beam",
        member_length_m=9.144,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=248_000_000.0,
        expected_yield_strength_pa=339_000_000.0,
        stiffness_multiplier_n=n,
        member_hardening_ratio=hardening,
    )
    column_length = 3.9624
    pcl = lower_bound_column_compression_strength_n(
        column_section,
        effective_length_factor=1.0,
        length_m=column_length,
        nominal_yield_strength_pa=345_000_000.0,
        elastic_modulus_pa=200_000_000_000.0,
    )
    preliminary = build_rotational_hinge_parameters(
        column_section,
        member_type="column",
        member_length_m=column_length,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=345_000_000.0,
        expected_yield_strength_pa=397_000_000.0,
        stiffness_multiplier_n=n,
        member_hardening_ratio=hardening,
        column_mode=ColumnHingeMode.PRELIMINARY_UNREDUCED,
        compression_force_n=0.35 * pcl,
    )
    deformation = build_rotational_hinge_parameters(
        column_section,
        member_type="column",
        member_length_m=column_length,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=345_000_000.0,
        expected_yield_strength_pa=397_000_000.0,
        stiffness_multiplier_n=n,
        member_hardening_ratio=hardening,
        column_mode=ColumnHingeMode.FINAL_AXIAL_REDUCED,
        compression_force_n=0.35 * pcl,
    )
    force = build_rotational_hinge_parameters(
        column_section,
        member_type="column",
        member_length_m=column_length,
        elastic_modulus_pa=200_000_000_000.0,
        nominal_yield_strength_pa=345_000_000.0,
        expected_yield_strength_pa=397_000_000.0,
        stiffness_multiplier_n=n,
        member_hardening_ratio=hardening,
        column_mode=ColumnHingeMode.FINAL_AXIAL_REDUCED,
        compression_force_n=0.5 * pcl,
    )
    monotonic = tuple(
        _run_monotonic_point(beam, ratio=ratio, nominal_steps=16, solver_policy=solver_policy)
        for ratio in (0.5, 1.0, 1.5)
    )
    refinements = {
        str(steps): _run_monotonic_point(
            beam, ratio=1.2, nominal_steps=steps, solver_policy=solver_policy
        ).calculated_rotation_rad
        for steps in refinement_steps
    }
    elastic_element_stiffness = (n + 1.0) / n * beam.member_stiffness_nm_per_rad
    series_initial = 1.0 / (
        1.0 / elastic_element_stiffness + 1.0 / beam.spring_initial_stiffness_nm_per_rad
    )
    spring_hardening = corrected_spring_hardening_ratio(
        member_hardening_ratio=hardening, stiffness_multiplier_n=n
    )
    series_postyield = 1.0 / (
        1.0 / elastic_element_stiffness
        + 1.0 / (spring_hardening * beam.spring_initial_stiffness_nm_per_rad)
    )
    return Phase6Artifact(
        dataset_id=repository.dataset.dataset_id,
        beam_section=beam_section.name,
        column_section=column_section.name,
        beam_parameters=beam,
        preliminary_column_parameters=preliminary,
        final_deformation_column_parameters=deformation,
        final_force_column_parameters=force,
        series_initial_stiffness_nm_per_rad=series_initial,
        series_postyield_stiffness_nm_per_rad=series_postyield,
        monotonic_points=monotonic,
        refinement_endpoint_rotations_rad=refinements,
        maximum_refinement_spread_rad=max(refinements.values()) - min(refinements.values()),
        controlled_failure=_run_controlled_failure(solver_policy),
        sources=(
            "REF-A PDF pp. 2-3 / journal pp. 78-79, Eqs. 1-10",
            "FEMA 356 printed pp. 5-13 to 5-15 and Table 5-6 pp. 5-40 to 5-43",
            "ASM-013 through ASM-015 and ASM-030 through ASM-031",
        ),
        limitations=(
            "Steel01 is the paper/REF-B bilinear hinge; FEMA a/b/c are retained but "
            "post-cap degradation is not modeled",
            "force-controlled columns remain elastic in the model and require the "
            "Phase 7 force check",
            "component evidence is not a full-frame pushover or professional engineering "
            "certification",
        ),
    )


def _parse_args() -> argparse.Namespace:
    """Parse explicit Phase 6 evidence input and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write canonical Phase 6 component evidence and a concise result summary."""

    args = _parse_args()
    artifact = build_phase6_artifact(sections_path=args.sections, policy_path=args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_stream:
        output_stream.write(canonical_json_dumps(artifact))
    maximum_error = max(point.relative_error for point in artifact.monotonic_points)
    print(
        f"Phase 6 points={len(artifact.monotonic_points)}, "
        f"max_relative_error={maximum_error:.3e}, "
        f"refinement_spread={artifact.maximum_refinement_spread_rad:.3e}, "
        f"controlled_failure={not artifact.controlled_failure.converged}"
    )


if __name__ == "__main__":
    main()
