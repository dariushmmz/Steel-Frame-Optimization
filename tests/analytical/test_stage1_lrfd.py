"""Independent equation and boundary checks for full Stage 1 LRFD calculations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from steel_frame_optimization.constraints.lrfd import (
    FlexuralStrength,
    beam_column_interaction_utilization,
    classify_axial_compression,
    classify_major_axis_flexure,
    effective_length_factor_unbraced,
    moment_gradient_factor,
    nominal_compression_strength_with_local_buckling,
    nominal_major_axis_flexural_strength,
)
from steel_frame_optimization.constraints.serviceability import (
    chord_relative_midspan_deflection_m,
    deflection_utilization,
    drift_utilization,
)
from steel_frame_optimization.models.frame import FrameDefinition, generate_frame
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.validation.phase5 import calculate_effective_lengths

pytestmark = pytest.mark.analytical


@pytest.fixture
def repository(project_root: Path) -> SectionRepository:
    """Load the frozen 273-row section baseline."""

    return SectionRepository.from_json(project_root / "data/processed/aisc_v14_1_w_shapes.json")


def test_effective_length_factor_matches_printed_equation() -> None:
    """Compare REF-A Eq. (26) with a direct hand calculation."""

    g_a = 1.0
    g_b = 2.5
    expected = math.sqrt((1.6 * g_a * g_b + 1.4 * (g_a + g_b) + 0.64) / (g_a + g_b + 7.5))
    assert effective_length_factor_unbraced(g_a=g_a, g_b=g_b) == pytest.approx(expected)


def test_compactness_classification_and_noncompact_flange_strength(
    repository: SectionRepository,
) -> None:
    """Check the known W6X15 noncompact flange and AISC F3-1 interpolation."""

    section = repository.get("W6X15")
    elastic_modulus = 200.0e9
    yield_strength = 248.0e6
    classification = classify_major_axis_flexure(
        section, elastic_modulus_pa=elastic_modulus, yield_strength_pa=yield_strength
    )
    assert classification.flange_class == "noncompact"
    assert classification.web_class == "compact"
    result = nominal_major_axis_flexural_strength(
        section,
        elastic_modulus_pa=elastic_modulus,
        yield_strength_pa=yield_strength,
        resistance_factor=0.9,
        unbraced_length_m=0.0,
        cb=1.0,
        fully_braced=True,
    )
    plastic = yield_strength * section.zx_m3
    expected_local = plastic - (plastic - 0.7 * yield_strength * section.sx_m3) * (
        (section.flange_slenderness - classification.flange_lambda_p)
        / (classification.flange_lambda_r - classification.flange_lambda_p)
    )
    assert result.nominal_strength_nm == pytest.approx(expected_local, rel=1e-12)
    assert result.controlling_limit_state == "noncompact_flange_local_buckling"


def test_flexural_compactness_boundaries_are_inclusive(repository: SectionRepository) -> None:
    """Exercise exact compact/noncompact/slender flange and web dispatch boundaries."""

    section = repository.get("W14X730")
    base = classify_major_axis_flexure(
        section, elastic_modulus_pa=200.0e9, yield_strength_pa=345.0e6
    )
    at_compact = section.model_copy(
        update={
            "flange_slenderness": base.flange_lambda_p,
            "web_slenderness": base.web_lambda_p,
        }
    )
    at_noncompact = section.model_copy(
        update={
            "flange_slenderness": base.flange_lambda_r,
            "web_slenderness": base.web_lambda_r,
        }
    )
    slender = section.model_copy(
        update={
            "flange_slenderness": base.flange_lambda_r * (1.0 + 1e-9),
            "web_slenderness": base.web_lambda_r * (1.0 + 1e-9),
        }
    )
    assert (
        classify_major_axis_flexure(
            at_compact, elastic_modulus_pa=200.0e9, yield_strength_pa=345.0e6
        ).flange_class
        == "compact"
    )
    noncompact_result = classify_major_axis_flexure(
        at_noncompact, elastic_modulus_pa=200.0e9, yield_strength_pa=345.0e6
    )
    assert (noncompact_result.flange_class, noncompact_result.web_class) == (
        "noncompact",
        "noncompact",
    )
    slender_result = classify_major_axis_flexure(
        slender, elastic_modulus_pa=200.0e9, yield_strength_pa=345.0e6
    )
    assert (slender_result.flange_class, slender_result.web_class) == ("slender", "slender")


def test_lateral_torsional_buckling_uses_all_three_length_regions(
    repository: SectionRepository,
) -> None:
    """Exercise yielding, inelastic LTB, and elastic LTB without changing the section."""

    section = repository.get("W16X57")

    def strength(unbraced_length_m: float) -> FlexuralStrength:
        return nominal_major_axis_flexural_strength(
            section,
            elastic_modulus_pa=200.0e9,
            yield_strength_pa=345.0e6,
            resistance_factor=0.9,
            cb=1.0,
            fully_braced=False,
            unbraced_length_m=unbraced_length_m,
        )

    short = strength(0.0)
    middle = nominal_major_axis_flexural_strength(
        section,
        elastic_modulus_pa=200.0e9,
        yield_strength_pa=345.0e6,
        resistance_factor=0.9,
        cb=1.0,
        fully_braced=False,
        unbraced_length_m=(short.lp_m + short.lr_m) / 2.0,
    )
    long = strength(1.2 * short.lr_m)
    assert short.controlling_limit_state == "yielding"
    assert middle.controlling_limit_state == "inelastic_ltb"
    assert long.controlling_limit_state == "elastic_ltb"
    assert short.nominal_strength_nm > middle.nominal_strength_nm > long.nominal_strength_nm


def test_interaction_branch_boundary_is_in_high_axial_branch() -> None:
    """Freeze REF-A Eq. (11)'s strict-low and inclusive-high split at 0.2."""

    nominal_p = 1_000_000.0
    nominal_m = 500_000.0
    phi = 0.9
    just_low = beam_column_interaction_utilization(
        compression_demand_n=0.2 * phi * nominal_p - 1.0,
        moment_demand_nm=0.1 * phi * nominal_m,
        nominal_compression_strength_n_value=nominal_p,
        nominal_flexural_strength_nm=nominal_m,
        compression_resistance_factor=phi,
        flexural_resistance_factor=phi,
    )
    at_boundary = beam_column_interaction_utilization(
        compression_demand_n=0.2 * phi * nominal_p,
        moment_demand_nm=0.1 * phi * nominal_m,
        nominal_compression_strength_n_value=nominal_p,
        nominal_flexural_strength_nm=nominal_m,
        compression_resistance_factor=phi,
        flexural_resistance_factor=phi,
    )
    assert just_low == pytest.approx((0.2 - 1.0 / (phi * nominal_p)) / 2.0 + 0.1)
    assert at_boundary == pytest.approx(0.2 + 8.0 / 9.0 * 0.1)


def test_moment_gradient_factor_known_linear_diagrams() -> None:
    """Check AISC F1-1 uniform and reverse-curvature linear moment diagrams."""

    uniform = moment_gradient_factor(
        moment_a_nm=10.0,
        moment_b_nm=10.0,
        moment_c_nm=10.0,
        maximum_moment_nm=10.0,
    )
    reverse_curvature = moment_gradient_factor(
        moment_a_nm=5.0,
        moment_b_nm=0.0,
        moment_c_nm=-5.0,
        maximum_moment_nm=10.0,
    )
    assert uniform == pytest.approx(1.0)
    assert reverse_curvature == pytest.approx(12.5 / 5.5)


def test_candidate_effective_length_matches_independent_joint_sum(
    project_root: Path, repository: SectionRepository
) -> None:
    """Hand-check the bottom exterior column's candidate-specific G and K."""

    definition = FrameDefinition.from_json(project_root / "configs/frames/three_story_frame.json")
    generated = generate_frame(definition)
    case = json.loads(
        (project_root / "configs/cases/three_story_case2_gwo.json").read_text(encoding="utf-8")
    )
    sections = {group: repository.get(name) for group, name in case["sections_by_group"].items()}
    actual = next(
        item
        for item in calculate_effective_lengths(generated, sections)
        if item.member_id == "C-S1-C0"
    )
    column = sections["1"]
    beam = sections["3"]
    story_height = definition.story_heights_m[0]
    bay_width = definition.bay_widths_m[0]
    expected_top_g = 2.0 * column.ix_m4 / story_height / (beam.ix_m4 / bay_width)
    expected_k = effective_length_factor_unbraced(g_a=1.0, g_b=expected_top_g)
    assert actual.g_i == pytest.approx(1.0)
    assert actual.g_j == pytest.approx(expected_top_g, rel=1e-12)
    assert actual.effective_length_factor == pytest.approx(expected_k, rel=1e-12)


def test_slender_axial_web_uses_e7_effective_area(repository: SectionRepository) -> None:
    """Check that a slender W10X12 web reduces area while a stocky W14X730 does not."""

    slender = repository.get("W10X12")
    stocky = repository.get("W14X730")
    slender_class = classify_axial_compression(
        slender, elastic_modulus_pa=200.0e9, yield_strength_pa=345.0e6
    )
    assert slender_class.web_class == "slender"
    reduced = nominal_compression_strength_with_local_buckling(
        slender,
        effective_length_factor=1.0,
        length_m=3.9624,
        radius_of_gyration_m=slender.rx_m,
        yield_strength_pa=345.0e6,
        elastic_modulus_pa=200.0e9,
    )
    unreduced = nominal_compression_strength_with_local_buckling(
        stocky,
        effective_length_factor=1.0,
        length_m=3.9624,
        radius_of_gyration_m=stocky.rx_m,
        yield_strength_pa=345.0e6,
        elastic_modulus_pa=200.0e9,
    )
    assert reduced.effective_area_m2 < reduced.gross_area_m2
    assert reduced.controlling_limit_state == "local_and_flexural_buckling"
    assert unreduced.effective_area_m2 == unreduced.gross_area_m2
    assert unreduced.controlling_limit_state == "flexural_buckling"


def test_serviceability_helpers_use_displaced_chord_and_absolute_drift() -> None:
    """Check chord-relative deflection and both signs of drift against hand values."""

    deflection = chord_relative_midspan_deflection_m(
        midpoint_vertical_m=-0.012,
        i_end_vertical_m=-0.002,
        j_end_vertical_m=-0.004,
    )
    assert deflection == pytest.approx(0.009)
    assert deflection_utilization(
        deflection_m=deflection, span_m=9.0, denominator=360.0
    ) == pytest.approx(0.36)
    assert drift_utilization(drift_ratio=-0.005, allowable_ratio=0.025) == pytest.approx(0.2)
