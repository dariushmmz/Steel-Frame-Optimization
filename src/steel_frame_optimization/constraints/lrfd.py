"""Provide source-traceable LRFD strength and compactness calculations.

Preliminary-filter functions and full Stage 1 functions share elementary equations here, while the
calling evaluator keeps their policies and results separate.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.sections.records import SectionRecord


class FlexuralClassification(StrictConfigModel):
    """Record AISC Table B4.1b flange and web flexural classifications."""

    flange_class: Literal["compact", "noncompact", "slender"]
    web_class: Literal["compact", "noncompact", "slender"]
    flange_slenderness: float = Field(gt=0.0, allow_inf_nan=False)
    web_slenderness: float = Field(gt=0.0, allow_inf_nan=False)
    flange_lambda_p: float = Field(gt=0.0, allow_inf_nan=False)
    flange_lambda_r: float = Field(gt=0.0, allow_inf_nan=False)
    web_lambda_p: float = Field(gt=0.0, allow_inf_nan=False)
    web_lambda_r: float = Field(gt=0.0, allow_inf_nan=False)


class FlexuralStrength(StrictConfigModel):
    """Record nominal/design major-axis strength and the controlling limit state."""

    classification: FlexuralClassification
    nominal_strength_nm: float = Field(gt=0.0, allow_inf_nan=False)
    design_strength_nm: float = Field(gt=0.0, allow_inf_nan=False)
    controlling_limit_state: Literal[
        "yielding",
        "inelastic_ltb",
        "elastic_ltb",
        "noncompact_flange_local_buckling",
        "slender_flange_local_buckling",
    ]
    unbraced_length_m: float = Field(ge=0.0, allow_inf_nan=False)
    lp_m: float = Field(gt=0.0, allow_inf_nan=False)
    lr_m: float = Field(gt=0.0, allow_inf_nan=False)
    cb: float = Field(gt=0.0, allow_inf_nan=False)


class AxialClassification(StrictConfigModel):
    """Record AISC Table B4.1a axial flange/web slenderness classifications."""

    flange_class: Literal["nonslender", "slender"]
    web_class: Literal["nonslender", "slender"]
    flange_slenderness: float = Field(gt=0.0, allow_inf_nan=False)
    web_slenderness: float = Field(gt=0.0, allow_inf_nan=False)
    flange_lambda_r: float = Field(gt=0.0, allow_inf_nan=False)
    web_lambda_r: float = Field(gt=0.0, allow_inf_nan=False)


class CompressionStrength(StrictConfigModel):
    """Record AISC E3/E7 nominal compression strength and effective-area reduction."""

    classification: AxialClassification
    critical_stress_pa: float = Field(gt=0.0, allow_inf_nan=False)
    gross_area_m2: float = Field(gt=0.0, allow_inf_nan=False)
    effective_area_m2: float = Field(gt=0.0, allow_inf_nan=False)
    nominal_strength_n: float = Field(gt=0.0, allow_inf_nan=False)
    controlling_limit_state: Literal["flexural_buckling", "local_and_flexural_buckling"]


def column_nondimensional_slenderness(
    *,
    effective_length_factor: float,
    length_m: float,
    radius_of_gyration_m: float,
    yield_strength_pa: float,
    elastic_modulus_pa: float,
) -> float:
    """Return REF-A Eq. (14) nondimensional column slenderness ``lambda_c``."""

    values = (
        effective_length_factor,
        length_m,
        radius_of_gyration_m,
        yield_strength_pa,
        elastic_modulus_pa,
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("column slenderness inputs must be finite and positive")
    return (
        effective_length_factor
        * length_m
        / (radius_of_gyration_m * math.pi)
        * math.sqrt(yield_strength_pa / elastic_modulus_pa)
    )


def critical_compression_stress_pa(
    *, nondimensional_slenderness: float, yield_strength_pa: float
) -> float:
    """Return either critical-stress branch printed together in REF-A Eq. (13)."""

    if nondimensional_slenderness <= 0.0 or not math.isfinite(nondimensional_slenderness):
        raise ValueError("nondimensional slenderness must be finite and positive")
    if yield_strength_pa <= 0.0 or not math.isfinite(yield_strength_pa):
        raise ValueError("yield strength must be finite and positive")
    if nondimensional_slenderness <= 1.5:
        return float(0.658 ** (nondimensional_slenderness**2)) * yield_strength_pa
    return 0.877 / (nondimensional_slenderness**2) * yield_strength_pa


def nominal_compression_strength_n(
    *,
    area_m2: float,
    effective_length_factor: float,
    length_m: float,
    radius_of_gyration_m: float,
    yield_strength_pa: float,
    elastic_modulus_pa: float,
) -> float:
    """Return REF-A Eq. (15) nominal strength using Eqs. (13)-(14)."""

    if area_m2 <= 0.0 or not math.isfinite(area_m2):
        raise ValueError("section area must be finite and positive")
    slenderness = column_nondimensional_slenderness(
        effective_length_factor=effective_length_factor,
        length_m=length_m,
        radius_of_gyration_m=radius_of_gyration_m,
        yield_strength_pa=yield_strength_pa,
        elastic_modulus_pa=elastic_modulus_pa,
    )
    return area_m2 * critical_compression_stress_pa(
        nondimensional_slenderness=slenderness,
        yield_strength_pa=yield_strength_pa,
    )


def compact_beam_design_flexural_strength_nm(
    *, plastic_section_modulus_m3: float, yield_strength_pa: float, resistance_factor: float
) -> float:
    """Return ``phi_b Fy Zx`` for the paper's compact preliminary-beam assumption."""

    values = (plastic_section_modulus_m3, yield_strength_pa, resistance_factor)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("compact beam strength inputs must be finite and positive")
    return resistance_factor * yield_strength_pa * plastic_section_modulus_m3


def classify_major_axis_flexure(
    section: SectionRecord, *, elastic_modulus_pa: float, yield_strength_pa: float
) -> FlexuralClassification:
    """Classify rolled-W flange and web slenderness using AISC Table B4.1b cases 10 and 15."""

    if elastic_modulus_pa <= 0.0 or yield_strength_pa <= 0.0:
        raise ValueError("elastic modulus and yield strength must be positive")
    root = math.sqrt(elastic_modulus_pa / yield_strength_pa)
    flange_p = 0.38 * root
    flange_r = 1.0 * root
    web_p = 3.76 * root
    web_r = 5.70 * root

    def classify(
        value: float, compact: float, slender: float
    ) -> Literal["compact", "noncompact", "slender"]:
        if value <= compact:
            return "compact"
        if value <= slender:
            return "noncompact"
        return "slender"

    return FlexuralClassification(
        flange_class=classify(section.flange_slenderness, flange_p, flange_r),
        web_class=classify(section.web_slenderness, web_p, web_r),
        flange_slenderness=section.flange_slenderness,
        web_slenderness=section.web_slenderness,
        flange_lambda_p=flange_p,
        flange_lambda_r=flange_r,
        web_lambda_p=web_p,
        web_lambda_r=web_r,
    )


def classify_axial_compression(
    section: SectionRecord, *, elastic_modulus_pa: float, yield_strength_pa: float
) -> AxialClassification:
    """Classify rolled-W elements using AISC Table B4.1a cases 1 and 5."""

    if elastic_modulus_pa <= 0.0 or yield_strength_pa <= 0.0:
        raise ValueError("elastic modulus and yield strength must be positive")
    root = math.sqrt(elastic_modulus_pa / yield_strength_pa)
    flange_r = 0.56 * root
    web_r = 1.49 * root
    return AxialClassification(
        flange_class="nonslender" if section.flange_slenderness <= flange_r else "slender",
        web_class="nonslender" if section.web_slenderness <= web_r else "slender",
        flange_slenderness=section.flange_slenderness,
        web_slenderness=section.web_slenderness,
        flange_lambda_r=flange_r,
        web_lambda_r=web_r,
    )


def _effective_width_ratio(
    *,
    slenderness: float,
    lambda_r: float,
    yield_strength_pa: float,
    critical_stress_pa: float,
    c1: float,
) -> float:
    """Return AISC E7-2/E7-3 effective width divided by gross element width."""

    if slenderness <= lambda_r * math.sqrt(yield_strength_pa / critical_stress_pa):
        return 1.0
    c2 = (1.0 - math.sqrt(1.0 - 4.0 * c1)) / (2.0 * c1)
    elastic_local_stress = (c2 * lambda_r / slenderness) ** 2 * yield_strength_pa
    root = math.sqrt(elastic_local_stress / critical_stress_pa)
    return max(min((1.0 - c1 * root) * root, 1.0), 0.0)


def nominal_compression_strength_with_local_buckling(
    section: SectionRecord,
    *,
    effective_length_factor: float,
    length_m: float,
    radius_of_gyration_m: float,
    yield_strength_pa: float,
    elastic_modulus_pa: float,
) -> CompressionStrength:
    """Return AISC E3/E7 W-shape compression strength including slender-element reduction."""

    slenderness = column_nondimensional_slenderness(
        effective_length_factor=effective_length_factor,
        length_m=length_m,
        radius_of_gyration_m=radius_of_gyration_m,
        yield_strength_pa=yield_strength_pa,
        elastic_modulus_pa=elastic_modulus_pa,
    )
    critical_stress = critical_compression_stress_pa(
        nondimensional_slenderness=slenderness,
        yield_strength_pa=yield_strength_pa,
    )
    classification = classify_axial_compression(
        section,
        elastic_modulus_pa=elastic_modulus_pa,
        yield_strength_pa=yield_strength_pa,
    )
    effective_area = section.area_m2
    if classification.flange_class == "slender":
        ratio = _effective_width_ratio(
            slenderness=section.flange_slenderness,
            lambda_r=classification.flange_lambda_r,
            yield_strength_pa=yield_strength_pa,
            critical_stress_pa=critical_stress,
            c1=0.22,
        )
        one_outstand_width = section.flange_width_m / 2.0
        effective_area -= 4.0 * one_outstand_width * (1.0 - ratio) * section.flange_thickness_m
    if classification.web_class == "slender":
        ratio = _effective_width_ratio(
            slenderness=section.web_slenderness,
            lambda_r=classification.web_lambda_r,
            yield_strength_pa=yield_strength_pa,
            critical_stress_pa=critical_stress,
            c1=0.18,
        )
        web_height = section.web_slenderness * section.web_thickness_m
        effective_area -= web_height * (1.0 - ratio) * section.web_thickness_m
    if effective_area <= 0.0:
        raise ValueError(f"effective area is nonpositive for {section.name}")
    reduced = not math.isclose(effective_area, section.area_m2, rel_tol=0.0, abs_tol=1e-15)
    return CompressionStrength(
        classification=classification,
        critical_stress_pa=critical_stress,
        gross_area_m2=section.area_m2,
        effective_area_m2=effective_area,
        nominal_strength_n=critical_stress * effective_area,
        controlling_limit_state="local_and_flexural_buckling" if reduced else "flexural_buckling",
    )


def effective_length_factor_unbraced(*, g_a: float, g_b: float) -> float:
    """Return REF-A Eq. (26) unbraced-frame effective-length factor from end ratios."""

    if any(value < 0.0 or not math.isfinite(value) for value in (g_a, g_b)):
        raise ValueError("end stiffness ratios must be finite and nonnegative")
    numerator = 1.6 * g_a * g_b + 1.4 * (g_a + g_b) + 0.64
    denominator = g_a + g_b + 7.5
    return math.sqrt(numerator / denominator)


def moment_gradient_factor(
    *, moment_a_nm: float, moment_b_nm: float, moment_c_nm: float, maximum_moment_nm: float
) -> float:
    """Return AISC Eq. F1-1 ``Cb`` from absolute quarter-point moments."""

    moments = tuple(abs(value) for value in (moment_a_nm, moment_b_nm, moment_c_nm))
    maximum = abs(maximum_moment_nm)
    if maximum <= 0.0 or not math.isfinite(maximum):
        return 1.0
    denominator = 2.5 * maximum + 3.0 * moments[0] + 4.0 * moments[1] + 3.0 * moments[2]
    if denominator <= 0.0:
        raise ValueError("moment-gradient denominator must be positive")
    return 12.5 * maximum / denominator


def nominal_major_axis_flexural_strength(
    section: SectionRecord,
    *,
    elastic_modulus_pa: float,
    yield_strength_pa: float,
    resistance_factor: float,
    unbraced_length_m: float,
    cb: float,
    fully_braced: bool,
) -> FlexuralStrength:
    """Return AISC F2/F3 major-axis strength for a rolled W shape with compact web."""

    values = (elastic_modulus_pa, yield_strength_pa, resistance_factor, cb)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("flexural strength inputs must be finite and positive")
    if unbraced_length_m < 0.0 or not math.isfinite(unbraced_length_m):
        raise ValueError("unbraced length must be finite and nonnegative")
    classification = classify_major_axis_flexure(
        section, elastic_modulus_pa=elastic_modulus_pa, yield_strength_pa=yield_strength_pa
    )
    if classification.web_class != "compact":
        raise ValueError("current W-shape Stage 1 domain requires a compact flexural web")

    plastic_moment = yield_strength_pa * section.zx_m3
    lp = 1.76 * section.ry_m * math.sqrt(elastic_modulus_pa / yield_strength_pa)
    j_term = section.torsional_j_m4 / (section.sx_m3 * section.flange_centroid_distance_ho_m)
    lr = (
        1.95
        * section.rts_m
        * elastic_modulus_pa
        / (0.7 * yield_strength_pa)
        * math.sqrt(
            j_term
            + math.sqrt(j_term**2 + 6.76 * (0.7 * yield_strength_pa / elastic_modulus_pa) ** 2)
        )
    )
    if fully_braced or unbraced_length_m <= lp:
        lateral_strength = plastic_moment
        lateral_state: Literal["yielding", "inelastic_ltb", "elastic_ltb"] = "yielding"
    elif unbraced_length_m <= lr:
        lateral_strength = min(
            cb
            * (
                plastic_moment
                - (plastic_moment - 0.7 * yield_strength_pa * section.sx_m3)
                * (unbraced_length_m - lp)
                / (lr - lp)
            ),
            plastic_moment,
        )
        lateral_state = "inelastic_ltb"
    else:
        ratio = unbraced_length_m / section.rts_m
        critical_stress = (
            cb
            * math.pi**2
            * elastic_modulus_pa
            / ratio**2
            * math.sqrt(1.0 + 0.078 * j_term * ratio**2)
        )
        lateral_strength = min(critical_stress * section.sx_m3, plastic_moment)
        lateral_state = "elastic_ltb"

    if classification.flange_class == "compact":
        local_strength = plastic_moment
        local_state: Literal[
            "yielding", "noncompact_flange_local_buckling", "slender_flange_local_buckling"
        ] = "yielding"
    elif classification.flange_class == "noncompact":
        local_strength = plastic_moment - (
            plastic_moment - 0.7 * yield_strength_pa * section.sx_m3
        ) * (
            (section.flange_slenderness - classification.flange_lambda_p)
            / (classification.flange_lambda_r - classification.flange_lambda_p)
        )
        local_state = "noncompact_flange_local_buckling"
    else:
        kc = min(max(4.0 / math.sqrt(section.web_slenderness), 0.35), 0.76)
        local_strength = (
            0.9 * elastic_modulus_pa * kc * section.sx_m3 / section.flange_slenderness**2
        )
        local_state = "slender_flange_local_buckling"

    if local_strength < lateral_strength:
        nominal = local_strength
        controlling: Literal[
            "yielding",
            "inelastic_ltb",
            "elastic_ltb",
            "noncompact_flange_local_buckling",
            "slender_flange_local_buckling",
        ] = local_state
    else:
        nominal = lateral_strength
        controlling = lateral_state
    return FlexuralStrength(
        classification=classification,
        nominal_strength_nm=nominal,
        design_strength_nm=resistance_factor * nominal,
        controlling_limit_state=controlling,
        unbraced_length_m=unbraced_length_m,
        lp_m=lp,
        lr_m=lr,
        cb=cb,
    )


def beam_column_interaction_utilization(
    *,
    compression_demand_n: float,
    moment_demand_nm: float,
    nominal_compression_strength_n_value: float,
    nominal_flexural_strength_nm: float,
    compression_resistance_factor: float,
    flexural_resistance_factor: float,
) -> float:
    """Return REF-A Eq. (11) axial-flexural utilization with its 0.2 branch."""

    values = (
        compression_demand_n,
        moment_demand_nm,
        nominal_compression_strength_n_value,
        nominal_flexural_strength_nm,
        compression_resistance_factor,
        flexural_resistance_factor,
    )
    if any(value < 0.0 or not math.isfinite(value) for value in values[:2]):
        raise ValueError("interaction demands must be finite and nonnegative")
    if any(value <= 0.0 or not math.isfinite(value) for value in values[2:]):
        raise ValueError("interaction strengths and resistance factors must be positive")
    axial_ratio = compression_demand_n / (
        compression_resistance_factor * nominal_compression_strength_n_value
    )
    moment_ratio = moment_demand_nm / (flexural_resistance_factor * nominal_flexural_strength_nm)
    if axial_ratio < 0.2:
        return axial_ratio / 2.0 + moment_ratio
    return axial_ratio + 8.0 * moment_ratio / 9.0
