"""Generate the deterministic Phase 4 published-design modal and elastic evidence artifact."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from steel_frame_optimization.analysis.elastic import (
    LoadFactors,
    ModalResult,
    StaticResult,
    run_modal_analysis,
    run_static_analysis,
)
from steel_frame_optimization.analysis.modal import (
    distribute_story_forces_n,
    elastic_base_shear_n,
    spectral_acceleration_g,
)
from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.models.frame import FrameDefinition, generate_frame
from steel_frame_optimization.sections.repository import SectionRepository
from steel_frame_optimization.serialization import write_canonical_json


class SpectrumInput(StrictConfigModel):
    """Validate the source-transcribed Case 2 spectrum and force-distribution inputs."""

    short_period_g: float = Field(gt=0.0)
    one_second_g: float = Field(gt=0.0)
    fa: float = Field(gt=0.0)
    fv: float = Field(gt=0.0)
    r_factor: float = Field(gt=0.0)
    distribution_exponent: float = Field(gt=0.0)


class AnalysisPolicyInput(StrictConfigModel):
    """Validate the predeclared Phase 4 formulation and numerical tolerances."""

    element_formulation: Literal["elastic_euler_bernoulli"]
    second_order_transformation: Literal["PDelta"]
    beam_midpoint_nodes: Literal[True]
    column_elements_per_story: Literal[1]
    numerical_relative_tolerance: float = Field(gt=0.0)
    equilibrium_relative_tolerance: float = Field(gt=0.0)
    repeat_relative_tolerance: float = Field(gt=0.0)
    exact_beam_column_relative_tolerance: float = Field(gt=0.0)
    pdelta_refinement_sequence: tuple[int, ...]
    minimum_superposition_error: float = Field(gt=0.0)
    status: Literal["agent-decided, pending owner/engineer review"]

    @model_validator(mode="after")
    def require_predeclared_tolerances(self) -> AnalysisPolicyInput:
        """Reject any silent tolerance change from the owner-frozen Phase 4 values."""

        observed = (
            self.numerical_relative_tolerance,
            self.equilibrium_relative_tolerance,
            self.repeat_relative_tolerance,
        )
        if observed != (1e-6, 1e-8, 1e-12):
            raise ValueError("Phase 4 tolerances are frozen at 1e-6, 1e-8, and 1e-12")
        if self.exact_beam_column_relative_tolerance != 1e-4:
            raise ValueError("exact beam-column tolerance is frozen at 1e-4")
        if self.pdelta_refinement_sequence != (1, 2, 4, 8, 16, 32):
            raise ValueError("P-Delta refinement sequence is frozen at 1, 2, 4, 8, 16, 32")
        if self.minimum_superposition_error != 0.05:
            raise ValueError("minimum superposition-error fixture is frozen at 0.05")
        return self


class PublishedElasticCase(StrictConfigModel):
    """Validate one paper section set and its complete direct-combination analysis inputs."""

    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    frame_id: Literal["3-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    source: Literal["REF-A Table 4, Case 2 GWO"]
    sections_by_group: dict[str, str]
    spectrum: SpectrumInput
    load_combination: LoadFactors
    analysis_policy: AnalysisPolicyInput

    @classmethod
    def from_json(cls, path: Path) -> PublishedElasticCase:
        """Load and strictly validate one published elastic-case configuration."""

        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class ColumnStabilityDiagnostic(StrictConfigModel):
    """Record a non-code K=1 Euler axial ratio for P-delta mesh-review triage."""

    member_id: str
    group_id: str
    axial_demand_n: float
    euler_pcr_k1_n: float
    demand_to_pcr: float


class Phase4Artifact(StrictConfigModel):
    """Persist accepted Phase 4 layer-1/2 results without claiming paper agreement."""

    schema_version: Literal["1.0"] = "1.0"
    validation_status: Literal["not engineering-validated"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    case: PublishedElasticCase
    modal: ModalResult
    spectral_acceleration_g: float
    base_shear_n: float
    story_forces_n: tuple[float, ...]
    positive_e: StaticResult
    negative_e: StaticResult
    column_stability_diagnostics: tuple[ColumnStabilityDiagnostic, ...]
    column_p_over_pcr_min: float
    column_p_over_pcr_max: float


def build_phase4_artifact(
    frame_path: Path,
    sections_path: Path,
    case_path: Path,
) -> Phase4Artifact:
    """Run direct staged ±E combinations and assemble the deterministic evidence model."""

    definition = FrameDefinition.from_json(frame_path)
    generated = generate_frame(definition)
    repository = SectionRepository.from_json(sections_path)
    case = PublishedElasticCase.from_json(case_path)
    sections = {
        group_id: repository.get(section_name)
        for group_id, section_name in case.sections_by_group.items()
    }
    modal = run_modal_analysis(generated, sections)
    acceleration_g = spectral_acceleration_g(
        modal.periods_s[0],
        short_period_g=case.spectrum.short_period_g,
        one_second_g=case.spectrum.one_second_g,
        fa=case.spectrum.fa,
        fv=case.spectrum.fv,
    )
    weights = tuple(floor.seismic_weight_n for floor in definition.floors)
    base_shear = elastic_base_shear_n(sum(weights), acceleration_g, case.spectrum.r_factor)
    elevation = 0.0
    elevations: list[float] = []
    for height in definition.story_heights_m:
        elevation += height
        elevations.append(elevation)
    story_forces = distribute_story_forces_n(
        base_shear,
        weights,
        elevations,
        exponent=case.spectrum.distribution_exponent,
    )
    positive = run_static_analysis(
        generated, sections, case.load_combination, story_forces, p_delta=True
    )
    negative = run_static_analysis(
        generated,
        sections,
        case.load_combination,
        tuple(-force for force in story_forces),
        p_delta=True,
    )
    members = {member.member_id: member for member in generated.topology.members}
    nodes = {node.node_id: node for node in generated.topology.nodes}
    diagnostics: list[ColumnStabilityDiagnostic] = []
    for force in positive.element_forces:
        member = members[force.semantic_member_id]
        if member.member_type != "column":
            continue
        section = sections[member.group_id]
        i_node = nodes[member.i_node]
        j_node = nodes[member.j_node]
        length = math.hypot(j_node.x_m - i_node.x_m, j_node.y_m - i_node.y_m)
        pcr = math.pi**2 * definition.materials[0].elastic_modulus_pa * section.ix_m4 / length**2
        axial = max(abs(force.local_end_forces_n_nm[0]), abs(force.local_end_forces_n_nm[3]))
        diagnostics.append(
            ColumnStabilityDiagnostic(
                member_id=member.member_id,
                group_id=member.group_id,
                axial_demand_n=axial,
                euler_pcr_k1_n=pcr,
                demand_to_pcr=axial / pcr,
            )
        )
    ratios = [diagnostic.demand_to_pcr for diagnostic in diagnostics]
    return Phase4Artifact(
        validation_status="not engineering-validated",
        domain_label=case.domain_label,
        case=case,
        modal=modal,
        spectral_acceleration_g=acceleration_g,
        base_shear_n=base_shear,
        story_forces_n=story_forces,
        positive_e=positive,
        negative_e=negative,
        column_stability_diagnostics=tuple(diagnostics),
        column_p_over_pcr_min=min(ratios),
        column_p_over_pcr_max=max(ratios),
    )


def _parser() -> argparse.ArgumentParser:
    """Create the explicit Phase 4 evidence-generation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate canonical Phase 4 evidence and report its primary response values."""

    arguments = _parser().parse_args(argv)
    artifact = build_phase4_artifact(arguments.frame, arguments.sections, arguments.case)
    write_canonical_json(arguments.output, artifact)
    print(
        f"T1={artifact.modal.periods_s[0]:.9f} s, "
        f"Vb={artifact.base_shear_n:.3f} N, "
        f"roof(+E)={artifact.positive_e.roof_displacement_m:.9f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
