"""Define strict Phase 1 configuration schemas without structural implementation.

The models validate identities, references, decision state, and explicit ambiguity handling. They do
not calculate response, capacity, loads, or optimization fitness.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Reject unknown fields and mutation in all persisted configuration models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NodeSpec(StrictConfigModel):
    """Identify one configured node and its canonical-metre coordinates."""

    node_id: str = Field(min_length=1)
    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)


class MemberSpec(StrictConfigModel):
    """Describe member connectivity and grouping without assigning section properties."""

    member_id: str = Field(min_length=1)
    i_node: str = Field(min_length=1)
    j_node: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    member_type: Literal["beam", "column"]

    @model_validator(mode="after")
    def reject_self_connection(self) -> MemberSpec:
        """Reject zero-topology members before any future geometry processing."""

        if self.i_node == self.j_node:
            raise ValueError("member endpoints must be different")
        return self


class SupportSpec(StrictConfigModel):
    """Declare restrained planar degrees of freedom for one configured node."""

    node_id: str = Field(min_length=1)
    restrained_dofs: tuple[Literal["ux", "uy", "rz"], ...] = Field(min_length=1)


class FrameSpec(StrictConfigModel):
    """Validate frame identities and references without generating a structural model."""

    schema_version: Literal["1.0"] = "1.0"
    frame_id: str = Field(min_length=1)
    length_unit: Literal["m"] = "m"
    force_unit: Literal["N"] = "N"
    mass_unit: Literal["kg"] = "kg"
    design_groups: tuple[str, ...] = Field(min_length=1)
    nodes: tuple[NodeSpec, ...] = Field(min_length=2)
    members: tuple[MemberSpec, ...] = Field(min_length=1)
    supports: tuple[SupportSpec, ...]

    @model_validator(mode="after")
    def validate_references_and_uniqueness(self) -> FrameSpec:
        """Require unique semantic IDs and resolvable node/group references."""

        node_ids = [node.node_id for node in self.nodes]
        member_ids = [member.member_id for member in self.members]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node IDs must be unique")
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("member IDs must be unique")
        if len(set(self.design_groups)) != len(self.design_groups):
            raise ValueError("design group IDs must be unique")
        known_nodes = set(node_ids)
        known_groups = set(self.design_groups)
        for member in self.members:
            if member.i_node not in known_nodes or member.j_node not in known_nodes:
                raise ValueError(f"member {member.member_id} references an unknown node")
            if member.group_id not in known_groups:
                raise ValueError(f"member {member.member_id} references an unknown group")
        for support in self.supports:
            if support.node_id not in known_nodes:
                raise ValueError(f"support references unknown node {support.node_id}")
        return self


class AnalysisCase(StrictConfigModel):
    """Persist DEC-001 variants while prohibiting a silent DEC-002 mapping."""

    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    variant_kind: Literal["primary", "sensitivity"]
    ls_target_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    ls_target_decision: Literal["DEC-001"] = "DEC-001"
    performance_level_mapping: None = None
    mapping_status: Literal["awaiting_structural_engineer_DEC-002"]
    solver_policy_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def enforce_named_decision_variants(self) -> AnalysisCase:
        """Require 0.025H for primary and 0.02H for the named sensitivity."""

        expected = 0.025 if self.variant_kind == "primary" else 0.02
        if self.ls_target_ratio != expected:
            raise ValueError(
                f"{self.variant_kind} DEC-001 variant requires ls_target_ratio={expected}"
            )
        return self


class DecisionOption(StrictConfigModel):
    """Describe one decision option and its explicit trade-offs."""

    option_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tradeoffs: tuple[str, ...] = Field(min_length=1)


class DecisionVariant(StrictConfigModel):
    """Represent an approved or unresolved DEC item without silently selecting an option."""

    schema_version: Literal["1.0"] = "1.0"
    decision_id: str = Field(pattern=r"^DEC-\d{3}$")
    title: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    blocking_phases: tuple[str, ...] = Field(min_length=1)
    status: Literal["open", "awaiting_owner", "approved", "rejected"]
    recommendation: str | None = None
    options: tuple[DecisionOption, ...] = Field(min_length=1)
    selected_option_id: str | None = None

    @model_validator(mode="after")
    def validate_selection_state(self) -> DecisionVariant:
        """Require unique options and a selected option only for decided states."""

        option_ids = [option.option_id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("decision option IDs must be unique")
        if self.status in {"open", "awaiting_owner"} and self.selected_option_id is not None:
            raise ValueError("unresolved decisions cannot select an option")
        if self.status in {"approved", "rejected"} and self.selected_option_id not in option_ids:
            raise ValueError("resolved decisions require a known selected option")
        return self


class SolverAttempt(StrictConfigModel):
    """Name a future solver attempt without interpreting its structural behavior."""

    attempt_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    convergence_test: str = Field(min_length=1)
    maximum_iterations: int = Field(gt=0)


class SolverPolicy(StrictConfigModel):
    """Validate solver-policy state while allowing an explicit unimplemented placeholder."""

    schema_version: Literal["1.0"] = "1.0"
    policy_id: str = Field(min_length=1)
    implementation_phase: str = Field(min_length=1)
    status: Literal["placeholder_unimplemented", "approved"]
    attempts: tuple[SolverAttempt, ...]

    @model_validator(mode="after")
    def require_attempts_for_approved_policy(self) -> SolverPolicy:
        """Prevent an empty placeholder from being labeled as an approved solver policy."""

        if self.status == "approved" and not self.attempts:
            raise ValueError("approved solver policy requires at least one attempt")
        return self


class OptimizationCampaign(StrictConfigModel):
    """Represent a future campaign with an explicit delegated or ratified decision state."""

    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str = Field(min_length=1)
    case_ids: tuple[str, ...] = Field(min_length=1)
    seed_manifest_path: str = Field(min_length=1)
    status: Literal["agent_decided_pending_ratification_DEC-011_DEC-012", "approved"]
    population_size: int | None = Field(default=None, gt=0)
    generations: int | None = Field(default=None, gt=0)
    structural_evaluation_budget: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_complete_settings(self) -> OptimizationCampaign:
        """Require the complete predeclared budget for delegated and ratified campaigns."""

        settings = (self.population_size, self.generations, self.structural_evaluation_budget)
        if any(setting is None for setting in settings):
            raise ValueError("campaign requires population, generations, and evaluation budget")
        return self


class CampaignAggregateResult(StrictConfigModel):
    """Define the client-aligned aggregate output contract for one 30-run campaign.

    The contract stores both all-run and feasible-only weight statistics so an infeasible final
    design cannot silently influence the paper comparison. It is a schema only; no optimization or
    structural result is produced in Phase 2.
    """

    schema_version: Literal["1.0"] = "1.0"
    frame_id: Literal["3-story", "9-story"]
    domain_label: Literal["273-section domain, pending client confirmation"]
    run_count: Literal[30] = 30
    best_normalized_objective: float = Field(allow_inf_nan=False)
    champion_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    champion_unweighted_uniformity: float = Field(ge=0.0, allow_inf_nan=False)
    champion_weight_contribution: float = Field(allow_inf_nan=False)
    champion_uniformity_contribution: float = Field(allow_inf_nan=False)
    all_runs_mean_final_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    all_runs_worst_final_weight_kn: float = Field(gt=0.0, allow_inf_nan=False)
    feasible_runs_mean_final_weight_kn: float | None = Field(default=None, gt=0.0)
    feasible_runs_worst_final_weight_kn: float | None = Field(default=None, gt=0.0)
    feasible_final_design_count: int = Field(ge=0, le=30)
    infeasible_final_design_count: int = Field(ge=0, le=30)

    @model_validator(mode="after")
    def validate_counts_and_feasible_statistics(self) -> CampaignAggregateResult:
        """Require exhaustive counts and feasible statistics when feasible runs exist."""

        if self.feasible_final_design_count + self.infeasible_final_design_count != self.run_count:
            raise ValueError("feasible and infeasible counts must sum to 30")
        feasible_statistics = (
            self.feasible_runs_mean_final_weight_kn,
            self.feasible_runs_worst_final_weight_kn,
        )
        if self.feasible_final_design_count == 0 and any(
            value is not None for value in feasible_statistics
        ):
            raise ValueError("feasible-only statistics must be null when no feasible run exists")
        if self.feasible_final_design_count > 0 and any(
            value is None for value in feasible_statistics
        ):
            raise ValueError("feasible-only statistics are required when feasible runs exist")
        return self
