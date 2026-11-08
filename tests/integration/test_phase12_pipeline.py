"""Integration checks for nine-story evaluator profiles and GA campaign contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from steel_frame_optimization.optimization.campaign import CampaignRunner
from steel_frame_optimization.optimization.evaluator import EvaluatorPaths, StructuralEvaluator


def test_nine_story_profiles_preserve_domains_and_case_scopes(project_root: Path) -> None:
    """Distinguish filtered/no-filter domains and source-defined Case 1/Case 2 constraints."""

    filtered = CampaignRunner(
        project_root=project_root,
        policy_path=project_root / "configs/optimization/phase12_case1_filtered_main.json",
    )
    unfiltered = CampaignRunner(
        project_root=project_root,
        policy_path=project_root / "configs/optimization/phase12_case1_no_filter_main.json",
    )
    case2 = CampaignRunner(
        project_root=project_root,
        policy_path=project_root / "configs/optimization/phase12_case2_main.json",
    )
    assert filtered.codec.domain_sizes == (35, 29, 31, 36, 33, 36, 36, 36, 228, 228, 228, 228, 232)
    assert unfiltered.codec.domain_sizes == (36,) * 8 + (273,) * 5
    assert (
        filtered.evaluator.evaluation_policy.stage1_constraint_scope
        == "gravity_strength_and_service"
    )
    assert case2.evaluator.evaluation_policy.stage1_constraint_scope == "full"
    assert len(case2.codec.group_ids) == 13


def test_nine_story_campaigns_freeze_exact_scale(project_root: Path) -> None:
    """Require the owner-approved 100-by-200, seeds 0-29 campaign accounting."""

    for name in (
        "phase12_case1_filtered_main.json",
        "phase12_case1_no_filter_main.json",
        "phase12_case2_main.json",
    ):
        runner = CampaignRunner(
            project_root=project_root,
            policy_path=project_root / "configs/optimization" / name,
        )
        assert runner.policy.population_size == 100
        assert runner.policy.evaluated_population_epochs == 200
        assert runner.policy.evaluation_request_budget_per_seed == 20_000
        assert runner.policy.seeds == tuple(range(30))


def test_modal_backend_enforces_timeout_and_aggregate_cost_gate(project_root: Path) -> None:
    """Keep measured runtime, recovered memory, and authorized cost within hard bounds."""

    backend = json.loads(
        (project_root / "configs/optimization/phase12_modal_backend.json").read_text(
            encoding="utf-8"
        )
    )
    projected_seed_seconds = (
        backend["evaluation_requests_per_seed"] * backend["measured_seconds_per_request"]
    )
    assert backend["seed_timeout_seconds"] >= projected_seed_seconds * 1.2

    container_cost_per_second = (
        backend["container_cpu_cores"] * backend["cpu_cost_usd_per_core_second"]
        + backend["container_memory_mib"] / 1024.0 * backend["memory_cost_usd_per_gib_second"]
    )
    projected_experiment_cost = (
        backend["campaign_count"]
        * backend["seeds_per_campaign"]
        * backend["evaluation_requests_per_seed"]
        * backend["measured_seconds_per_request"]
        * container_cost_per_second
        * backend["cost_contingency_factor"]
    )
    assert backend["container_memory_mib"] >= 4096
    assert backend["container_memory_limit_mib"] >= 8192
    assert math.isclose(projected_experiment_cost, 70.45386286544555, rel_tol=1e-12)
    assert backend["owner_approved_estimate_usd"] >= projected_experiment_cost
    assert projected_experiment_cost <= backend["maximum_authorized_total_compute_cost_usd"]

    timeout_bound_all_seeds = (
        backend["seeds_per_campaign"] * backend["seed_timeout_seconds"] * container_cost_per_second
    )
    assert timeout_bound_all_seeds <= backend["maximum_projected_compute_cost_usd_per_submission"]
    assert backend["maximum_parallel_campaigns"] == backend["campaign_count"]
    assert backend["maximum_parallel_seeds_global"] == (
        backend["maximum_parallel_campaigns"] * backend["maximum_parallel_seeds"]
    )


def test_published_design_evaluator_dispatch_is_explicit(project_root: Path) -> None:
    """Verify Case 1 subset counts, Case 2 full counts, and transparent dispositions."""

    cases = (
        ("nine_story_case1_filtered", "nine_story_case1_gwo.json", 234, "stage2_infeasible"),
        ("nine_story_case1_no_filter", "nine_story_case1_no_filter_gwo.json", 234, "feasible"),
        ("nine_story_case2", "nine_story_case2_gwo.json", 452, "stage2_infeasible"),
    )
    for profile, case_name, expected_constraints, expected_status in cases:
        evaluator = StructuralEvaluator(EvaluatorPaths.from_profile(project_root, profile))
        content = json.loads(
            (project_root / "configs/cases" / case_name).read_text(encoding="utf-8")
        )
        result = evaluator.evaluate(
            evaluation_id=f"test-{profile}",
            section_names_by_group=content["sections_by_group"],
            use_cache=False,
        ).result
        assert result.stage1.constraint_count == expected_constraints
        assert result.status == expected_status
