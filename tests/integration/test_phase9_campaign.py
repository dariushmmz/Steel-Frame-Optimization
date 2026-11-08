"""Layer 1/2 integration tests for process-isolated GA campaigns and recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from steel_frame_optimization.optimization.campaign import (
    CampaignPolicy,
    CampaignRunner,
    DesignCodec,
)
from steel_frame_optimization.optimization.evaluator import EvaluatorPaths, StructuralEvaluator
from steel_frame_optimization.sections.domains import DomainCatalog
from steel_frame_optimization.serialization import write_canonical_json

pytestmark = pytest.mark.integration


def _pilot_policy() -> CampaignPolicy:
    """Return a two-epoch campaign small enough for locked regression tests."""

    return CampaignPolicy.model_validate(
        {
            "campaign_id": "phase9-test-resume",
            "mode": "bounded_pilot",
            "frame_id": "3-story",
            "case_id": "3_story_case_2_principal",
            "population_size": 4,
            "evaluated_population_epochs": 2,
            "evaluation_request_budget_per_seed": 8,
            "seeds": [0],
            "seed_manifest_path": "configs/optimization/seed_manifest.json",
            "worker_processes": 2,
            "checkpoint_every_epochs": 1,
            "initial_designs": [
                {
                    "design_id": "verified-anchor",
                    "sections_by_group": {
                        "1": "W16X77",
                        "2": "W16X89",
                        "3": "W21X68",
                        "4": "W21X62",
                        "5": "W21X44",
                    },
                }
            ],
            "ga": {
                "selection": "tournament",
                "tournament_size": 2,
                "crossover": "one_point",
                "crossover_probability": 0.9,
                "mutation": "ordered_neighborhood",
                "mutation_probability_per_gene": 0.2,
                "mutation_window_fraction": 0.1,
                "elite_count": 1,
            },
            "assumption_ids": ["ASM-002", "ASM-016", "ASM-038"],
        }
    )


def test_design_codec_uses_approved_filtered_operator_domains() -> None:
    """Genome domains reproduce the accepted 3/14/45/45/41 removal baseline."""

    root = Path(__file__).resolve().parents[2]
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    codec = DesignCodec(
        evaluator,
        DomainCatalog.from_json(root / "configs" / "frames" / "phase2_candidate_domains.json"),
    )
    assert codec.group_ids == ("1", "2", "3", "4", "5")
    assert codec.domain_sizes == (270, 259, 228, 228, 232)
    sections = {"1": "W16X77", "2": "W16X89", "3": "W21X68", "4": "W21X62", "5": "W21X44"}
    assert codec.decode(codec.encode(sections)) == sections


def test_main_campaign_loads_only_the_frozen_approved_scale() -> None:
    """Machine-readable approval opens exactly the frozen 30-seed, 10,000-request policy."""

    root = Path(__file__).resolve().parents[2]
    runner = CampaignRunner(
        project_root=root,
        policy_path=root / "configs" / "optimization" / "phase9_three_story_main.json",
    )
    assert runner.policy.mode == "main_30_seed"
    assert runner.policy.seeds == tuple(range(30))
    assert runner.policy.population_size == 50
    assert runner.policy.evaluated_population_epochs == 200
    assert runner.policy.evaluation_request_budget_per_seed == 10_000
    assert runner.policy.worker_processes == 4
    assert runner.penalty_approval.approval_status == "approved"
    assert runner.penalty_approval.penalty_policy_sha256 == (
        "a9ebb35ad9274c21e136375979fa11397699a661e535a48fed2e074174aebe48"
    )
    assert runner.campaign_sha256 == (
        "aa2f2f23e33bd8579b3c80f33584597e3a39cbecf3c712c3dba8ba17cf2e1ab8"
    )


def test_spawned_campaign_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    """A canonical checkpoint restores RNG, population, cache, counts and convergence exactly."""

    root = Path(__file__).resolve().parents[2]
    policy_path = tmp_path / "pilot.json"
    write_canonical_json(policy_path, _pilot_policy())
    runner = CampaignRunner(project_root=root, policy_path=policy_path)
    uninterrupted = runner.run_seed(seed=0, checkpoint_path=tmp_path / "full.json")
    partial = runner.run_seed(
        seed=0,
        checkpoint_path=tmp_path / "resume.json",
        maximum_epochs=1,
    )
    assert partial.completed_epochs == 1
    resumed = runner.run_seed(
        seed=0,
        checkpoint_path=tmp_path / "resume.json",
        resume=True,
    )
    assert resumed.completed_epochs == 2
    assert resumed.evaluation_requests == uninterrupted.evaluation_requests == 8
    assert resumed.structural_executions == uninterrupted.structural_executions
    assert resumed.cache_hits == uninterrupted.cache_hits
    assert resumed.structural_executions + resumed.cache_hits == resumed.evaluation_requests
    assert resumed.champion_genome == uninterrupted.champion_genome
    assert resumed.champion == uninterrupted.champion
    assert resumed.final_population == uninterrupted.final_population
    assert resumed.final_results == uninterrupted.final_results
    assert resumed.history == uninterrupted.history
    assert resumed.independently_reevaluated_champion_equal
    assert uninterrupted.independently_reevaluated_champion_equal
    assert uninterrupted.worker_process_ids
    assert len(uninterrupted.worker_process_ids) == 2
    assert os.getpid() not in uninterrupted.worker_process_ids

    checkpoint_content = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    checkpoint_content["campaign_sha256"] = "0" * 64
    (tmp_path / "corrupt.json").write_text(json.dumps(checkpoint_content), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign identity"):
        runner.run_seed(seed=0, checkpoint_path=tmp_path / "corrupt.json", resume=True)
