"""Layer 1/2 integration tests for deterministic two-stage structural evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import steel_frame_optimization.optimization.evaluator as evaluator_module
from steel_frame_optimization.models.failures import FailureCategory
from steel_frame_optimization.objective.functions import calculate_objective, drift_uniformity_term
from steel_frame_optimization.optimization.evaluator import (
    EvaluationResult,
    EvaluationStatus,
    EvaluatorPaths,
    StructuralEvaluator,
    evaluation_order_key,
)
from steel_frame_optimization.serialization import canonical_json_dumps
from steel_frame_optimization.validation.phase8 import (
    Phase8Artifact,
    build_phase8_artifact,
    phase8_gate_failures,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def phase8_artifact() -> Phase8Artifact:
    """Build the three representative evaluations once for this module."""

    root = Path(__file__).resolve().parents[2]
    return build_phase8_artifact(
        project_root=root,
        representative_designs_path=(
            root / "configs" / "cases" / "phase8_representative_designs.json"
        ),
    )


def _result_by_status(artifact: Phase8Artifact, status: EvaluationStatus) -> EvaluationResult:
    """Return the unique representative result with the requested status."""

    return next(result for result in artifact.results if result.status == status)


def test_phase8_gate_passes(phase8_artifact: Phase8Artifact) -> None:
    """Representative dispatch, cache, and feasibility separation all pass."""

    assert phase8_gate_failures(phase8_artifact) == ()


def test_stage2_executes_only_after_stage1_feasibility(
    phase8_artifact: Phase8Artifact,
) -> None:
    """The threshold design stops after Stage 1 while both Stage-1-feasible designs continue."""

    threshold = _result_by_status(phase8_artifact, "stage1_infeasible")
    stage2_failed = _result_by_status(phase8_artifact, "stage2_infeasible")
    feasible = _result_by_status(phase8_artifact, "feasible")
    assert threshold.stage1.executed and not threshold.stage1.feasible
    assert not threshold.stage2.executed
    assert threshold.objective is None
    assert stage2_failed.stage1.feasible and stage2_failed.stage2.executed
    assert feasible.stage1.feasible and feasible.stage2.executed


def test_objective_recomputes_from_saved_raw_cp_profile(
    phase8_artifact: Phase8Artifact,
) -> None:
    """An independent calculation from raw weight/displacements reproduces the saved objective."""

    result = _result_by_status(phase8_artifact, "feasible")
    assert result.objective is not None
    assert result.structural_weight_n is not None
    assert result.cp_story_displacements_m is not None
    uniformity = drift_uniformity_term(
        floor_displacements_m=result.cp_story_displacements_m,
        floor_elevations_m=(3.9624, 7.9248, 11.8872),
    )
    recomputed = calculate_objective(
        structural_weight=result.structural_weight_n,
        maximum_structural_weight=result.objective.maximum_structural_weight_n,
        uniformity_term=uniformity,
        weight_coefficient=0.95,
        uniformity_coefficient=0.05,
    )
    assert recomputed == result.objective


def test_ranking_orders_feasible_then_structural_infeasible_then_failure(
    phase8_artifact: Phase8Artifact,
) -> None:
    """The rank class and penalty score cannot place an infeasible result before feasible."""

    ordered = sorted(phase8_artifact.results, key=evaluation_order_key)
    assert [result.status for result in ordered] == [
        "feasible",
        "stage2_infeasible",
        "stage1_infeasible",
    ]


def test_invalid_design_is_typed_and_never_scored() -> None:
    """Unknown section input returns an explicit class-2 failure without running either stage."""

    root = Path(__file__).resolve().parents[2]
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    execution = evaluator.evaluate(
        evaluation_id="invalid",
        section_names_by_group={
            "1": "W999X999",
            "2": "W16X89",
            "3": "W21X68",
            "4": "W21X62",
            "5": "W21X44",
        },
    )
    result = execution.result
    assert result.status == "invalid_input"
    assert result.failure is not None
    assert result.failure.category is FailureCategory.INPUT_INVALID
    assert result.penalty is None
    assert not result.stage1.executed
    assert not result.stage2.executed


def test_known_stage2_nonconvergence_is_typed_without_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed pushover failure cannot become a finite infeasible structural response."""

    root = Path(__file__).resolve().parents[2]
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    design = json.loads(
        (root / "configs" / "cases" / "phase8_representative_designs.json").read_text(
            encoding="utf-8"
        )
    )["designs"][0]

    def fail_pushover(*args: object, **kwargs: object) -> None:
        """Emulate the typed prefix produced by the production two-pass runner."""

        raise RuntimeError("final pushover failed: controlled test nonconvergence")

    monkeypatch.setattr(evaluator_module, "evaluate_design_pushover", fail_pushover)
    result = evaluator.evaluate(
        evaluation_id="nonconvergent",
        section_names_by_group=design["sections_by_group"],
        use_cache=False,
    ).result
    assert result.status == "numerical_failure"
    assert result.failure is not None
    assert result.failure.category is FailureCategory.NUMERICAL_NONCONVERGENCE
    assert result.stage1.executed
    assert not result.stage2.executed
    assert result.penalty is None
    assert result.objective is None


def test_invalid_hinge_capacity_is_typed_without_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A domain-legal design with no positive hinge capacity becomes a structural failure."""

    root = Path(__file__).resolve().parents[2]
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    design = json.loads(
        (root / "configs" / "cases" / "phase8_representative_designs.json").read_text(
            encoding="utf-8"
        )
    )["designs"][0]

    def fail_capacity(*args: object, **kwargs: object) -> None:
        """Emulate the verified hinge constructor's physical-capacity rejection."""

        raise ValueError("compression eliminates positive column hinge moment capacity")

    monkeypatch.setattr(evaluator_module, "evaluate_design_pushover", fail_capacity)
    result = evaluator.evaluate(
        evaluation_id="invalid-hinge-capacity",
        section_names_by_group=design["sections_by_group"],
        use_cache=False,
    ).result
    assert result.status == "structural_failure"
    assert result.failure is not None
    assert result.failure.category is FailureCategory.STRUCTURAL_INFEASIBILITY
    assert result.stage1.feasible
    assert not result.stage2.executed
    assert result.penalty is None
    assert result.objective is None


def test_cache_reuses_result_but_preserves_request_identity() -> None:
    """Identical designs reuse structural results without leaking the first caller label."""

    root = Path(__file__).resolve().parents[2]
    evaluator = StructuralEvaluator(EvaluatorPaths.from_project_root(root))
    design = json.loads(
        (root / "configs" / "cases" / "phase8_representative_designs.json").read_text(
            encoding="utf-8"
        )
    )["designs"][0]
    first = evaluator.evaluate(
        evaluation_id="first", section_names_by_group=design["sections_by_group"]
    )
    second = evaluator.evaluate(
        evaluation_id="second", section_names_by_group=design["sections_by_group"]
    )
    fresh = evaluator.evaluate(
        evaluation_id="fresh-bypass",
        section_names_by_group=design["sections_by_group"],
        use_cache=False,
    )
    assert not first.cache_hit
    assert second.cache_hit
    assert first.result == second.result
    assert not fresh.cache_hit
    assert fresh.result == first.result
    assert first.request_id == "first"
    assert second.request_id == "second"


def test_configuration_change_invalidates_cache_identity(tmp_path: Path) -> None:
    """Changing even diagnostic policy bytes changes evaluator and design cache identities."""

    root = Path(__file__).resolve().parents[2]
    paths = EvaluatorPaths.from_project_root(root)
    original = StructuralEvaluator(paths)
    content = json.loads(paths.evaluation_policy.read_text(encoding="utf-8"))
    content["objective"]["wmax_research_reference_n"] = 1_974_011.0
    modified_policy = tmp_path / "phase8_policy.json"
    modified_policy.write_text(json.dumps(content), encoding="utf-8")
    modified = StructuralEvaluator(paths.model_copy(update={"evaluation_policy": modified_policy}))
    names = {"1": "W16X77", "2": "W16X89", "3": "W21X68", "4": "W21X62", "5": "W21X44"}
    assert original.identity.evaluation_policy_sha256 != modified.identity.evaluation_policy_sha256
    assert original.cache_key(names) != modified.cache_key(names)


def test_cli_writes_one_structured_evaluation(tmp_path: Path) -> None:
    """The Phase 1 CLI placeholder is replaced by the real deterministic evaluator."""

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "evaluation.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "steel_frame_optimization",
            "evaluate",
            "--project-root",
            str(root),
            "--evaluation-id",
            "cli-case3",
            "--section",
            "1=W16X77",
            "--section",
            "2=W16X89",
            "--section",
            "3=W21X68",
            "--section",
            "4=W21X62",
            "--section",
            "5=W21X44",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    content = json.loads(output.read_text(encoding="utf-8"))
    assert content["request_id"] == "cli-case3"
    assert content["result"]["status"] == "feasible"


def test_committed_phase8_evidence_is_canonical_and_current(
    phase8_artifact: Phase8Artifact,
) -> None:
    """Committed numerical evidence equals the nonstructural handover runtime result."""

    root = Path(__file__).resolve().parents[2]
    path = root / "docs" / "evidence" / "phase08" / "evaluation_pipeline.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    fresh = json.loads(canonical_json_dumps(phase8_artifact))
    for artifact in (persisted, fresh):
        artifact.pop("evaluator_identity")
        for result in artifact["results"]:
            result.pop("cache_key")
            result.pop("evaluator_identity")
            result["stage1"].pop("complete_result_sha256")
            result["stage2"].pop("complete_result_sha256")
    assert persisted == fresh
