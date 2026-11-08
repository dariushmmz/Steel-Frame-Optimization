"""Run deterministic nonlinear static increments with a bounded convergence ladder."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import openseespy.opensees as ops
from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel
from steel_frame_optimization.models.failures import FailureCategory, FailureRecord


class SolverAlgorithm(StrictConfigModel):
    """Declare one supported OpenSees tangent algorithm in deterministic attempt order."""

    name: Literal["Newton", "NewtonLineSearch", "KrylovNewton"]


class NonlinearSolverPolicy(StrictConfigModel):
    """Freeze nonlinear test, algorithm-ladder, and load-step refinement controls."""

    schema_version: Literal["1.0"] = "1.0"
    convergence_test: Literal["NormDispIncr"] = "NormDispIncr"
    tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    max_iterations: int = Field(gt=0)
    print_flag: int = Field(ge=0)
    constraint_handler: Literal["Transformation", "Lagrange", "Penalty"] = "Transformation"
    constraint_penalty_factor: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    algorithms: tuple[SolverAlgorithm, ...] = Field(min_length=1)
    refinement_divisors: tuple[int, ...] = Field(min_length=1)
    time_tolerance: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ladder(self) -> NonlinearSolverPolicy:
        """Require unique algorithms and strictly increasing integer refinement divisors."""

        names = [algorithm.name for algorithm in self.algorithms]
        if len(names) != len(set(names)):
            raise ValueError("nonlinear algorithms must be unique")
        if self.refinement_divisors[0] != 1:
            raise ValueError("refinement divisors must start at one")
        if any(value <= 0 for value in self.refinement_divisors):
            raise ValueError("refinement divisors must be positive")
        if tuple(sorted(set(self.refinement_divisors))) != self.refinement_divisors:
            raise ValueError("refinement divisors must be unique and strictly increasing")
        return self


class SolverAttempt(StrictConfigModel):
    """Record one OpenSees analysis attempt, including failed fallback attempts."""

    nominal_step: int = Field(gt=0)
    refinement_divisor: int = Field(gt=0)
    algorithm: str = Field(min_length=1)
    start_load_factor: float = Field(allow_inf_nan=False)
    requested_increment: float = Field(gt=0.0, allow_inf_nan=False)
    return_code: int
    iterations: int = Field(ge=0)
    final_norm: float | None = Field(default=None, ge=0.0)
    end_load_factor: float = Field(allow_inf_nan=False)


class NonlinearSolveResult(StrictConfigModel):
    """Persist deterministic nonlinear path success or explicit numerical failure."""

    converged: bool
    start_load_factor: float = Field(allow_inf_nan=False)
    target_load_factor: float = Field(allow_inf_nan=False)
    achieved_load_factor: float = Field(allow_inf_nan=False)
    nominal_steps: int = Field(gt=0)
    successful_increments: int = Field(ge=0)
    attempts: tuple[SolverAttempt, ...]
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def validate_result(self) -> NonlinearSolveResult:
        """Require failure details exactly when the requested path did not converge."""

        if self.converged is (self.failure is not None):
            raise ValueError("converged results must omit failure; failed results must include it")
        return self


class DisplacementStep(StrictConfigModel):
    """Record one successfully committed displacement-controlled increment."""

    nominal_step: int = Field(gt=0)
    refinement_divisor: int = Field(gt=0)
    algorithm: str = Field(min_length=1)
    control_displacement: float = Field(allow_inf_nan=False)
    load_factor: float = Field(allow_inf_nan=False)


class DisplacementControlResult(StrictConfigModel):
    """Persist one displacement-controlled path segment and its explicit failure state."""

    converged: bool
    start_displacement: float = Field(allow_inf_nan=False)
    target_displacement: float = Field(allow_inf_nan=False)
    achieved_displacement: float = Field(allow_inf_nan=False)
    successful_increments: int = Field(ge=0)
    attempts: tuple[SolverAttempt, ...]
    committed_steps: tuple[DisplacementStep, ...]
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def validate_result(self) -> DisplacementControlResult:
        """Require failure details exactly when the displacement target was not reached."""

        if self.converged is (self.failure is not None):
            raise ValueError("converged results must omit failure; failed results must include it")
        return self


def _configure_algorithm(algorithm: SolverAlgorithm) -> None:
    """Configure one supported OpenSees solution algorithm."""

    if algorithm.name == "Newton":
        ops.algorithm("Newton")
    elif algorithm.name == "NewtonLineSearch":
        ops.algorithm("NewtonLineSearch", "-type", "Bisection")
    else:
        ops.algorithm("KrylovNewton", "-iterate", "current", "-increment", "current", "-maxDim", 3)


def _configure_constraints(policy: NonlinearSolverPolicy) -> None:
    """Configure the declared exact multi-point-constraint formulation."""

    if policy.constraint_handler == "Lagrange":
        ops.constraints("Lagrange", 1.0, 1.0)
    elif policy.constraint_handler == "Penalty":
        ops.constraints(
            "Penalty",
            policy.constraint_penalty_factor,
            policy.constraint_penalty_factor,
        )
    else:
        ops.constraints("Transformation")


def _last_test_norm() -> float | None:
    """Return the final convergence norm when OpenSees exposes one."""

    raw = ops.testNorm()
    if isinstance(raw, float):
        return abs(raw) if math.isfinite(raw) else None
    values = [abs(float(value)) for value in raw if math.isfinite(float(value))]
    return values[-1] if values else None


def _attempt_increment(
    *, increment: float, algorithm: SolverAlgorithm, policy: NonlinearSolverPolicy
) -> tuple[int, int, float | None]:
    """Try one load increment after rebuilding only the OpenSees analysis objects."""

    ops.wipeAnalysis()
    _configure_constraints(policy)
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test(policy.convergence_test, policy.tolerance, policy.max_iterations, policy.print_flag)
    _configure_algorithm(algorithm)
    ops.integrator("LoadControl", increment)
    ops.analysis("Static")
    return_code = int(ops.analyze(1))
    return return_code, max(int(ops.testIter()), 0), _last_test_norm()


def _attempt_displacement_increment(
    *,
    control_node: int,
    control_dof: int,
    increment: float,
    algorithm: SolverAlgorithm,
    policy: NonlinearSolverPolicy,
) -> tuple[int, int, float | None]:
    """Try one displacement increment after rebuilding only OpenSees analysis objects."""

    ops.wipeAnalysis()
    _configure_constraints(policy)
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test(policy.convergence_test, policy.tolerance, policy.max_iterations, policy.print_flag)
    _configure_algorithm(algorithm)
    ops.integrator("DisplacementControl", control_node, control_dof, increment)
    ops.analysis("Static")
    return_code = int(ops.analyze(1))
    return return_code, max(int(ops.testIter()), 0), _last_test_norm()


def run_static_load_path(
    *, target_load_increment: float, nominal_steps: int, policy: NonlinearSolverPolicy
) -> NonlinearSolveResult:
    """Advance the current OpenSees domain through a bounded, auditable load-control path."""

    if target_load_increment <= 0.0 or not math.isfinite(target_load_increment):
        raise ValueError("target load increment must be finite and positive")
    if nominal_steps <= 0:
        raise ValueError("nominal steps must be positive")
    start = float(ops.getTime())
    target = start + target_load_increment
    base_increment = target_load_increment / nominal_steps
    attempts: list[SolverAttempt] = []
    successes = 0

    for nominal_step in range(1, nominal_steps + 1):
        nominal_target = start + nominal_step * base_increment
        refinement_index = 0
        while nominal_target - float(ops.getTime()) > policy.time_tolerance:
            if refinement_index >= len(policy.refinement_divisors):
                achieved = float(ops.getTime())
                return NonlinearSolveResult(
                    converged=False,
                    start_load_factor=start,
                    target_load_factor=target,
                    achieved_load_factor=achieved,
                    nominal_steps=nominal_steps,
                    successful_increments=successes,
                    attempts=tuple(attempts),
                    failure=FailureRecord(
                        category=FailureCategory.NUMERICAL_NONCONVERGENCE,
                        stage="nonlinear_static_load_path",
                        message="all configured algorithms and refinement levels failed",
                        recoverable=False,
                        details={
                            "nominal_step": nominal_step,
                            "achieved_load_factor": achieved,
                            "target_load_factor": target,
                            "attempt_count": len(attempts),
                        },
                    ),
                )
            divisor = policy.refinement_divisors[refinement_index]
            remaining = nominal_target - float(ops.getTime())
            requested = min(base_increment / divisor, remaining)
            increment_succeeded = False
            for algorithm in policy.algorithms:
                attempt_start = float(ops.getTime())
                return_code, iterations, final_norm = _attempt_increment(
                    increment=requested, algorithm=algorithm, policy=policy
                )
                attempt_end = float(ops.getTime())
                attempts.append(
                    SolverAttempt(
                        nominal_step=nominal_step,
                        refinement_divisor=divisor,
                        algorithm=algorithm.name,
                        start_load_factor=attempt_start,
                        requested_increment=requested,
                        return_code=return_code,
                        iterations=iterations,
                        final_norm=final_norm,
                        end_load_factor=attempt_end,
                    )
                )
                if return_code == 0:
                    successes += 1
                    increment_succeeded = True
                    break
                if not math.isclose(
                    attempt_end, attempt_start, rel_tol=0.0, abs_tol=policy.time_tolerance
                ):
                    raise RuntimeError(
                        "OpenSees advanced time on a failed step; safe fallback is impossible"
                    )
            if increment_succeeded:
                continue
            refinement_index += 1

    achieved = float(ops.getTime())
    if not math.isclose(achieved, target, rel_tol=0.0, abs_tol=policy.time_tolerance):
        raise RuntimeError(f"solver reported success at load factor {achieved}, expected {target}")
    return NonlinearSolveResult(
        converged=True,
        start_load_factor=start,
        target_load_factor=target,
        achieved_load_factor=achieved,
        nominal_steps=nominal_steps,
        successful_increments=successes,
        attempts=tuple(attempts),
    )


def run_displacement_control_path(
    *,
    control_node: int,
    control_dof: int,
    target_displacement_increment: float,
    nominal_displacement_increment: float,
    policy: NonlinearSolverPolicy,
    on_commit: Callable[[DisplacementStep], None] | None = None,
) -> DisplacementControlResult:
    """Advance an OpenSees domain to an exact positive displacement using bounded refinement."""

    if control_node <= 0 or control_dof <= 0:
        raise ValueError("control node and degree of freedom must be positive")
    values = (target_displacement_increment, nominal_displacement_increment)
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("target and nominal displacement increments must be finite and positive")
    start_displacement = float(ops.nodeDisp(control_node, control_dof))
    target_displacement = start_displacement + target_displacement_increment
    nominal_steps = math.ceil(target_displacement_increment / nominal_displacement_increment)
    attempts: list[SolverAttempt] = []
    committed_steps: list[DisplacementStep] = []

    for nominal_step in range(1, nominal_steps + 1):
        nominal_target = min(
            start_displacement + nominal_step * nominal_displacement_increment,
            target_displacement,
        )
        refinement_index = 0
        while (
            nominal_target - float(ops.nodeDisp(control_node, control_dof)) > policy.time_tolerance
        ):
            if refinement_index >= len(policy.refinement_divisors):
                achieved = float(ops.nodeDisp(control_node, control_dof))
                return DisplacementControlResult(
                    converged=False,
                    start_displacement=start_displacement,
                    target_displacement=target_displacement,
                    achieved_displacement=achieved,
                    successful_increments=len(committed_steps),
                    attempts=tuple(attempts),
                    committed_steps=tuple(committed_steps),
                    failure=FailureRecord(
                        category=FailureCategory.NUMERICAL_NONCONVERGENCE,
                        stage="nonlinear_displacement_control_path",
                        message="all configured algorithms and refinement levels failed",
                        recoverable=False,
                        details={
                            "nominal_step": nominal_step,
                            "achieved_displacement": achieved,
                            "target_displacement": target_displacement,
                            "attempt_count": len(attempts),
                        },
                    ),
                )
            divisor = policy.refinement_divisors[refinement_index]
            current_displacement = float(ops.nodeDisp(control_node, control_dof))
            remaining = nominal_target - current_displacement
            requested = min(nominal_displacement_increment / divisor, remaining)
            increment_succeeded = False
            for algorithm in policy.algorithms:
                attempt_start_time = float(ops.getTime())
                attempt_start_displacement = float(ops.nodeDisp(control_node, control_dof))
                return_code, iterations, final_norm = _attempt_displacement_increment(
                    control_node=control_node,
                    control_dof=control_dof,
                    increment=requested,
                    algorithm=algorithm,
                    policy=policy,
                )
                attempt_end_time = float(ops.getTime())
                attempt_end_displacement = float(ops.nodeDisp(control_node, control_dof))
                attempts.append(
                    SolverAttempt(
                        nominal_step=nominal_step,
                        refinement_divisor=divisor,
                        algorithm=algorithm.name,
                        start_load_factor=attempt_start_time,
                        requested_increment=requested,
                        return_code=return_code,
                        iterations=iterations,
                        final_norm=final_norm,
                        end_load_factor=attempt_end_time,
                    )
                )
                if return_code == 0:
                    step = DisplacementStep(
                        nominal_step=nominal_step,
                        refinement_divisor=divisor,
                        algorithm=algorithm.name,
                        control_displacement=attempt_end_displacement,
                        load_factor=attempt_end_time,
                    )
                    committed_steps.append(step)
                    if on_commit is not None:
                        on_commit(step)
                    increment_succeeded = True
                    break
                if not math.isclose(
                    attempt_end_time,
                    attempt_start_time,
                    rel_tol=0.0,
                    abs_tol=policy.time_tolerance,
                ) or not math.isclose(
                    attempt_end_displacement,
                    attempt_start_displacement,
                    rel_tol=0.0,
                    abs_tol=policy.time_tolerance,
                ):
                    raise RuntimeError(
                        "OpenSees changed committed state on a failed displacement step"
                    )
            if increment_succeeded:
                continue
            refinement_index += 1

    achieved = float(ops.nodeDisp(control_node, control_dof))
    if not math.isclose(achieved, target_displacement, rel_tol=0.0, abs_tol=policy.time_tolerance):
        raise RuntimeError(
            f"solver reported success at displacement {achieved}, expected {target_displacement}"
        )
    return DisplacementControlResult(
        converged=True,
        start_displacement=start_displacement,
        target_displacement=target_displacement,
        achieved_displacement=achieved,
        successful_increments=len(committed_steps),
        attempts=tuple(attempts),
        committed_steps=tuple(committed_steps),
    )
