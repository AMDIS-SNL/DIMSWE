"""Dual-native exact discrete HVPs for a tiny Firedrake RK problem.

This module is deliberately independent of the DIMSWE timesteppers.  It
implements explicit Runge--Kutta stages for

    u_t = kappa * Delta(u) + p**2 * u

using weak mass solves, copied Firedrake caches, ordinary reverse stages, and
their exact directional reverse.  All state objects are primal ``Function``
instances and all adjoints are dual ``Cofunction`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

from firedrake import (
    Cofunction,
    Constant,
    Function,
    LinearSolver,
    TestFunction,
    TrialFunction,
    action,
    assemble,
    derivative,
    dx,
    grad,
    inner,
)


_DIRECT_SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}


def _as_float(name: str, value: Real) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    return float(value)


def _copy_function(value: Function, name: str) -> Function:
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _copy_cofunction(value: Cofunction, name: str) -> Cofunction:
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def dual_pairing(dual: Cofunction, primal: Function) -> float:
    """Evaluate the natural pairing ``<dual, primal>``.

    The UFL action preserves the distinction between a dual coefficient vector
    and a primal L2 Riesz representative.
    """
    if not isinstance(dual, Cofunction):
        raise TypeError("dual must be a Firedrake Cofunction")
    if not isinstance(primal, Function):
        raise TypeError("primal must be a Firedrake Function")
    if dual.function_space().dual() != primal.function_space():
        raise ValueError("dual and primal spaces are incompatible")
    return float(assemble(action(dual, primal)))


@dataclass(frozen=True)
class ButcherTableau:
    """An explicit Runge--Kutta tableau."""

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    name: str

    def __post_init__(self) -> None:
        nstages = len(self.b)
        if nstages == 0 or len(self.a) != nstages or len(self.c) != nstages:
            raise ValueError("tableau dimensions do not agree")
        if any(len(row) != nstages for row in self.a):
            raise ValueError("tableau a must be square")
        for i, row in enumerate(self.a):
            if any(row[j] != 0.0 for j in range(i, nstages)):
                raise ValueError("tableau must be explicit")

    @property
    def nstages(self) -> int:
        return len(self.b)


EULER = ButcherTableau(
    a=((0.0,),),
    b=(1.0,),
    c=(0.0,),
    name="Euler",
)


CLASSICAL_RK4 = ButcherTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    ),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 0.5, 1.0),
    name="classical RK4",
)


@dataclass(frozen=True)
class PrimalStageCache:
    stage_state: Function
    stage_tendency: Function


@dataclass(frozen=True)
class TangentStageCache:
    stage_state_direction: Function
    stage_tendency_direction: Function


@dataclass(frozen=True)
class PrimalStepCache:
    dt: float
    parameter: float
    state_in: Function
    state_out: Function
    stages: tuple[PrimalStageCache, ...]


@dataclass(frozen=True)
class TangentStepCache:
    primal: PrimalStepCache
    parameter_direction: float
    state_direction_in: Function
    state_direction_out: Function
    stages: tuple[TangentStageCache, ...]


@dataclass(frozen=True)
class ReverseStageData:
    tendency_adjoint: Cofunction
    auxiliary: Function
    state_adjoint: Cofunction
    parameter_gradient: float

    @property
    def bar_k_star(self) -> Cofunction:
        return self.tendency_adjoint

    @property
    def psi(self) -> Function:
        return self.auxiliary

    @property
    def lambda_u_stage_star(self) -> Cofunction:
        return self.state_adjoint


@dataclass(frozen=True)
class IncrementalReverseStageData:
    incremental_tendency_adjoint: Cofunction
    incremental_auxiliary: Function
    incremental_state_adjoint: Cofunction
    parameter_hvp: float

    @property
    def delta_bar_k_star(self) -> Cofunction:
        return self.incremental_tendency_adjoint

    @property
    def delta_psi(self) -> Function:
        return self.incremental_auxiliary

    @property
    def mu_u_stage_star(self) -> Cofunction:
        return self.incremental_state_adjoint


@dataclass(frozen=True)
class ReverseStepResult:
    state_adjoint_in: Cofunction
    parameter_gradient: float
    stages: tuple[ReverseStageData, ...]


@dataclass(frozen=True)
class HVPReverseStepResult:
    ordinary: ReverseStepResult
    incremental_state_adjoint_in: Cofunction
    parameter_hvp: float
    incremental_stages: tuple[IncrementalReverseStageData, ...]


@dataclass(frozen=True)
class PrimalTrajectory:
    states: tuple[Function, ...]
    steps: tuple[PrimalStepCache, ...]


@dataclass(frozen=True)
class TangentTrajectory:
    primal_states: tuple[Function, ...]
    state_directions: tuple[Function, ...]
    steps: tuple[TangentStepCache, ...]


@dataclass(frozen=True)
class GradientResult:
    objective: float
    states: tuple[Function, ...]
    terminal_adjoint: Cofunction
    initial_state_adjoint: Cofunction
    parameter_gradient: float
    reverse_steps: tuple[ReverseStepResult, ...]

    @property
    def lambda_plus_star(self) -> Cofunction:
        return self.terminal_adjoint

    @property
    def lambda_u_star(self) -> Cofunction:
        return self.initial_state_adjoint


@dataclass(frozen=True)
class HVPResult:
    objective: float
    states: tuple[Function, ...]
    state_directions: tuple[Function, ...]
    terminal_adjoint: Cofunction
    terminal_incremental_adjoint: Cofunction
    initial_state_adjoint: Cofunction
    initial_incremental_state_adjoint: Cofunction
    parameter_gradient: float
    parameter_hvp: float
    tangent_steps: tuple[TangentStepCache, ...]
    reverse_steps: tuple[HVPReverseStepResult, ...]

    @property
    def lambda_plus_star(self) -> Cofunction:
        return self.terminal_adjoint

    @property
    def mu_plus_star(self) -> Cofunction:
        return self.terminal_incremental_adjoint

    @property
    def lambda_u_star(self) -> Cofunction:
        return self.initial_state_adjoint

    @property
    def mu_u_star(self) -> Cofunction:
        return self.initial_incremental_state_adjoint


class WeakStageModel:
    """Weak stage operator with an explicit L2 mass solve.

    The stage equation is

    ``(eta, K) = p**2 (eta, Y) - kappa (grad(eta), grad(Y))``.

    Homogeneous boundary conditions, if supplied, are applied to primal
    fields, the assembled mass matrix, and assembled one-forms.  The latter is
    important because this Firedrake version does not accept a ``Cofunction``
    in ``DirichletBC.apply``.
    """

    def __init__(
        self,
        function_space,
        *,
        kappa: Real = 0.0,
        bcs: Sequence | None = None,
        solver_parameters: dict | None = None,
    ) -> None:
        self.function_space = function_space
        self.dual_space = function_space.dual()
        self.kappa = _as_float("kappa", kappa)
        if self.kappa < 0.0:
            raise ValueError("kappa must be nonnegative")
        self.bcs = tuple(() if bcs is None else bcs)
        self._test = TestFunction(function_space)
        self._trial = TrialFunction(function_space)
        mass_matrix = assemble(
            inner(self._test, self._trial) * dx,
            bcs=self.bcs or None,
            mat_type="aij",
        )
        parameters = dict(_DIRECT_SOLVER_PARAMETERS)
        if solver_parameters is not None:
            parameters.update(solver_parameters)
        self._mass_solver = LinearSolver(
            mass_matrix,
            solver_parameters=parameters,
        )

    def _require_primal(self, name: str, value: Function) -> None:
        if not isinstance(value, Function):
            raise TypeError(f"{name} must be a Firedrake Function")
        if value.function_space() != self.function_space:
            raise ValueError(f"{name} belongs to the wrong primal space")

    def _require_dual(self, name: str, value: Cofunction) -> None:
        if not isinstance(value, Cofunction):
            raise TypeError(f"{name} must be a Firedrake Cofunction")
        if value.function_space() != self.dual_space:
            raise ValueError(f"{name} belongs to the wrong dual space")

    def apply_boundary_conditions(self, value: Function) -> Function:
        """Apply the model's homogeneous BCs to an owned primal object."""
        self._require_primal("value", value)
        for bc in self.bcs:
            bc.apply(value)
        return value

    def _assemble_dual(self, form, name: str) -> Cofunction:
        result = assemble(form, bcs=self.bcs or None)
        if not isinstance(result, Cofunction):
            raise TypeError("expected one-form assembly to return a Cofunction")
        result.rename(name)
        return result

    def mass_map(self, value: Function, *, name: str = "mass_map") -> Cofunction:
        """Apply the L2 Riesz map ``M: V_h -> V_h*``."""
        self._require_primal("value", value)
        return self._assemble_dual(inner(self._test, value) * dx, name)

    def l2_riesz_representative(
        self,
        dual: Cofunction,
        *,
        name: str = "l2_riesz_representative",
    ) -> Function:
        """Solve ``M z = dual`` without forming an inverse mass matrix."""
        self._require_dual("dual", dual)
        # Own the RHS so a solver implementation can never mutate caller data.
        rhs = _copy_cofunction(dual, f"{name}_rhs")
        result = Function(self.function_space, name=name)
        self._mass_solver.solve(result, rhs)
        self.apply_boundary_conditions(result)
        return result

    def _stage_rhs_form(self, stage_state: Function, parameter: Constant):
        form = parameter * parameter * inner(self._test, stage_state) * dx
        if self.kappa:
            form -= (
                self.kappa
                * inner(grad(self._test), grad(stage_state))
                * dx
            )
        return form

    def _contracted_stage_form(
        self,
        stage_state: Function,
        parameter: Constant,
        auxiliary: Function,
    ):
        form = parameter * parameter * inner(auxiliary, stage_state) * dx
        if self.kappa:
            form -= (
                self.kappa
                * inner(grad(auxiliary), grad(stage_state))
                * dx
            )
        return form

    def solve_stage(
        self,
        stage_state: Function,
        parameter: Real,
        *,
        name: str = "stage_tendency",
    ) -> Function:
        """Solve the primal weak stage equation."""
        self._require_primal("stage_state", stage_state)
        p = Constant(_as_float("parameter", parameter))
        rhs = self._assemble_dual(
            self._stage_rhs_form(stage_state, p),
            f"{name}_rhs",
        )
        return self.l2_riesz_representative(rhs, name=name)

    def solve_tangent_stage(
        self,
        stage_state: Function,
        parameter: Real,
        stage_state_direction: Function,
        parameter_direction: Real,
        *,
        name: str = "stage_tendency_direction",
    ) -> Function:
        """Differentiate the weak primal stage and solve its tangent."""
        self._require_primal("stage_state", stage_state)
        self._require_primal("stage_state_direction", stage_state_direction)
        p = Constant(_as_float("parameter", parameter))
        q = Constant(_as_float("parameter_direction", parameter_direction))
        primal_rhs = self._stage_rhs_form(stage_state, p)
        tangent_rhs_form = (
            derivative(primal_rhs, stage_state, stage_state_direction)
            + derivative(primal_rhs, p, q)
        )
        tangent_rhs = self._assemble_dual(tangent_rhs_form, f"{name}_rhs")
        return self.l2_riesz_representative(tangent_rhs, name=name)

    def reverse_stage(
        self,
        stage_state: Function,
        parameter: Real,
        tendency_adjoint: Cofunction,
        *,
        stage_index: int,
    ) -> ReverseStageData:
        """Apply one ordinary reverse stage with a primal mass solve."""
        self._require_primal("stage_state", stage_state)
        self._require_dual("tendency_adjoint", tendency_adjoint)
        p = Constant(_as_float("parameter", parameter))
        psi = self.l2_riesz_representative(
            tendency_adjoint,
            name=f"stage_{stage_index}_auxiliary",
        )
        contracted = self._contracted_stage_form(stage_state, p, psi)
        state_adjoint = self._assemble_dual(
            derivative(contracted, stage_state),
            f"stage_{stage_index}_state_adjoint",
        )
        parameter_gradient = float(
            assemble(derivative(contracted, p, Constant(1.0)))
        )
        return ReverseStageData(
            tendency_adjoint=_copy_cofunction(
                tendency_adjoint,
                f"stage_{stage_index}_tendency_adjoint",
            ),
            auxiliary=psi,
            state_adjoint=state_adjoint,
            parameter_gradient=parameter_gradient,
        )

    def incremental_reverse_stage(
        self,
        stage_state: Function,
        stage_state_direction: Function,
        parameter: Real,
        parameter_direction: Real,
        ordinary_auxiliary: Function,
        incremental_tendency_adjoint: Cofunction,
        *,
        stage_index: int,
    ) -> IncrementalReverseStageData:
        """Apply one exact incremental reverse stage using contracted UFL forms."""
        self._require_primal("stage_state", stage_state)
        self._require_primal("stage_state_direction", stage_state_direction)
        self._require_primal("ordinary_auxiliary", ordinary_auxiliary)
        self._require_dual(
            "incremental_tendency_adjoint",
            incremental_tendency_adjoint,
        )
        p = Constant(_as_float("parameter", parameter))
        q = Constant(_as_float("parameter_direction", parameter_direction))
        delta_psi = self.l2_riesz_representative(
            incremental_tendency_adjoint,
            name=f"stage_{stage_index}_incremental_auxiliary",
        )

        ordinary_contracted = self._contracted_stage_form(
            stage_state,
            p,
            ordinary_auxiliary,
        )
        incremental_contracted = self._contracted_stage_form(
            stage_state,
            p,
            delta_psi,
        )

        ordinary_state_pullback = derivative(
            ordinary_contracted,
            stage_state,
        )
        incremental_state_form = (
            derivative(incremental_contracted, stage_state)
            + derivative(
                ordinary_state_pullback,
                stage_state,
                stage_state_direction,
            )
            + derivative(ordinary_state_pullback, p, q)
        )
        incremental_state_adjoint = self._assemble_dual(
            incremental_state_form,
            f"stage_{stage_index}_incremental_state_adjoint",
        )

        ordinary_parameter_pullback = derivative(
            ordinary_contracted,
            p,
            Constant(1.0),
        )
        parameter_hvp_form = (
            derivative(incremental_contracted, p, Constant(1.0))
            + derivative(
                ordinary_parameter_pullback,
                stage_state,
                stage_state_direction,
            )
            + derivative(ordinary_parameter_pullback, p, q)
        )
        parameter_hvp = float(assemble(parameter_hvp_form))

        return IncrementalReverseStageData(
            incremental_tendency_adjoint=_copy_cofunction(
                incremental_tendency_adjoint,
                f"stage_{stage_index}_incremental_tendency_adjoint",
            ),
            incremental_auxiliary=delta_psi,
            incremental_state_adjoint=incremental_state_adjoint,
            parameter_hvp=parameter_hvp,
        )


class ExplicitRungeKutta:
    """One generic copied-cache algorithm for Euler and classical RK4."""

    def __init__(self, model: WeakStageModel, tableau: ButcherTableau) -> None:
        if not isinstance(model, WeakStageModel):
            raise TypeError("model must be a WeakStageModel")
        if not isinstance(tableau, ButcherTableau):
            raise TypeError("tableau must be a ButcherTableau")
        self.model = model
        self.tableau = tableau

    def _primal_sum(
        self,
        base: Function,
        terms: Iterable[tuple[float, Function]],
        name: str,
    ) -> Function:
        self.model._require_primal("base", base)
        result = _copy_function(base, name)
        for scale, value in terms:
            self.model._require_primal("summand", value)
            result += float(scale) * value
        self.model.apply_boundary_conditions(result)
        return result

    def _dual_sum(
        self,
        terms: Iterable[tuple[float, Cofunction]],
        name: str,
    ) -> Cofunction:
        result = Cofunction(self.model.dual_space, name=name)
        result.zero()
        for scale, value in terms:
            self.model._require_dual("dual summand", value)
            # Scaled Cofunction arithmetic otherwise promotes to a symbolic
            # FormSum in this Firedrake version.  AXPY preserves the concrete
            # dual object and does not mutate the summand.
            with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
                result_vec.axpy(float(scale), value_vec)
        return result

    def forward_step(
        self,
        state: Function,
        parameter: Real,
        dt: Real,
    ) -> PrimalStepCache:
        self.model._require_primal("state", state)
        p = _as_float("parameter", parameter)
        step_size = _as_float("dt", dt)
        stage_caches: list[PrimalStageCache] = []

        for i in range(self.tableau.nstages):
            stage_state = self._primal_sum(
                state,
                (
                    (step_size * self.tableau.a[i][j], stage_caches[j].stage_tendency)
                    for j in range(i)
                ),
                f"stage_{i}_state",
            )
            tendency = self.model.solve_stage(
                stage_state,
                p,
                name=f"stage_{i}_tendency",
            )
            stage_caches.append(
                PrimalStageCache(
                    stage_state=_copy_function(stage_state, f"stage_{i}_state_cache"),
                    stage_tendency=_copy_function(
                        tendency,
                        f"stage_{i}_tendency_cache",
                    ),
                )
            )

        state_out = self._primal_sum(
            state,
            (
                (step_size * self.tableau.b[i], stage_caches[i].stage_tendency)
                for i in range(self.tableau.nstages)
            ),
            "state_out",
        )
        return PrimalStepCache(
            dt=step_size,
            parameter=p,
            state_in=_copy_function(state, "state_in_cache"),
            state_out=_copy_function(state_out, "state_out_cache"),
            stages=tuple(stage_caches),
        )

    def linearize_step(
        self,
        state: Function,
        parameter: Real,
        state_direction: Function,
        parameter_direction: Real,
        dt: Real,
    ) -> TangentStepCache:
        primal = self.forward_step(state, parameter, dt)
        return self.linearize_cached_step(
            primal,
            state_direction,
            parameter_direction,
        )

    def linearize_cached_step(
        self,
        primal: PrimalStepCache,
        state_direction: Function,
        parameter_direction: Real,
    ) -> TangentStepCache:
        if not isinstance(primal, PrimalStepCache):
            raise TypeError("primal must be a PrimalStepCache")
        self.model._require_primal("state_direction", state_direction)
        q = _as_float("parameter_direction", parameter_direction)
        tangent_stages: list[TangentStageCache] = []

        for i in range(self.tableau.nstages):
            stage_direction = self._primal_sum(
                state_direction,
                (
                    (
                        primal.dt * self.tableau.a[i][j],
                        tangent_stages[j].stage_tendency_direction,
                    )
                    for j in range(i)
                ),
                f"stage_{i}_state_direction",
            )
            tendency_direction = self.model.solve_tangent_stage(
                primal.stages[i].stage_state,
                primal.parameter,
                stage_direction,
                q,
                name=f"stage_{i}_tendency_direction",
            )
            tangent_stages.append(
                TangentStageCache(
                    stage_state_direction=_copy_function(
                        stage_direction,
                        f"stage_{i}_state_direction_cache",
                    ),
                    stage_tendency_direction=_copy_function(
                        tendency_direction,
                        f"stage_{i}_tendency_direction_cache",
                    ),
                )
            )

        direction_out = self._primal_sum(
            state_direction,
            (
                (
                    primal.dt * self.tableau.b[i],
                    tangent_stages[i].stage_tendency_direction,
                )
                for i in range(self.tableau.nstages)
            ),
            "state_direction_out",
        )
        return TangentStepCache(
            primal=primal,
            parameter_direction=q,
            state_direction_in=_copy_function(
                state_direction,
                "state_direction_in_cache",
            ),
            state_direction_out=_copy_function(
                direction_out,
                "state_direction_out_cache",
            ),
            stages=tuple(tangent_stages),
        )

    def reverse_step(
        self,
        step: PrimalStepCache,
        state_adjoint_out: Cofunction,
    ) -> ReverseStepResult:
        if not isinstance(step, PrimalStepCache):
            raise TypeError("step must be a PrimalStepCache")
        self.model._require_dual("state_adjoint_out", state_adjoint_out)
        reverse_stages: list[ReverseStageData | None] = [
            None for _ in range(self.tableau.nstages)
        ]
        gradient = 0.0

        for i in range(self.tableau.nstages - 1, -1, -1):
            terms: list[tuple[float, Cofunction]] = [
                (step.dt * self.tableau.b[i], state_adjoint_out)
            ]
            for j in range(i + 1, self.tableau.nstages):
                later = reverse_stages[j]
                assert later is not None
                terms.append(
                    (
                        step.dt * self.tableau.a[j][i],
                        later.state_adjoint,
                    )
                )
            tendency_adjoint = self._dual_sum(
                terms,
                f"stage_{i}_tendency_adjoint_sum",
            )
            stage_result = self.model.reverse_stage(
                step.stages[i].stage_state,
                step.parameter,
                tendency_adjoint,
                stage_index=i,
            )
            reverse_stages[i] = stage_result
            gradient += stage_result.parameter_gradient

        completed_stages = tuple(stage for stage in reverse_stages if stage is not None)
        state_adjoint_in = self._dual_sum(
            [(1.0, state_adjoint_out)]
            + [(1.0, stage.state_adjoint) for stage in completed_stages],
            "state_adjoint_in",
        )
        return ReverseStepResult(
            state_adjoint_in=state_adjoint_in,
            parameter_gradient=gradient,
            stages=completed_stages,
        )

    def reverse_hvp_step(
        self,
        step: TangentStepCache,
        state_adjoint_out: Cofunction,
        incremental_state_adjoint_out: Cofunction,
    ) -> HVPReverseStepResult:
        if not isinstance(step, TangentStepCache):
            raise TypeError("step must be a TangentStepCache")
        ordinary = self.reverse_step(step.primal, state_adjoint_out)
        self.model._require_dual(
            "incremental_state_adjoint_out",
            incremental_state_adjoint_out,
        )
        incremental_stages: list[IncrementalReverseStageData | None] = [
            None for _ in range(self.tableau.nstages)
        ]
        parameter_hvp = 0.0

        for i in range(self.tableau.nstages - 1, -1, -1):
            terms: list[tuple[float, Cofunction]] = [
                (
                    step.primal.dt * self.tableau.b[i],
                    incremental_state_adjoint_out,
                )
            ]
            for j in range(i + 1, self.tableau.nstages):
                later = incremental_stages[j]
                assert later is not None
                terms.append(
                    (
                        step.primal.dt * self.tableau.a[j][i],
                        later.incremental_state_adjoint,
                    )
                )
            incremental_tendency_adjoint = self._dual_sum(
                terms,
                f"stage_{i}_incremental_tendency_adjoint_sum",
            )
            stage_result = self.model.incremental_reverse_stage(
                step.primal.stages[i].stage_state,
                step.stages[i].stage_state_direction,
                step.primal.parameter,
                step.parameter_direction,
                ordinary.stages[i].auxiliary,
                incremental_tendency_adjoint,
                stage_index=i,
            )
            incremental_stages[i] = stage_result
            parameter_hvp += stage_result.parameter_hvp

        completed_stages = tuple(
            stage for stage in incremental_stages if stage is not None
        )
        incremental_state_adjoint_in = self._dual_sum(
            [(1.0, incremental_state_adjoint_out)]
            + [
                (1.0, stage.incremental_state_adjoint)
                for stage in completed_stages
            ],
            "incremental_state_adjoint_in",
        )
        return HVPReverseStepResult(
            ordinary=ordinary,
            incremental_state_adjoint_in=incremental_state_adjoint_in,
            parameter_hvp=parameter_hvp,
            incremental_stages=completed_stages,
        )

    def forward(
        self,
        nsteps: int,
        state_initial: Function,
        parameter: Real,
        dt: Real,
    ) -> PrimalTrajectory:
        if not isinstance(nsteps, int) or nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        self.model._require_primal("state_initial", state_initial)
        states = [_copy_function(state_initial, "state_0")]
        steps: list[PrimalStepCache] = []
        current = _copy_function(state_initial, "current_state")
        for n in range(nsteps):
            step = self.forward_step(current, parameter, dt)
            steps.append(step)
            current = _copy_function(step.state_out, f"current_state_{n + 1}")
            states.append(_copy_function(current, f"state_{n + 1}"))
        return PrimalTrajectory(states=tuple(states), steps=tuple(steps))

    def linearize(
        self,
        nsteps: int,
        state_initial: Function,
        parameter: Real,
        state_direction_initial: Function,
        parameter_direction: Real,
        dt: Real,
    ) -> TangentTrajectory:
        if not isinstance(nsteps, int) or nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        self.model._require_primal("state_initial", state_initial)
        self.model._require_primal(
            "state_direction_initial",
            state_direction_initial,
        )
        primal_states = [_copy_function(state_initial, "state_0")]
        directions = [
            _copy_function(state_direction_initial, "state_direction_0")
        ]
        steps: list[TangentStepCache] = []
        current = _copy_function(state_initial, "current_state")
        current_direction = _copy_function(
            state_direction_initial,
            "current_state_direction",
        )
        for n in range(nsteps):
            step = self.linearize_step(
                current,
                parameter,
                current_direction,
                parameter_direction,
                dt,
            )
            steps.append(step)
            current = _copy_function(
                step.primal.state_out,
                f"current_state_{n + 1}",
            )
            current_direction = _copy_function(
                step.state_direction_out,
                f"current_state_direction_{n + 1}",
            )
            primal_states.append(_copy_function(current, f"state_{n + 1}"))
            directions.append(
                _copy_function(current_direction, f"state_direction_{n + 1}")
            )
        return TangentTrajectory(
            primal_states=tuple(primal_states),
            state_directions=tuple(directions),
            steps=tuple(steps),
        )


def _residual(model: WeakStageModel, state: Function, target: Function) -> Function:
    model._require_primal("state", state)
    model._require_primal("target", target)
    residual = _copy_function(state, "terminal_residual")
    residual -= target
    return residual


def terminal_least_squares_objective(
    timestepper: ExplicitRungeKutta,
    nsteps: int,
    state_initial: Function,
    parameter: Real,
    dt: Real,
    target: Function,
) -> float:
    """Compute the scalar objective using a primal pass only."""
    trajectory = timestepper.forward(nsteps, state_initial, parameter, dt)
    residual = _residual(timestepper.model, trajectory.states[-1], target)
    return 0.5 * float(assemble(inner(residual, residual) * dx))


def terminal_least_squares_gradient(
    timestepper: ExplicitRungeKutta,
    nsteps: int,
    state_initial: Function,
    parameter: Real,
    dt: Real,
    target: Function,
) -> GradientResult:
    """Compute an ordinary discrete gradient independently of the HVP path."""
    trajectory = timestepper.forward(nsteps, state_initial, parameter, dt)
    residual = _residual(timestepper.model, trajectory.states[-1], target)
    objective = 0.5 * float(assemble(inner(residual, residual) * dx))
    terminal_adjoint = timestepper.model.mass_map(
        residual,
        name="terminal_adjoint",
    )
    current_adjoint = _copy_cofunction(
        terminal_adjoint,
        "current_state_adjoint",
    )
    gradient = 0.0
    reverse_steps: list[ReverseStepResult] = []
    for step in reversed(trajectory.steps):
        reverse = timestepper.reverse_step(step, current_adjoint)
        reverse_steps.append(reverse)
        gradient += reverse.parameter_gradient
        current_adjoint = _copy_cofunction(
            reverse.state_adjoint_in,
            "current_state_adjoint",
        )
    return GradientResult(
        objective=objective,
        states=trajectory.states,
        terminal_adjoint=terminal_adjoint,
        initial_state_adjoint=current_adjoint,
        parameter_gradient=gradient,
        reverse_steps=tuple(reverse_steps),
    )


def terminal_least_squares_hvp(
    timestepper: ExplicitRungeKutta,
    nsteps: int,
    state_initial: Function,
    parameter: Real,
    dt: Real,
    target: Function,
    parameter_direction: Real,
    state_direction_initial: Function | None = None,
) -> HVPResult:
    """Compute the exact parameter HVP for a combined ``(w, q)`` direction."""
    if state_direction_initial is None:
        state_direction_initial = Function(
            timestepper.model.function_space,
            name="zero_initial_state_direction",
        )
    tangent = timestepper.linearize(
        nsteps,
        state_initial,
        parameter,
        state_direction_initial,
        parameter_direction,
        dt,
    )
    residual = _residual(timestepper.model, tangent.primal_states[-1], target)
    objective = 0.5 * float(assemble(inner(residual, residual) * dx))
    terminal_adjoint = timestepper.model.mass_map(
        residual,
        name="terminal_adjoint",
    )
    terminal_incremental_adjoint = timestepper.model.mass_map(
        tangent.state_directions[-1],
        name="terminal_incremental_adjoint",
    )
    current_adjoint = _copy_cofunction(
        terminal_adjoint,
        "current_state_adjoint",
    )
    current_incremental_adjoint = _copy_cofunction(
        terminal_incremental_adjoint,
        "current_incremental_state_adjoint",
    )
    gradient = 0.0
    parameter_hvp = 0.0
    reverse_steps: list[HVPReverseStepResult] = []
    for step in reversed(tangent.steps):
        reverse = timestepper.reverse_hvp_step(
            step,
            current_adjoint,
            current_incremental_adjoint,
        )
        reverse_steps.append(reverse)
        gradient += reverse.ordinary.parameter_gradient
        parameter_hvp += reverse.parameter_hvp
        current_adjoint = _copy_cofunction(
            reverse.ordinary.state_adjoint_in,
            "current_state_adjoint",
        )
        current_incremental_adjoint = _copy_cofunction(
            reverse.incremental_state_adjoint_in,
            "current_incremental_state_adjoint",
        )

    return HVPResult(
        objective=objective,
        states=tangent.primal_states,
        state_directions=tangent.state_directions,
        terminal_adjoint=terminal_adjoint,
        terminal_incremental_adjoint=terminal_incremental_adjoint,
        initial_state_adjoint=current_adjoint,
        initial_incremental_state_adjoint=current_incremental_adjoint,
        parameter_gradient=gradient,
        parameter_hvp=parameter_hvp,
        tangent_steps=tangent.steps,
        reverse_steps=tuple(reverse_steps),
    )
