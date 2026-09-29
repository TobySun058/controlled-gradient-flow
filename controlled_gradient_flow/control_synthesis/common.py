import dataclasses
from typing import TypeAlias, Callable, Tuple, Optional

import diffrax as dfx
import equinox as eqx
import jax
from diffrax import diffeqsolve
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike

from .models import AbstractControlAffineSystem
from .numeric import LinearSolver, make_linear_solver, make_sampled_integrator

ControlFn: TypeAlias = Callable[[float, ArrayLike], ArrayLike]


@dataclasses.dataclass
class ControlObjective:
    """
    A compact data class that make the synthesis process more tractable and organized.

    It contains the control problem formulation, basic numerical methods (ODE solver, sampled integrator,
    linear solver) and the heavily reused helper functions.
    """

    system: AbstractControlAffineSystem
    x0: Array
    x1: Array
    t0: float
    t1: float
    ODE_solver_wrapper: Callable[[dfx.ODETerm, dfx.SaveAt, Tuple, Array, float, float], dfx.Solution]
    forward: bool
    tol_x: Optional[float]
    tol_u: Optional[float]
    min_iterations: int
    max_iterations: int
    linear_solver: LinearSolver
    linear_solver_eps: float
    integrator: Callable[[Array, Array, int], Array]
    integration_ts: Array
    seed_control: ControlFn
    JPhiB_fn: Callable[[float, Array], Array]
    target_y: Array
    record_ts: Array
    interpolation_ts: Optional[Array]


def _make_fl_min_energy_seed_pendulum(
    system: AbstractControlAffineSystem,
    init_state: Array,
    target_state: Array,
    t0: float,
    t1: float,
    eps: float = 1e-18,
) -> ControlFn:
    if system.state_dim != 2:
        raise ValueError('seed_control="fl_min_energy" assumes state_dim=2.')
    if system.input_dim != 1:
        raise ValueError('seed_control="fl_min_energy" assumes input_dim=1.')

    theta0, omega0 = init_state
    thetaT, omegaT = target_state
    T = t1 - t0
    if T <= 0:
        raise ValueError("Require t1 > t0 for fl_min_energy seed.")

    dtheta = thetaT - theta0
    domega = omegaT - omega0

    a = -(12.0 / T**3) * (dtheta - 0.5 * T * (omega0 + omegaT))
    b = (domega / T) - 0.5 * a * T

    def v_ref(t: float) -> Array:
        s = t - t0
        return a * s + b

    def omega_ref(t: float) -> Array:
        s = t - t0
        return omega0 + 0.5 * a * s**2 + b * s

    def theta_ref(t: float) -> Array:
        s = t - t0
        return theta0 + omega0 * s + (a * s**3) / 6.0 + (b * s**2) / 2.0

    def x_ref(t: float) -> Array:
        return jnp.stack([theta_ref(t), omega_ref(t)])

    def xdot_ref(t: float) -> Array:
        return jnp.stack([omega_ref(t), v_ref(t)])

    def u_seed(t: float) -> Array:
        xt = x_ref(t)
        r = xdot_ref(t) - system.N_fn(t, xt)
        Bt = system.B_fn(t, xt).reshape((system.state_dim, system.input_dim))
        bcol = Bt[:, 0]
        u = (bcol @ r) / (bcol @ bcol + eps)
        return jnp.asarray([u], dtype=init_state.dtype)

    return u_seed


def setup_control_synthesis(
    system: AbstractControlAffineSystem,
    init_state: Array,
    target_state: Array,
    t0: float,
    t1: float,
    ODE_solver_wrapper: Callable,
    seed_control: ControlFn | ArrayLike | None,
    tol_x: Optional[float],
    tol_u: Optional[float],
    forward_mode: bool,
    integration_method: str | Callable[[Array, Array, int], Array] = "simpsons",
    integration_samples: int = 1000,
    max_iterations: int = 30,
    min_iterations: int = 10,
    linear_solver_method: str | LinearSolver = "cholesky",
    linear_solver_eps: float = 0.0,
    record_samples: Optional[int] = None,
    interpolation_samples: Optional[int] = None,
) -> ControlObjective:
    linear_solver = make_linear_solver(linear_solver_method)
    integrator = make_sampled_integrator(integration_method)
    linear_solver = jax.jit(linear_solver)
    integrator = jax.jit(integrator)

    integration_ts = jnp.linspace(t0, t1, integration_samples)

    if seed_control is None:
        seed_control = lambda t: jnp.zeros(system.input_dim)

    elif isinstance(seed_control, Array):
        seed_control = lambda t: seed_control  # constant control

    elif callable(seed_control):
        seed_control = seed_control

    elif isinstance(seed_control, str):
        sc = seed_control.lower()

        if sc == "feedback":

            def seed_control(t):
                dx = (target_state - init_state) / (t1 - t0)
                xt = init_state + t * dx
                return linear_solver(
                    system.B_fn(t, xt),
                    dx - system.N_fn(t, xt),
                    linear_solver_eps,
                )

        elif sc == "fl_min_energy":
            # Pendulum-only seed (2D, 1 input) using your closed-form minimum-energy reference
            seed_control = _make_fl_min_energy_seed_pendulum(
                system=system,
                init_state=init_state,
                target_state=target_state,
                t0=t0,
                t1=t1,
            )

        else:
            raise ValueError(
                f'Unknown seed_control="{seed_control}". Supported: None, callable, Array, '
                f'"feedback", "fl_min_energy".'
            )

    else:
        raise TypeError(f"seed_control {seed_control} is not a callable / supported type")

    if record_samples is None:
        record_ts = integration_ts
    else:
        record_ts = jnp.linspace(t0, t1, record_samples)

    if interpolation_samples is None:
        interpolation_ts = None
    else:
        interpolation_ts = jnp.linspace(t0, t1, interpolation_samples)

    if forward_mode:

        def _solve_JPhiB_tau_minus_t(t, xt):
            sol, decouple_fn = solve_nominal_JPhiB(
                system,
                x0=xt,
                t0=t,
                t1=t0,
                saveat=dfx.SaveAt(t1=True),
                ODE_solver_wrapper=ODE_solver_wrapper,
            )
            return decouple_fn(sol.ys[-1])[1]

        y = compute_forward_y(system, init_state, target_state, t0, t1, ODE_solver_wrapper)

    else:

        def _solve_JPhiB_tau_minus_t(t, xt):
            sol, decouple_fn = solve_nominal_JPhiB(
                system,
                xt,
                t0=t,
                t1=t1,
                saveat=dfx.SaveAt(t1=True),
                ODE_solver_wrapper=ODE_solver_wrapper,
            )
            return decouple_fn(sol.ys[-1])[1]

        y = compute_backward_y(system, init_state, target_state, t0, t1, ODE_solver_wrapper)

    #solve_JPhiB_tau_minus_t = eqx.filter_jit(_solve_JPhiB_tau_minus_t)
    solve_JPhiB_tau_minus_t = _solve_JPhiB_tau_minus_t

    return ControlObjective(
        system=system,
        x0=init_state,
        x1=target_state,
        t0=t0,
        t1=t1,
        ODE_solver_wrapper=ODE_solver_wrapper,
        forward=forward_mode,
        tol_x=tol_x,
        tol_u=tol_u,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        linear_solver=linear_solver,
        linear_solver_eps=linear_solver_eps,
        integrator=integrator,
        integration_ts=integration_ts,
        seed_control=seed_control,
        JPhiB_fn=solve_JPhiB_tau_minus_t,
        target_y=y,
        record_ts=record_ts,
        interpolation_ts=interpolation_ts,
    )


def make_solver_wrapper(
    solver: dfx.AbstractSolver,
    stepsize_controller: Optional[dfx.AbstractStepSizeController] = None,
    max_steps=5000,
    dt0=None,
) -> Callable:
    def ODE_solver_wrapper(
        terms: dfx.ODETerm,
        saveat: dfx.SaveAt,
        args: Tuple,
        y0: Array,
        t0: float,
        t1: float,
    ) -> dfx.Solution:
        local_dt0 = dt0
        if local_dt0 is not None:
            local_dt0 = jnp.where(t1 >= t0, jnp.abs(local_dt0), -jnp.abs(local_dt0))

        return diffeqsolve(
            terms=terms,
            saveat=saveat,
            args=args,
            y0=y0,
            t0=t0,
            t1=t1,
            solver=solver,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            dt0=local_dt0,
        )

    return ODE_solver_wrapper


def integrate_control_energy(u_func, ts, integrator):
    us = jax.vmap(u_func)(ts)
    integrand_u_l2 = jnp.sum(us**2, axis=1)
    u_l2 = integrator(integrand_u_l2, ts)
    u_l2_norm = jnp.sqrt(u_l2)
    return u_l2_norm


def integrate_gramian(ts, eval_xt, DPhiB_fn, integrator):
    DPhiBs = jax.vmap(lambda t: DPhiB_fn(t, eval_xt(t)))(ts)
    integrand = DPhiBs @ DPhiBs.mT
    M = integrator(integrand, ts)
    M = 0.5 * (M + M.T)
    return M


def compute_forward_y(system, x0, x1, t0, t1, ODE_solver_wrapper):
    sol = solve_nominal_ivp(system, x1, t1, t0, dfx.SaveAt(t1=True), ODE_solver_wrapper)
    y = sol.ys[-1] - x0
    return y


def compute_backward_y(system, x0, x1, t0, t1, ODE_solver_wrapper):
    sol = solve_nominal_ivp(
        system,
        x0,
        t0,
        t1,
        saveat=dfx.SaveAt(t1=True),
        ODE_solver_wrapper=ODE_solver_wrapper,
    )
    y = x1 - sol.ys[-1]
    return y


def solve_nominal_JPhiB(system, x0, t0, t1, saveat, ODE_solver_wrapper):
    state_dim = system.state_dim
    input_dim = system.input_dim

    def decouple_state_and_DPhiB(y):
        x = y[:state_dim]
        JPhiB = y[state_dim : state_dim * input_dim + state_dim].reshape(state_dim, input_dim)
        return x, JPhiB

    def rhs(t, y, args):
        x, JPhiB = decouple_state_and_DPhiB(y)
        dxdt = system.N_fn(t, x)
        dNdx = system.jacobian_N(t, x)
        dJPhiBdt = dNdx @ JPhiB
        return jnp.concatenate([dxdt.flatten(), dJPhiBdt.flatten()])

    y0 = jnp.concatenate([x0.flatten(), system.B_fn(t0, x0).flatten()])
    sol = ODE_solver_wrapper(
        terms=dfx.ODETerm(rhs),
        saveat=saveat,
        args=(),
        y0=y0,
        t0=t0,
        t1=t1,
    )
    return sol, decouple_state_and_DPhiB


def solve_controlled_ivp(system, x0, u_fn, t0, t1, saveat, ODE_solver_wrapper):
    def rhs(t, x, args):
        return system.F_fn(t, x, u_fn(t))

    y0 = x0
    sol = ODE_solver_wrapper(dfx.ODETerm(rhs), saveat, args=(), y0=y0, t0=t0, t1=t1)
    return sol


def solve_nominal_ivp(system, x0, t0, t1, saveat, ODE_solver_wrapper):
    def rhs(t, x, args):
        return system.N_fn(t, x)

    y0 = x0
    sol = ODE_solver_wrapper(dfx.ODETerm(rhs), saveat, args=(), y0=y0, t0=t0, t1=t1)
    return sol
