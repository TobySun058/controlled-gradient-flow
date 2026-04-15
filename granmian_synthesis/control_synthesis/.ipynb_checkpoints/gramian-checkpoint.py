import dataclasses
from typing import Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from control_synthesis.common import ControlFn, ControlObjective, integrate_control_energy, integrate_gramian, \
    solve_controlled_ivp, solve_nominal_ivp


@dataclasses.dataclass
class GramianIterationState:
    u: ControlFn
    gramian: Optional[Array]
    target_error: Array
    energy: Array
    energy_certificate: Optional[Array]
    controlled_solution: dfx.Solution


def solve_gramian_synthesis(control_obj: ControlObjective):
    logs = {}
    M = compute_nominal_gramian(
        control_obj)
    M_eigs = jnp.linalg.eigh(M)[0]
    if jnp.min(M_eigs) < 1e-12:
        jax.debug.print("Nominal gramian is ill-conditioned. Minimum eigenvalue: {}".format(jnp.min(M_eigs)))
    sol = solve_nominal_ivp(control_obj.system, control_obj.x0, control_obj.t0, control_obj.t1,
                            saveat=dfx.SaveAt(dense=True),
                            ODE_solver_wrapper=control_obj.ODE_solver_wrapper)
    nominal_xT = sol.evaluate(control_obj.t1)
    target_error = jnp.linalg.norm(nominal_xT - control_obj.x1)
    logs["mode"] = "forward" if control_obj.forward else "backward"
    logs["nominal_M"] = M
    logs["nominal_trajectory"] = jax.vmap(sol.evaluate)(control_obj.record_ts)

    logs["nominal_error"] = target_error
    logs["y"] = control_obj.target_y
    jax.debug.print(
        "Nominal trajectory:\n"
        "||x_target-x_0(T)||2: {target_error:.5e} |"
        "min(λ(M(T))) = {min_eig_val:.5f} | "
        "max(λ(M(T)))) = {max_eig_val:.5f}\n" +
        "---" * 30, target_error=target_error,
        max_eig_val=jnp.max(M_eigs), min_eig_val=jnp.min(M_eigs))
    picard_logs, history_states, M_next = gramian_picard_iteration_loop(
        control_obj)

    logs.update(picard_logs)

    # ensure all nominal_logs are computed before returning
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, logs)

    return control_obj, logs, history_states, M_next


def gramian_picard_iteration_loop(obj):
    record_ts = obj.record_ts
    logs = {"||x_target-x_{iter}(T)||2": [], "||u(t)||2": [], "||u(t)||'2": [], "gramians": [],
            "x_{iter}(T)": [],
            "u(t)": [], "recorded_ts": record_ts}
    history_states = []
    iter_state, M_next = gramian_initial_step(obj)
    u_cur = iter_state.u
    sampled_u_prevs = jax.vmap(u_cur)(obj.integration_ts)  # used to compute ||u_cur-u_prev||_inf
    logs["x_{iter}(T)"].append(jax.device_get(jax.vmap(iter_state.controlled_solution.evaluate)(record_ts)))
    logs["u(t)"].append(jax.device_get(jax.vmap(u_cur)(record_ts)))
    logs["||x_target-x_{iter}(T)||2"].append(jax.device_get(iter_state.target_error))
    logs["||u(t)||2"].append(float(jax.device_get(iter_state.energy)))

    history_states.append(jax.device_get(iter_state))
    jax.debug.print(
        "Seed control:\n"
        "||x_target-x_{iter}(T)||2 {target_error:.5e} | "
        "||u_0(t)||2: {u_l2:.5e}\n" +
        "---" * 50,
        iter=0,
        target_error=iter_state.target_error,
        u_l2=iter_state.energy)
    for iter in range(1, obj.max_iterations + 1):
        iter_state, M_next = gramian_iteration_step(obj, iter_state.controlled_solution, M_next)
        u_cur = iter_state.u
        sampled_u_curs = jax.vmap(u_cur)(obj.integration_ts)
        max_u_l2_norm_diff = jnp.max(jnp.abs(sampled_u_curs - sampled_u_prevs))
        sampled_u_prevs = sampled_u_curs

        M_eigs = jnp.linalg.eigh(iter_state.gramian)[0]

        logs["x_{iter}(T)"].append(jax.device_get(jax.vmap(iter_state.controlled_solution.evaluate)(record_ts)))
        logs["u(t)"].append(jax.device_get(jax.vmap(u_cur)(record_ts)))
        logs["||x_target-x_{iter}(T)||2"].append(jax.device_get(iter_state.target_error))
        logs["||u(t)||2"].append(float(jax.device_get(iter_state.energy)))
        logs["||u(t)||'2"].append(float(jax.device_get(iter_state.energy_certificate)))
        logs["gramians"].append(jax.device_get(iter_state.gramian))
        history_states.append(jax.device_get(iter_state))


        jax.debug.print(
            "Iteration: {iter_cur}:\n"
            "||x_target-x_{iter_cur}(T)||2: {target_error:.5e} | "
            "||u_{iter_cur}(t)||2: {u_l2:.5e} | "
            "||u_{iter_cur}(t)||'2: {u_l2p:.5e} | "
            "min(λ(M_{iter_cur})): {mmin:.5f} | "
            "max(λ(M_{iter_cur})): {mmax:.5f} | "
            "||u_{iter_cur}(t)-u_{iter_prev}(t)||_inf): {diff_u:.5e}\n" +
            "---" * 50,
            iter_cur=iter,
            iter_prev=iter - 1,
            target_error=iter_state.target_error,
            u_l2=iter_state.energy,
            u_l2p=iter_state.energy_certificate,
            mmin=jnp.min(M_eigs),
            mmax=jnp.max(M_eigs),
            diff_u=max_u_l2_norm_diff)
        if iter == obj.max_iterations:  # maximum iteration reached, just evaluate the last control
            jax.debug.print(
                "Maximum iteration {max_it} reached.",
                max_it=obj.max_iterations,
            )
            break
        elif obj.tol_x is not None and logs["||x_target-x_{iter}(T)||2"][-1] < obj.tol_x and iter >= obj.min_iterations:
            jax.debug.print(
                "||x_target-x_{iter_cur}(T)||2 {err:.5e} < {tol:.5e} at iteration {iter_cur}",
                iter_cur=iter,
                err=logs["||x_target-x_{iter}(T)||2"][-1],
                tol=obj.tol_x
            )
            jax.debug.print(
                "Minimum iteration {min_it} reached.",
                min_it=obj.min_iterations,
            )
            break

        elif obj.tol_u is not None and max_u_l2_norm_diff < obj.tol_u and iter >= obj.min_iterations:
            jax.debug.print(
                "||u_{iter_cur}(t)-u_{iter_prev}(t)||_inf): {diff:.5e} < {tol:.5e} at iteration {iter_cur}",
                diff=max_u_l2_norm_diff,
                tol=obj.tol_u,
                iter_cur=iter,
                iter_prev=iter - 1,
            )
            jax.debug.print(
                "Minimum iteration {min_it} reached.",
                min_it=obj.min_iterations)
            break
        else:
            pass
    return logs, history_states, M_next


def gramian_initial_step(obj):
    u_cur = obj.seed_control
    sol = solve_controlled_ivp(obj.system, obj.x0, u_cur, obj.t0, obj.t1, saveat=dfx.SaveAt(dense=True),
                               ODE_solver_wrapper=obj.ODE_solver_wrapper)
    M_next = integrate_gramian(obj.integration_ts, lambda t: sol.evaluate(t), obj.JPhiB_fn, obj.integrator)
    u_l2_norm = integrate_control_energy(u_cur, obj.integration_ts, obj.integrator)
    target_error = jnp.linalg.norm(sol.evaluate(obj.t1) - obj.x1)
    return GramianIterationState(
        u=u_cur,
        gramian=None,
        target_error=target_error,
        energy=u_l2_norm,
        energy_certificate=None,
        controlled_solution=sol), M_next


def gramian_iteration_step(obj, sol, M):
    invM_y = obj.linear_solver(M, obj.target_y, obj.linear_solver_eps)
    _u_cur = lambda t: obj.JPhiB_fn(
        t,
        sol.evaluate(t)).T @ invM_y
    if obj.interpolation_ts is not None:
        u_cur_sampled = jax.vmap(_u_cur)(obj.interpolation_ts)
        u_cur_interpolation = dfx.LinearInterpolation(ts=obj.interpolation_ts, ys=u_cur_sampled)
        u_cur: ControlFn = lambda t: u_cur_interpolation.evaluate(t)
    else:
        u_cur: ControlFn = _u_cur
    sol = solve_controlled_ivp(obj.system, obj.x0, u_cur, obj.t0, obj.t1, saveat=dfx.SaveAt(dense=True),
                               ODE_solver_wrapper=obj.ODE_solver_wrapper)
    target_error = jnp.linalg.norm(sol.evaluate(obj.t1) - obj.x1)
    u_l2_norm = integrate_control_energy(u_cur, obj.integration_ts, obj.integrator)
    u_l2_norm_prime = jnp.sqrt(obj.target_y.T @ invM_y)

    M_next = integrate_gramian(obj.integration_ts, lambda t: sol.evaluate(t), obj.JPhiB_fn, obj.integrator)

    return GramianIterationState(
        u=u_cur,
        gramian=M,
        target_error=target_error,
        energy=u_l2_norm,
        energy_certificate=u_l2_norm_prime,
        controlled_solution=sol), M_next


def compute_nominal_gramian(obj: ControlObjective) -> Array:
    if obj.forward:
        return compute_forward_nominal_gramian(
            obj.system,
            obj.x0,
            obj.t0,
            obj.t1,
            obj.JPhiB_fn,
            obj.ODE_solver_wrapper)[2]
    else:
        return compute_backward_nominal_gramian(
            obj.system,
            obj.x0,
            obj.t0,
            obj.t1,
            obj.ODE_solver_wrapper)[2]


def compute_forward_nominal_gramian(system, x0, t0, t1, solve_JPhiB_tau_minus_t, ODE_solver_wrapper):
    state_dim = system.state_dim
    sol = solve_nominal_ivp(system, x0, t0, t1, saveat=dfx.SaveAt(dense=True), ODE_solver_wrapper=ODE_solver_wrapper)

    def rhs(t, y, args):
        J = solve_JPhiB_tau_minus_t(t, sol.evaluate(t))

        dMdt = J @ J.T
        return dMdt.flatten()

    y0 = jnp.zeros((state_dim * state_dim,))
    sol_ = ODE_solver_wrapper(
        y0=y0,
        t0=t0,
        t1=t1,
        terms=dfx.ODETerm(rhs),
        saveat=dfx.SaveAt(t1=True),
        args=())
    M = sol_.ys[-1].reshape(state_dim, state_dim)
    M = 0.5 * (M + M.T)
    return sol.evaluate(t1), solve_JPhiB_tau_minus_t(t1, sol.evaluate(t1)), M


def compute_backward_nominal_gramian(system, x0, t0, t1, ODE_solver_wrapper):
    state_dim = system.state_dim

    def decouple_state_JPhi_and_granmian(y):
        x = y[:state_dim]
        JPhi = y[state_dim:state_dim * state_dim + state_dim].reshape(state_dim, state_dim)
        M = y[state_dim * state_dim + state_dim:].reshape(state_dim, state_dim)
        M = 0.5 * (M + M.T)
        return x, JPhi, M

    def rhs(t, y, args):
        x, JPhi, M = decouple_state_JPhi_and_granmian(y)
        dxdt = system.N_fn(t, x)
        dNdx = system.jacobian_N(t, x)
        dJPhidt = dNdx @ JPhi
        Bt = system.B_fn(t, x)
        dJPhidtM = dJPhidt @ M
        dMdt = Bt @ Bt.T + dJPhidtM @ dJPhidtM.T
        return jnp.concatenate([dxdt.flatten(), dJPhidt.flatten(), dMdt.flatten()])

    y0 = jnp.concatenate(
        [x0.flatten(), jnp.eye(state_dim).flatten(),
         jnp.zeros((state_dim * state_dim,))])
    sol = ODE_solver_wrapper(
        y0=y0,
        t0=t0,
        t1=t1,
        terms=dfx.ODETerm(rhs),
        saveat=dfx.SaveAt(t1=True),
        args=())
    y1 = sol.ys[-1]
    return decouple_state_JPhi_and_granmian(y1)
