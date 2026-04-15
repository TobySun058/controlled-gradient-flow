import dataclasses
import math
from typing import Callable, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from control_synthesis.common import ControlObjective, integrate_control_energy, solve_controlled_ivp, solve_nominal_ivp
from control_synthesis.numeric import make_basis_function


@dataclasses.dataclass
class AlmostGramianIterationState:
    u: Callable[[Float], Array]
    almost_gramian: Optional[Array]
    classic_gramian: Optional[Array]
    target_error: Array
    energy: Array
    energy_certificate: Optional[Array]
    controlled_solution: dfx.Solution
    AK_fn: Callable[[float], Array]


def almost_gramian_initial_step(obj: ControlObjective):
    ts = obj.integration_ts

    u_cur_raw = obj.seed_control
    if obj.interpolation_ts is not None:
        u_cur_sampled = jax.vmap(u_cur_raw)(obj.interpolation_ts)
        u_cur_interpolation = dfx.LinearInterpolation(ts=obj.interpolation_ts, ys=u_cur_sampled)
        u_cur = lambda t: u_cur_interpolation.evaluate(t)
    else:
        u_cur = u_cur_raw

    u_l2_norm = integrate_control_energy(u_cur, obj.integration_ts, obj.integrator)
    sol = solve_controlled_ivp(obj.system, obj.x0, u_cur, obj.t0, obj.t1, saveat=dfx.SaveAt(dense=True),
                               ODE_solver_wrapper=obj.ODE_solver_wrapper)

    AL_fn = make_func_AL(obj, sol)
    AK_fn = make_func_AK(obj, sol, u_cur)

    #integrand_P = jax.vmap(lambda t: AL_fn(t) @ AK_fn(t).T)(ts)
    #P_next = obj.integrator(integrand_P, ts)
    AL_ts = jax.vmap(AL_fn)(ts)
    AK_ts = jax.vmap(AK_fn)(ts) 
    integrand_P = jax.vmap(lambda A, K: A @ K.T)(AL_ts, AK_ts)
    P_next = obj.integrator(integrand_P, ts)

    target_error = jnp.linalg.norm(sol.evaluate(obj.t1) - obj.x1)

    return AlmostGramianIterationState(
        u=u_cur,
        almost_gramian=None,
        classic_gramian=None,
        target_error=target_error,
        energy=u_l2_norm,
        energy_certificate=None,
        controlled_solution=sol,
        AK_fn=AK_fn
    ), P_next


def almost_gramian_iteration_step(obj: ControlObjective, AK_fn_pre, P):
    invP_y = obj.linear_solver(P, obj.target_y, obj.linear_solver_eps)

    _u_cur = lambda t: AK_fn_pre(t).T @ invP_y

    if obj.interpolation_ts is not None:
        u_cur_sampled = jax.vmap(_u_cur)(obj.interpolation_ts)
        u_cur_interpolation = dfx.LinearInterpolation(ts=obj.interpolation_ts, ys=u_cur_sampled)
        u_cur = lambda t: u_cur_interpolation.evaluate(t)
    else:
        u_cur = _u_cur

    sol = solve_controlled_ivp(obj.system, obj.x0, u_cur, obj.t0, obj.t1, saveat=dfx.SaveAt(dense=True),
                               ODE_solver_wrapper=obj.ODE_solver_wrapper)

    target_error = jnp.linalg.norm(sol.evaluate(obj.t1) - obj.x1)
    u_l2_norm = integrate_control_energy(u_cur, obj.integration_ts, obj.integrator)
    classic_M = compute_classic_control_gramian(obj, sol, u_cur)
    u_l2_norm_prime = jnp.sqrt(invP_y.T @ classic_M @ invP_y)

    ts = obj.integration_ts

    AL_fn = make_func_AL(obj, sol)
    AK_fn = make_func_AK(obj, sol, u_cur)

    #integrand_P = jax.vmap(lambda t: AL_fn(t) @ AK_fn(t).T)(ts)
    #P_next = obj.integrator(integrand_P, ts)
    AL_ts = jax.vmap(AL_fn)(ts)
    AK_ts = jax.vmap(AK_fn)(ts)
    integrand_P = jax.vmap(lambda A, K: A @ K.T)(AL_ts, AK_ts)
    P_next = obj.integrator(integrand_P, ts)

    return AlmostGramianIterationState(
        u=u_cur,
        almost_gramian=P,
        classic_gramian=classic_M,
        target_error=target_error,
        energy=u_l2_norm,
        energy_certificate=u_l2_norm_prime,
        controlled_solution=sol,
        AK_fn=AK_fn
    ), P_next


def solve_almost_gramian_synthesis(control_obj: ControlObjective):
    logs = {}

    sol = solve_nominal_ivp(control_obj.system, control_obj.x0, control_obj.t0, control_obj.t1,
                            saveat=dfx.SaveAt(dense=True),
                            ODE_solver_wrapper=control_obj.ODE_solver_wrapper)
    nominal_xT = sol.evaluate(control_obj.t1)
    target_error = jnp.linalg.norm(nominal_xT - control_obj.x1)
    logs["mode"] = "forward" if control_obj.forward else "backward"
    logs["nominal_trajectory"] = jax.vmap(sol.evaluate)(control_obj.record_ts)

    logs["nominal_error"] = target_error
    logs["y"] = control_obj.target_y
    jax.debug.print(
        "Nominal trajectory:\n "
        "||x_target-x_0(T)||2: {target_error:.5e}\n" +
        "---" * 30, target_error=target_error)
    picard_logs, history_states, P_next = almost_gramian_picard_iteration_loop(
        control_obj)

    logs.update(picard_logs)

    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, logs)

    return control_obj, logs, history_states, P_next


def almost_gramian_picard_iteration_loop(obj):
    record_ts = obj.record_ts
    logs = {"||x_target-x_{iter}(T)||2": [], "||u(t)||2": [], "||u(t)||'2": [], "gramians": [],
            "x_{iter}(T)": [],
            "u(t)": [], "recorded_ts": record_ts}
    history_states = []
    iter_state, P_next = almost_gramian_initial_step(obj)
    u_cur = iter_state.u
    sampled_u_prevs = jax.vmap(u_cur)(obj.integration_ts)
    logs["x_{iter}(T)"].append(jax.device_get(jax.vmap(iter_state.controlled_solution.evaluate)(record_ts)))
    logs["u(t)"].append(jax.device_get(jax.vmap(u_cur)(record_ts)))
    logs["||x_target-x_{iter}(T)||2"].append(jax.device_get(iter_state.target_error))
    logs["||u(t)||2"].append(float(jax.device_get(iter_state.energy)))
    history_states.append(jax.device_get(iter_state))
    jax.debug.print(
        "Seed control:\n"
        "||x_target-x_{iter}(T)||2 {target_error:.5e} | "
        "||u_0(t)||2: {u_l2:.5e} \n" +
        "---" * 50,
        iter=0,
        target_error=iter_state.target_error,
        u_l2=iter_state.energy)

    for iter in range(1, obj.max_iterations + 1):
        iter_state, P_next = almost_gramian_iteration_step(obj, iter_state.AK_fn, P_next)

        u_cur = iter_state.u
        sampled_u_curs = jax.vmap(u_cur)(obj.integration_ts)
        max_u_l2_norm_diff = jnp.max(jnp.abs(sampled_u_curs - sampled_u_prevs))

        sampled_u_prevs = sampled_u_curs

        P_eigs = jnp.linalg.eig(iter_state.almost_gramian)[0]

        logs["x_{iter}(T)"].append(jax.device_get(jax.vmap(iter_state.controlled_solution.evaluate)(record_ts)))
        logs["u(t)"].append(jax.device_get(jax.vmap(u_cur)(record_ts)))
        logs["||x_target-x_{iter}(T)||2"].append(jax.device_get(iter_state.target_error))
        logs["||u(t)||2"].append(float(jax.device_get(iter_state.energy)))
        logs["||u(t)||'2"].append(float(jax.device_get(iter_state.energy_certificate)))
        logs["gramians"].append(jax.device_get(iter_state.almost_gramian))
        history_states.append(jax.device_get(iter_state))

        jax.debug.print(
            "Iteration: {iter_cur}:\n"
            "||x_target-x_{iter_cur}(T)||2: {target_error:.5e} | "
            "||u_{iter_cur}(t)||2: {u_l2:.5e} | "
            "||u_{iter_cur}(t)||'2: {u_l2p:.5e} | "
            "min(λ(P_{iter_cur})): {min_P_eig:.5f} | "
            "max(λ(P_{iter_cur})): {max_P_eig:.5f} | "
            "||u_{iter_cur}(t)-u_{iter_prev}(t)||_inf): {diff_u:.5e}\n" +
            "---" * 50,
            iter_cur=iter,
            iter_prev=iter - 1,
            target_error=iter_state.target_error,
            u_l2=iter_state.energy,
            u_l2p=iter_state.energy_certificate,
            min_P_eig=jnp.min(P_eigs),
            max_P_eig=jnp.max(P_eigs),
            diff_u=max_u_l2_norm_diff)
        if iter == obj.max_iterations:
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
    return logs, history_states, P_next


def make_func_AK(obj, sol, u_func):
    model = obj.system
    n = model.state_dim
    m = model.input_dim

    # Build an x(t) interpolant from *sampled values* so we never close over sol.evaluate in a jitted context
    x_ts = obj.integration_ts
    x_sampled = jax.vmap(sol.evaluate)(x_ts)
    x_interp = dfx.LinearInterpolation(ts=x_ts, ys=x_sampled)

    # Build a u(t) interpolant from sampled values too (even if u_func is already an interpolation)
    u_sampled = jax.vmap(u_func)(x_ts)
    u_interp = dfx.LinearInterpolation(ts=x_ts, ys=u_sampled)

    def dR_dt(t, y, args):
        x_interp_, u_interp_ = args
        R = y.reshape((n, n))
        x = x_interp_.evaluate(t)
        u = u_interp_.evaluate(t)
        J = model.jacobian_F(t, x, u)
        dR = -R @ J
        return dR.reshape(-1)

    solR = obj.ODE_solver_wrapper(
        terms=dfx.ODETerm(dR_dt),
        saveat=dfx.SaveAt(dense=True),  # OK: this is only n×n
        args=(x_interp, u_interp),
        y0=jnp.eye(n).reshape(-1),
        t0=obj.t1,
        t1=obj.t0,
    )

    def AK_fn(t):
        Rt = solR.evaluate(t).reshape((n, n))
        xt = x_interp.evaluate(t)
        Bt = model.B_fn(t, xt).reshape((n, m))
        return Rt @ Bt

    return AK_fn



#def make_func_AL(obj, sol):
#    func_AL = lambda t: obj.JPhiB_fn(t, sol.evaluate(t))
#    return func_AL
def make_func_AL(obj, sol):
    x_ts = obj.integration_ts
    x_sampled = jax.vmap(sol.evaluate)(x_ts)
    x_interp = dfx.LinearInterpolation(ts=x_ts, ys=x_sampled)

    def func_AL(t):
        return obj.JPhiB_fn(t, x_interp.evaluate(t))

    return func_AL


def compute_classic_control_gramian(obj, sol, u_func):
    model = obj.system

    def rhs(t, y, args):
        Q = y.reshape((model.state_dim, model.state_dim))
        xt = sol.evaluate(t)
        Bt = model.B_fn(t, xt)
        dFdt = model.jacobian_F(t, xt, u_func(t))
        dFdt_Q = dFdt @ Q
        Qt = Bt @ Bt.T + dFdt_Q + dFdt_Q.T
        return Qt.flatten()

    return obj.ODE_solver_wrapper(
        terms=dfx.ODETerm(rhs),
        saveat=dfx.SaveAt(t1=True),
        args=(),
        y0=jnp.zeros((obj.system.state_dim ** 2,)),
        t0=obj.t0,
        t1=obj.t1).ys[-1].reshape((obj.system.state_dim, obj.system.state_dim))


def check_minimum_energy_condition(obj, u_func, m, basis_type, normalized=False):
    model_params = obj.system
    ts = obj.integration_ts

    def rhs(t, y, args):
        x = y
        u = u_func(t)
        return model_params.F_fn(t, x, u)

    sol = obj.ODE_solver_wrapper(
        terms=dfx.ODETerm(rhs),
        saveat=dfx.SaveAt(dense=True),
        args=(u_func,),
        y0=obj.x0,
        t0=obj.t0,
        t1=obj.t1)

    AL_fn = make_func_AL(obj, sol)
    AK_fn = make_func_AK(obj, sol, u_func)
    integrand_P = jax.vmap(lambda t: AL_fn(t) @ AK_fn(t).T)(ts)
    P = obj.integrator(integrand_P, ts)
    min_m = math.ceil((model_params.state_dim / model_params.input_dim - 1) / 2)
    assert m >= min_m, f"m must be at least {min_m} for this system."
    if m < min_m:
        raise ValueError(f"m ({m}) must be at least {min_m} for this system.")

    basis_func = make_basis_function(m, obj.t1 - obj.t0, obj.t0, basis_type)

    def func(func, phi):
        integrand = jax.vmap(
            lambda t: func(t)[None, :, :] * phi(t)
            .squeeze()
            [:, None,
                None])(ts)
        return jnp.hstack(obj.integrator(integrand, ts))

    L = func(AL_fn, basis_func)
    K = func(AK_fn, basis_func)
    rk = jnp.linalg.matrix_rank(L)
    Q, R = jnp.linalg.qr(L.T, mode="complete")
    invP_y = jnp.linalg.solve(P, obj.target_y)
    v = invP_y

    Vs = K @ Q[:, rk:]
    if normalized:
        result = jnp.dot(v, Vs) / (jnp.linalg.norm(v) * jnp.linalg.norm(Vs, axis=0))
    else:
        result = jnp.dot(v, Vs)
    return result
