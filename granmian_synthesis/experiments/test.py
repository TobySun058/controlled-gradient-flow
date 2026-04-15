from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Any, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax as dfx

from control_synthesis import (
    setup_control_synthesis,
    solve_almost_gramian_synthesis,
    solve_gramian_synthesis,
    make_solver_wrapper,
)

from control_synthesis import *
from jax import config

config.update("jax_enable_x64", True)

Array = jnp.ndarray
VectorField = Callable[[Array], Array]
Control_t = Callable[[float], Array]

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_fig(fig, filename: str, dpi: int = 300):
    fig.savefig(PLOT_DIR / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Create Function and Gradient
@dataclass(frozen=True)
class SoftMin3Params:
    centers: Array
    As: Array
    offsets: Array
    k_softmin: float


# Define quadratic function
def q_i(x: Array, i: int, p: SoftMin3Params) -> Array:
    d = x - p.centers[i]
    return 0.5 * (d @ p.As[i] @ d) + p.offsets[i]


# Define gradient of single quadratic function
def grad_q_i(x: Array, i: int, p: SoftMin3Params) -> Array:
    return p.As[i] @ (x - p.centers[i])


# Compute smooth approximation
def f_softmin(x: Array, p: SoftMin3Params) -> Array:
    qs = jnp.stack([q_i(x, i, p) for i in range(3)], axis=0)
    m = jnp.min(qs)
    s = jnp.sum(jnp.exp(-p.k_softmin * (qs - m)))
    return m - (1.0 / p.k_softmin) * jnp.log(s)


# Compute gradient of softmin
def grad_f_softmin(x: Array, p: SoftMin3Params) -> Array:
    qs = jnp.stack([q_i(x, i, p) for i in range(3)], axis=0)
    m = jnp.min(qs)
    w_unnorm = jnp.exp(-p.k_softmin * (qs - m))
    w = w_unnorm / jnp.sum(w_unnorm)
    grads = jnp.stack([grad_q_i(x, i, p) for i in range(3)], axis=0)
    return jnp.sum(w[:, None] * grads, axis=0)


# Define N
def make_N(p: SoftMin3Params) -> VectorField:
    """Eq. (6) uses N(x) = -∇f(x)."""
    return lambda x: -grad_f_softmin(x, p)


# Change quadratic function to a system
def make_gradient_flow_system(N_of_x: VectorField, state_dim: int = 2):
    input_dim = state_dim

    def B_fn(t, x):
        return jnp.eye(state_dim, input_dim)

    def F_fn(t, x, u):
        return N_fn(t, x) + B_fn(t, x) @ u

    jac_N = jax.jit(jax.jacfwd(lambda x: N_of_x(x)))
    N_of_x_jit = jax.jit(N_of_x)

    def N_fn(t, x):
        return N_of_x_jit(x)

    def jacobian_N(t, x):
        return jac_N(x)

    def jacobian_F(t, x, u):
        return jac_N(x)

    return SimpleNamespace(
        state_dim=state_dim,
        input_dim=input_dim,
        N_fn=N_fn,
        B_fn=B_fn,
        F_fn=F_fn,
        jacobian_N=jacobian_N,
        jacobian_F=jacobian_F,
        dynamics=F_fn,
        N=N_of_x,
    )


# Build ODE solver wrapper
def build_ode_solver_wrapper(rtol=1e-12, atol=1e-12, max_steps=400000, dt0=None):
    return make_solver_wrapper(
        solver=dfx.Dopri8(),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        dt0=dt0,
    )


# Build control synthesis object
def build_control_obj(
    system: Any,
    x_init: Array,
    x_target: Array,
    t_start: float,
    t_target: float,
    ode_solver_wrapper: Any,
    seed_control: Optional[str] = None,
    tol_x: float = 1e-12,
    tol_u: float = 1e-12,
    forward_mode: bool = False,
    integration_method: str = "simpsons",
    integration_samples: int = 50000,
    max_iterations: int = 25,
    min_iterations: int = 5,
    linear_solver_method: str = "ls",
    linear_solver_eps: float = 0.0,
    record_samples: int = 5000,
    interpolation_samples: Optional[int] = 50000,
):
    return setup_control_synthesis(
        system=system,
        init_state=x_init,
        target_state=x_target,
        t0=t_start,
        t1=t_target,
        ODE_solver_wrapper=ode_solver_wrapper,
        seed_control=seed_control,
        tol_x=tol_x,
        tol_u=tol_u,
        forward_mode=forward_mode,
        integration_method=integration_method,
        integration_samples=integration_samples,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        linear_solver_method=linear_solver_method,
        linear_solver_eps=linear_solver_eps,
        record_samples=record_samples,
        interpolation_samples=interpolation_samples,
    )


# Function for almost minimum energy control
def run_u2_almost_gramian(obj: Any):
    return solve_almost_gramian_synthesis(control_obj=obj)


# Function for minimum energy control
def run_u2_tilde_gramian(obj: Any):
    return solve_gramian_synthesis(control_obj=obj)


# Visualization of function
def plot_f_softmin_2d(
    p: SoftMin3Params,
    x_init: Array,
    x_target: Array,
    xlim=(-2.0, 10.0),
    ylim=(-2.0, 10.0),
    n_grid: int = 300,
    levels: int = 40,
    filename: str = "f_softmin_2d.png",
):
    xs = jnp.linspace(xlim[0], xlim[1], n_grid)
    ys = jnp.linspace(ylim[0], ylim[1], n_grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    f_vals = jax.vmap(lambda z: f_softmin(z, p))(pts)
    Z = f_vals.reshape((n_grid, n_grid))

    plt.figure()
    plt.contourf(np.array(X), np.array(Y), np.array(Z), levels=levels)
    plt.contour(np.array(X), np.array(Y), np.array(Z), levels=levels)

    plt.scatter([float(x_init[0])], [float(x_init[1])], marker="o", s=80, label="x_init")
    plt.scatter([float(x_target[0])], [float(x_target[1])], marker="*", s=160, label="x_target")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original objective: f_softmin(x)")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(xlim)
    plt.ylim(ylim)

    save_current_figure(filename)


# Feedback linearization control
def make_u_fl(system: Any, x_init: Array, x_target: Array, t_start: float, t_target: float):
    T = t_target - t_start

    def u_fl(t: float) -> Array:
        dx = (x_target - x_init) / T
        x_ref = x_init + ((t - t_start) / T) * (x_target - x_init)
        return dx - system.N_fn(t, x_ref)

    return u_fl


# Plot linearized control trajectory on top of softmin contour
def plot_u_fl_trajectory(
    p: SoftMin3Params,
    x_init: Array,
    x_target: Array,
    traj: Array,
    xlim=(-2, 8),
    ylim=(-2, 10),
    n_grid: int = 300,
    levels: int = 40,
    filename: str = "u_fl_trajectory.png",
):
    xs = jnp.linspace(xlim[0], xlim[1], n_grid)
    ys = jnp.linspace(ylim[0], ylim[1], n_grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    f_vals = jax.vmap(lambda z: f_softmin(z, p))(pts)
    Z = f_vals.reshape((n_grid, n_grid))

    plt.figure(figsize=(8, 6))
    plt.contourf(np.array(X), np.array(Y), np.array(Z), levels=levels)
    plt.contour(np.array(X), np.array(Y), np.array(Z), levels=levels)

    plt.plot(np.array(traj[:, 0]), np.array(traj[:, 1]), linewidth=2, label="u_fl trajectory")
    plt.scatter([float(x_init[0])], [float(x_init[1])], marker="o", s=80, label="x_init")
    plt.scatter([float(x_target[0])], [float(x_target[1])], marker="*", s=160, label="x_target")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Feedback linearization control trajectory")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()

    save_current_figure(filename)


# Simulate uncontrolled system
def simulate_uncontrolled(system: Any, x_init: Array, t_start: float, t_target: float, saveat, ode_solver_wrapper):
    return solve_nominal_ivp(
        system=system,
        x0=x_init,
        t0=t_start,
        t1=t_target,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_wrapper,
    )


# Plain GD
def run_gd_np(X0, p, gamma=0.10, N=140):
    X = np.zeros((N + 1, 2))
    X[0] = np.array(X0, dtype=float)
    for n in range(N):
        g = np.array(grad_f_softmin(jnp.array(X[n]), p))
        X[n + 1] = X[n] - gamma * g
    return X


# Momentum GD
def run_momentum_np(X0, p, gamma=0.07, beta=0.90, N=140):
    X = np.zeros((N + 1, 2))
    X[0] = np.array(X0, dtype=float)

    g0 = np.array(grad_f_softmin(jnp.array(X[0]), p))
    X[1] = X[0] - gamma * g0

    for n in range(1, N):
        g = np.array(grad_f_softmin(jnp.array(X[n]), p))
        X[n + 1] = X[n] - gamma * g + beta * (X[n] - X[n - 1])
    return X


# Stochastic GD
def run_sgd_np(X0, p, gamma=0.10, sigma=0.15, N=140, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((N + 1, 2))
    X[0] = np.array(X0, dtype=float)

    for n in range(N):
        g = np.array(grad_f_softmin(jnp.array(X[n]), p))
        noise = rng.normal(0.0, sigma, size=2)
        X[n + 1] = X[n] - gamma * (g + noise)
    return X


# Print endpoint error and energy summary
def print_control_summary(name: str, traj: Array, x_target: Array, energy: float):
    final_state = traj[-1]
    endpoint_error = jnp.linalg.norm(final_state - x_target)

    print(f"{name}:")
    print(f"  Final state    = {np.array(final_state)}")
    print(f"  Endpoint error = {float(endpoint_error):.12e}")
    print(f"  Energy         = {float(energy):.12e}")
    print()


def plot_all_trajectories(
    p: SoftMin3Params,
    x_init: Array,
    x_target: Array,
    traj_zero: Array,
    traj_min: Array,
    traj_almost_min: Array,
    traj_fl: Array,
    traj_gd: Array,
    traj_momentum: Array,
    traj_sgd: Array,
    xlim=(-2, 8),
    ylim=(-2, 10),
    n_grid: int = 300,
    levels: int = 40,
    filename: str = "all_trajectories.png",
):
    xs = jnp.linspace(xlim[0], xlim[1], n_grid)
    ys = jnp.linspace(ylim[0], ylim[1], n_grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    f_vals = jax.vmap(lambda z: f_softmin(z, p))(pts)
    Z = f_vals.reshape((n_grid, n_grid))

    plt.figure(figsize=(9, 7))
    plt.contourf(np.array(X), np.array(Y), np.array(Z), levels=levels)
    plt.contour(np.array(X), np.array(Y), np.array(Z), levels=levels, linewidths=0.6)

    plt.plot(np.array(traj_zero[:, 0]), np.array(traj_zero[:, 1]), linewidth=2, label="u = 0")
    plt.plot(np.array(traj_min[:, 0]), np.array(traj_min[:, 1]), linewidth=2, label="u2_min")
    plt.plot(np.array(traj_almost_min[:, 0]), np.array(traj_almost_min[:, 1]), linewidth=2, label="u2_almost_min")
    plt.plot(np.array(traj_fl[:, 0]), np.array(traj_fl[:, 1]), linewidth=2, label="u_fl")
    plt.plot(np.array(traj_gd[:, 0]), np.array(traj_gd[:, 1]), linewidth=2, label="GD")
    plt.plot(np.array(traj_momentum[:, 0]), np.array(traj_momentum[:, 1]), linewidth=2, label="Momentum GD")
    plt.plot(np.array(traj_sgd[:, 0]), np.array(traj_sgd[:, 1]), linewidth=2, label="Stochastic GD")

    plt.scatter([float(x_init[0])], [float(x_init[1])], marker="o", s=80, label="x_init")
    plt.scatter([float(x_target[0])], [float(x_target[1])], marker="*", s=180, label="x_target")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Trajectory comparison")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()

    save_current_figure(filename)

# Main function
p_softmin = SoftMin3Params(
    centers=jnp.array([[0.0, 0.0], [4.0, 2.0], [1.0, 6.0]]),
    As=jnp.array([jnp.eye(2), jnp.eye(2), jnp.eye(2)]),
    offsets=jnp.array([2.0, 6.0, 8.0]),
    k_softmin=0.5,
)

N_fn = make_N(p_softmin)
system = make_gradient_flow_system(N_fn, state_dim=2)

x_init = jnp.array([2.0, 6.0])
x_target = jnp.array([4.5, 1.5])
t_start = 0.0
t_target = 0.75

ode_solver_wrapper = build_ode_solver_wrapper(
    rtol=1e-7,
    atol=1e-7,
    max_steps=100000,
    dt0=None,
)

obj = build_control_obj(
    system=system,
    x_init=x_init,
    x_target=x_target,
    t_start=t_start,
    t_target=t_target,
    ode_solver_wrapper=ode_solver_wrapper,
    integration_method="simpsons",
    integration_samples=50000,
)

plot_f_softmin_2d(
    p_softmin,
    x_init=x_init,
    x_target=x_target,
    xlim=(-2, 8),
    ylim=(-2, 10),
    n_grid=300,
    levels=40,
    filename="01_f_softmin_contour.png",
)

# Minimum control
out_min = run_u2_almost_gramian(obj)
obj_min, min_logs, min_hist, _P_min = out_min

plot_metrics(min_logs, log_scale=False)
save_current_figure("02_min_metrics.png")

fig = plot_alignment_metrics(obj, min_logs, plot_iters=[0, 1, 2, 15, 25])
save_fig(fig, "03_min_alignment_metrics.png")

fig = plot_2d_vector_field(
    obj,
    min_logs,
    500,
    xlim=(-5, 10),
    ylim=(-5, 5),
    plot_iters=[0, 3, 6, 9],
    alpha=0.8,
    cmap="turbo",
)
save_fig(fig, "04_min_vector_field.png")


# Almost Minimum control
out_almost_min = run_u2_tilde_gramian(obj)
obj_almost_min, almost_min_logs, almost_min_hist, _P_almost_min = out_almost_min

plot_metrics(almost_min_logs, log_scale=False)
save_current_figure("05_almost_min_metrics.png")

fig = plot_alignment_metrics(obj, almost_min_logs, plot_iters=[0, 1, 2, 15, 25])
save_fig(fig, "06_almost_min_alignment_metrics.png")

fig = plot_2d_vector_field(
    obj,
    almost_min_logs,
    500,
    xlim=(-5, 10),
    ylim=(-5, 5),
    plot_iters=[0, 3, 6, 9],
    alpha=0.8,
    cmap="turbo",
)
save_fig(fig, "07_almost_min_vector_field.png")

# Comparing the gramian and the almost gramian trajectories
fig = compare_2d_vector_field(
    obj,
    almost_min_logs,
    min_logs,
    500,
    xlim=(-5, 10),
    ylim=(-5, 5),
    alpha=0.8,
    cmap="turbo",
)
save_fig(fig, "08_compare_vector_fields.png")


# Feedback linearization control
u_fl = make_u_fl(
    system=system,
    x_init=x_init,
    x_target=x_target,
    t_start=t_start,
    t_target=t_target,
)

# Simulate the controlled nonlinear system
sol_fl = solve_controlled_ivp(
    system=system,
    x0=x_init,
    u_fn=u_fl,
    t0=t_start,
    t1=t_target,
    saveat=dfx.SaveAt(ts=obj.record_ts),
    ODE_solver_wrapper=ode_solver_wrapper,
)

traj_fl = sol_fl.ys

# Compute control energy for feedback linearization control
energy_fl = integrate_control_energy(u_fl, obj.integration_ts, obj.integrator)

print("Feedback linearization control:")
print("  Final state    =", np.array(traj_fl[-1]))
print("  Target state   =", np.array(x_target))
print("  Endpoint error =", float(jnp.linalg.norm(traj_fl[-1] - x_target)))
print("  Energy         =", float(energy_fl))

# Plot trajector for feedback linearization control
plot_u_fl_trajectory(
    p_softmin,
    x_init=x_init,
    x_target=x_target,
    traj=traj_fl,
    xlim=(-2, 8),
    ylim=(-2, 10),
    n_grid=300,
    levels=40,
    filename="09_u_fl_trajectory.png",
)

# Uncontrolled trajectory: u = 0
sol_zero = simulate_uncontrolled(
    system=system,
    x_init=x_init,
    t_start=t_start,
    t_target=t_target,
    saveat=dfx.SaveAt(ts=obj.record_ts),
    ode_solver_wrapper=ode_solver_wrapper,
)
traj_zero = sol_zero.ys

# Extra GD methods
traj_gd = run_gd_np(x_init, p_softmin, gamma=0.10, N=140)
traj_momentum = run_momentum_np(x_init, p_softmin, gamma=0.07, beta=0.90, N=140)
traj_sgd = run_sgd_np(x_init, p_softmin, gamma=0.10, sigma=0.15, N=140, seed=0)

# Final control functions from the two methods
u_min = min_hist[-1].u
u_almost_min = almost_min_hist[-1].u

# Simulate full trajectories using the final controls
sol_min = solve_controlled_ivp(
    system=system,
    x0=x_init,
    u_fn=u_min,
    t0=t_start,
    t1=t_target,
    saveat=dfx.SaveAt(ts=obj.record_ts),
    ODE_solver_wrapper=ode_solver_wrapper,
)

sol_almost_min = solve_controlled_ivp(
    system=system,
    x0=x_init,
    u_fn=u_almost_min,
    t0=t_start,
    t1=t_target,
    saveat=dfx.SaveAt(ts=obj.record_ts),
    ODE_solver_wrapper=ode_solver_wrapper,
)

traj_min = sol_min.ys
traj_almost_min = sol_almost_min.ys

# Energies
energy_zero = 0.0
energy_min = float(min_hist[-1].energy)
energy_almost_min = float(almost_min_hist[-1].energy)
energy_fl = float(energy_fl)

# Print summary
print("\n==================== SUMMARY ====================\n")

print_control_summary("u = 0", traj_zero, x_target, energy_zero)
print_control_summary("u2_min", traj_min, x_target, energy_min)
print_control_summary("u2_almost_min", traj_almost_min, x_target, energy_almost_min)
print_control_summary("u_fl", traj_fl, x_target, energy_fl)

print("=================================================\n")

# Combined comparison plot
plot_all_trajectories(
    p=p_softmin,
    x_init=x_init,
    x_target=x_target,
    traj_zero=traj_zero,
    traj_min=traj_min,
    traj_almost_min=traj_almost_min,
    traj_fl=traj_fl,
    traj_gd=traj_gd,
    traj_momentum=traj_momentum,
    traj_sgd=traj_sgd,
    xlim=(-2, 8),
    ylim=(-2, 10),
    n_grid=300,
    levels=40,
    filename="10_all_trajectories.png",
)