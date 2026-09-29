# from __future__ import annotations

# from types import SimpleNamespace
# from typing import Any, Optional

# import diffrax as dfx
# import jax
# import jax.numpy as jnp
# from jax import config
# import numpy as np

# from controlled_gradient_flow.control_synthesis import (
#     integrate_control_energy,
#     make_solver_wrapper,
#     setup_control_synthesis,
#     solve_almost_gramian_synthesis,
#     solve_controlled_ivp,
#     solve_gramian_synthesis,
#     solve_nominal_ivp,
# )

# # Comment this out if want lower precision for faster computation, but may cause issues with convergence for some problems
# # config.update("jax_enable_x64", True)

# Array = jnp.ndarray


# def build_control_affine_gradient_flow_system(
#     vector_field,
#     state_dimension: int = 2,
# ):
#     input_dimension = state_dimension

#     jacobian_vector_field = jax.jit(jax.jacfwd(lambda state: vector_field(state)))
#     jitted_vector_field = jax.jit(vector_field)

#     def input_matrix(_, __):
#         return jnp.eye(state_dimension, input_dimension)

#     def autonomous_dynamics(time, state):
#         del time
#         return jitted_vector_field(state)

#     def controlled_dynamics(time, state, control):
#         return autonomous_dynamics(time, state) + input_matrix(time, state) @ control

#     def autonomous_jacobian(time, state):
#         del time
#         return jacobian_vector_field(state)

#     def controlled_jacobian(time, state, control):
#         del time, control
#         return jacobian_vector_field(state)

#     return SimpleNamespace(
#         state_dim=state_dimension,
#         input_dim=input_dimension,
#         N_fn=autonomous_dynamics,
#         B_fn=input_matrix,
#         F_fn=controlled_dynamics,
#         jacobian_N=autonomous_jacobian,
#         jacobian_F=controlled_jacobian,
#         dynamics=controlled_dynamics,
#         N=vector_field,
#     )


# def create_ode_solver_interface(
#     relative_tolerance: float = 1e-6,
#     absolute_tolerance: float = 1e-6,
#     maximum_steps: int = 1000000,
#     initial_step_size=None,
# ):
#     return make_solver_wrapper(
#         solver=dfx.Dopri8(),
#         stepsize_controller=dfx.PIDController(
#             rtol=relative_tolerance,
#             atol=absolute_tolerance,
#         ),
#         max_steps=maximum_steps,
#         dt0=initial_step_size,
#     )


# # def create_control_synthesis_problem(
# #     system: Any,
# #     initial_state: Array,
# #     target_state: Array,
# #     initial_time: float,
# #     terminal_time: float,
# #     ode_solver_interface: Any,
# #     seed_control: Optional[str] = None,
# #     integration_method: str = "simpsons",
# #     integration_samples: int = 50000,
# #     maximum_iterations: int = 15,
# #     minimum_iterations: int = 5,
# #     record_samples: int = 5000,
# #     interpolation_samples: Optional[int] = 50000,
# # ):
# #     return setup_control_synthesis(
# #         system=system,
# #         init_state=initial_state,
# #         target_state=target_state,
# #         t0=initial_time,
# #         t1=terminal_time,
# #         ODE_solver_wrapper=ode_solver_interface,
# #         seed_control=seed_control,
# #         tol_x=1e-9,
# #         tol_u=1e-9,
# #         forward_mode=False,
# #         integration_method=integration_method,
# #         integration_samples=integration_samples,
# #         max_iterations=maximum_iterations,
# #         min_iterations=minimum_iterations,
# #         linear_solver_method="ls",
# #         linear_solver_eps=0.0,
# #         record_samples=record_samples,
# #         interpolation_samples=interpolation_samples,
# #     )

# def create_control_synthesis_problem(
#     system: Any,
#     initial_state: Array,
#     target_state: Array,
#     initial_time: float,
#     terminal_time: float,
#     ode_solver_interface: Any,
#     seed_control: Optional[str] = None,
#     integration_method: str = "simpsons",
#     integration_samples: int = 500,
#     maximum_iterations: int = 5,
#     minimum_iterations: int = 2,
#     record_samples: int = 200,
#     interpolation_samples: Optional[int] = 500,
# ):
#     return setup_control_synthesis(
#         system=system,
#         init_state=initial_state,
#         target_state=target_state,
#         t0=initial_time,
#         t1=terminal_time,
#         ODE_solver_wrapper=ode_solver_interface,
#         seed_control=seed_control,
#         tol_x=1e-3,
#         tol_u=1e-3,
#         forward_mode=False,
#         integration_method=integration_method,
#         integration_samples=integration_samples,
#         max_iterations=maximum_iterations,
#         min_iterations=minimum_iterations,
#         linear_solver_method="ls",
#         linear_solver_eps=1e-8,
#         record_samples=record_samples,
#         interpolation_samples=interpolation_samples,
#     )



# def create_feedback_linearization_control(
#     system: Any,
#     initial_state: Array,
#     target_state: Array,
#     initial_time: float,
#     terminal_time: float,
# ):
#     transfer_duration = terminal_time - initial_time

#     def feedback_linearizing_control(time: float) -> Array:
#         reference_velocity = (target_state - initial_state) / transfer_duration
#         reference_state = initial_state + (
#             (time - initial_time) / transfer_duration
#         ) * (target_state - initial_state)
#         return reference_velocity - system.N_fn(time, reference_state)

#     return feedback_linearizing_control


# def simulate_control_strategy_comparison(
#     system,
#     synthesis_problem,
#     initial_state,
#     target_state,
#     initial_time,
#     terminal_time,
#     ode_solver_interface,
# ):
#     minimum_energy_solution = solve_almost_gramian_synthesis(
#         control_obj=synthesis_problem
#     )
#     minimum_energy_problem, minimum_energy_logs, minimum_energy_history, _ = (
#         minimum_energy_solution
#     )

#     approximate_minimum_energy_solution = solve_gramian_synthesis(
#         control_obj=synthesis_problem
#     )
#     (
#         approximate_minimum_energy_problem,
#         approximate_minimum_energy_logs,
#         approximate_minimum_energy_history,
#         _,
#     ) = approximate_minimum_energy_solution

#     if not minimum_energy_history:
#         return {
#             "minimum_energy_problem": minimum_energy_problem,
#             "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
#             "minimum_energy_logs": minimum_energy_logs,
#             "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
#             "minimum_energy_history": [],
#             "approximate_minimum_energy_history": approximate_minimum_energy_history,
#             "uncontrolled_state_trajectory": None,
#             "minimum_energy_state_trajectory": None,
#             "approximate_minimum_energy_state_trajectory": None,
#             "feedback_linearization_state_trajectory": None,
#             "uncontrolled_energy": 0.0,
#             "minimum_energy_control_energy": np.inf,
#             "approximate_minimum_energy_control_energy": np.inf,
#             "feedback_linearization_control_energy": np.inf,
#             "failed": True,
#             "failure_reason": "empty minimum_energy_history",
#         }

#     if not approximate_minimum_energy_history:
#         return {
#             "minimum_energy_problem": minimum_energy_problem,
#             "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
#             "minimum_energy_logs": minimum_energy_logs,
#             "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
#             "minimum_energy_history": minimum_energy_history,
#             "approximate_minimum_energy_history": [],
#             "uncontrolled_state_trajectory": None,
#             "minimum_energy_state_trajectory": None,
#             "approximate_minimum_energy_state_trajectory": None,
#             "feedback_linearization_state_trajectory": None,
#             "uncontrolled_energy": 0.0,
#             "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
#             "approximate_minimum_energy_control_energy": np.inf,
#             "feedback_linearization_control_energy": np.inf,
#             "failed": True,
#             "failure_reason": "empty approximate_minimum_energy_history",
#         }

#     minimum_energy_control = minimum_energy_history[-1].u
#     approximate_minimum_energy_control = approximate_minimum_energy_history[-1].u
#     feedback_linearization_control = create_feedback_linearization_control(
#         system,
#         initial_state,
#         target_state,
#         initial_time,
#         terminal_time,
#     )

#     saveat = dfx.SaveAt(ts=synthesis_problem.record_ts)

#     uncontrolled_solution = solve_nominal_ivp(
#         system=system,
#         x0=initial_state,
#         t0=initial_time,
#         t1=terminal_time,
#         saveat=saveat,
#         ODE_solver_wrapper=ode_solver_interface,
#     )
#     minimum_energy_trajectory = solve_controlled_ivp(
#         system=system,
#         x0=initial_state,
#         u_fn=minimum_energy_control,
#         t0=initial_time,
#         t1=terminal_time,
#         saveat=saveat,
#         ODE_solver_wrapper=ode_solver_interface,
#     )
#     approximate_minimum_energy_trajectory = solve_controlled_ivp(
#         system=system,
#         x0=initial_state,
#         u_fn=approximate_minimum_energy_control,
#         t0=initial_time,
#         t1=terminal_time,
#         saveat=saveat,
#         ODE_solver_wrapper=ode_solver_interface,
#     )
#     feedback_linearization_trajectory = solve_controlled_ivp(
#         system=system,
#         x0=initial_state,
#         u_fn=feedback_linearization_control,
#         t0=initial_time,
#         t1=terminal_time,
#         saveat=saveat,
#         ODE_solver_wrapper=ode_solver_interface,
#     )

#     feedback_linearization_energy = float(
#         integrate_control_energy(
#             feedback_linearization_control,
#             synthesis_problem.integration_ts,
#             synthesis_problem.integrator,
#         )
#     )

#     return {
#         "minimum_energy_problem": minimum_energy_problem,
#         "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
#         "minimum_energy_logs": minimum_energy_logs,
#         "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
#         "minimum_energy_history": minimum_energy_history,
#         "approximate_minimum_energy_history": approximate_minimum_energy_history,
#         "uncontrolled_state_trajectory": uncontrolled_solution.ys,
#         "minimum_energy_state_trajectory": minimum_energy_trajectory.ys,
#         "approximate_minimum_energy_state_trajectory": (
#             approximate_minimum_energy_trajectory.ys
#         ),
#         "feedback_linearization_state_trajectory": (
#             feedback_linearization_trajectory.ys
#         ),
#         "uncontrolled_energy": 0.0,
#         "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
#         "approximate_minimum_energy_control_energy": float(
#             approximate_minimum_energy_history[-1].energy
#         ),
#         "feedback_linearization_control_energy": feedback_linearization_energy,
#         "failed": False,
#         "failure_reason": "",
#     }

# def simulate_minimum_energy(
#     system,
#     synthesis_problem,
#     initial_state,
#     target_state,
#     initial_time,
#     terminal_time,
#     ode_solver_interface,
# ):
#     minimum_energy_solution = solve_almost_gramian_synthesis(
#         control_obj=synthesis_problem
#     )
#     minimum_energy_problem, minimum_energy_logs, minimum_energy_history, _ = (
#         minimum_energy_solution
#     )

#     if not minimum_energy_history:
#         return {
#             "minimum_energy_problem": minimum_energy_problem,
#             "minimum_energy_logs": minimum_energy_logs,
#             "minimum_energy_history": [],
#             "minimum_energy_state_trajectory": None,
#             "minimum_energy_control_energy": np.inf,
#             "failed": True,
#             "failure_reason": "empty minimum_energy_history",
#         }

#     minimum_energy_control = minimum_energy_history[-1].u

#     saveat = dfx.SaveAt(ts=synthesis_problem.record_ts)

#     minimum_energy_trajectory = solve_controlled_ivp(
#         system=system,
#         x0=initial_state,
#         u_fn=minimum_energy_control,
#         t0=initial_time,
#         t1=terminal_time,
#         saveat=saveat,
#         ODE_solver_wrapper=ode_solver_interface,
#     )

#     return {
#         "minimum_energy_problem": minimum_energy_problem,
#         "minimum_energy_logs": minimum_energy_logs,
#         "minimum_energy_history": minimum_energy_history,
#         "minimum_energy_state_trajectory": minimum_energy_trajectory.ys,
#         "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
#         "failed": False,
#         "failure_reason": "",
#     }

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from controlled_gradient_flow.control_synthesis import (
    integrate_control_energy,
    make_solver_wrapper,
    setup_control_synthesis,
    solve_almost_gramian_synthesis,
    solve_controlled_ivp,
    solve_gramian_synthesis,
    solve_nominal_ivp,
)

# Uncomment for higher precision, but it can increase memory usage substantially.
# config.update("jax_enable_x64", True)

Array = jnp.ndarray


def build_control_affine_gradient_flow_system(
    vector_field,
    state_dimension: int = 2,
):
    input_dimension = state_dimension

    jacobian_vector_field = jax.jit(jax.jacfwd(lambda state: vector_field(state)))
    jitted_vector_field = jax.jit(vector_field)

    def input_matrix(_, __):
        return jnp.eye(state_dimension, input_dimension)

    def autonomous_dynamics(time, state):
        del time
        return jitted_vector_field(state)

    def controlled_dynamics(time, state, control):
        return autonomous_dynamics(time, state) + input_matrix(time, state) @ control

    def autonomous_jacobian(time, state):
        del time
        return jacobian_vector_field(state)

    def controlled_jacobian(time, state, control):
        del time, control
        return jacobian_vector_field(state)

    return SimpleNamespace(
        state_dim=state_dimension,
        input_dim=input_dimension,
        N_fn=autonomous_dynamics,
        B_fn=input_matrix,
        F_fn=controlled_dynamics,
        jacobian_N=autonomous_jacobian,
        jacobian_F=controlled_jacobian,
        dynamics=controlled_dynamics,
        N=vector_field,
    )


def create_ode_solver_interface(
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-6,
    maximum_steps: int = 200000,
    initial_step_size=None,
):
    return make_solver_wrapper(
        solver=dfx.Dopri8(),
        stepsize_controller=dfx.PIDController(
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ),
        max_steps=maximum_steps,
        dt0=initial_step_size,
    )


def create_control_synthesis_problem(
    system: Any,
    initial_state: Array,
    target_state: Array,
    initial_time: float,
    terminal_time: float,
    ode_solver_interface: Any,
    seed_control: Optional[str] = None,
    integration_method: str = "simpsons",
    integration_samples: int = 500,
    maximum_iterations: int = 5,
    minimum_iterations: int = 2,
    record_samples: int = 200,
    interpolation_samples: Optional[int] = 500,
):
    return setup_control_synthesis(
        system=system,
        init_state=initial_state,
        target_state=target_state,
        t0=initial_time,
        t1=terminal_time,
        ODE_solver_wrapper=ode_solver_interface,
        seed_control=seed_control,
        tol_x=1e-3,
        tol_u=1e-3,
        forward_mode=False,
        integration_method=integration_method,
        integration_samples=integration_samples,
        max_iterations=maximum_iterations,
        min_iterations=minimum_iterations,
        linear_solver_method="ls",
        linear_solver_eps=1e-8,
        record_samples=record_samples,
        interpolation_samples=interpolation_samples,
    )


def create_feedback_linearization_control(
    system: Any,
    initial_state: Array,
    target_state: Array,
    initial_time: float,
    terminal_time: float,
):
    transfer_duration = terminal_time - initial_time

    def feedback_linearizing_control(time: float) -> Array:
        reference_velocity = (target_state - initial_state) / transfer_duration
        reference_state = initial_state + (
            (time - initial_time) / transfer_duration
        ) * (target_state - initial_state)
        return reference_velocity - system.N_fn(time, reference_state)

    return feedback_linearizing_control


def simulate_control_strategy_comparison(
    system,
    synthesis_problem,
    initial_state,
    target_state,
    initial_time,
    terminal_time,
    ode_solver_interface,
):
    minimum_energy_solution = solve_almost_gramian_synthesis(
        control_obj=synthesis_problem
    )
    minimum_energy_problem, minimum_energy_logs, minimum_energy_history, _ = (
        minimum_energy_solution
    )

    approximate_minimum_energy_solution = solve_gramian_synthesis(
        control_obj=synthesis_problem
    )
    (
        approximate_minimum_energy_problem,
        approximate_minimum_energy_logs,
        approximate_minimum_energy_history,
        _,
    ) = approximate_minimum_energy_solution

    if not minimum_energy_history:
        return {
            "minimum_energy_problem": minimum_energy_problem,
            "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
            "minimum_energy_logs": minimum_energy_logs,
            "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
            "minimum_energy_history": [],
            "approximate_minimum_energy_history": approximate_minimum_energy_history,
            "uncontrolled_state_trajectory": None,
            "minimum_energy_state_trajectory": None,
            "approximate_minimum_energy_state_trajectory": None,
            "feedback_linearization_state_trajectory": None,
            "uncontrolled_energy": 0.0,
            "minimum_energy_control_energy": np.inf,
            "approximate_minimum_energy_control_energy": np.inf,
            "feedback_linearization_control_energy": np.inf,
            "failed": True,
            "failure_reason": "empty minimum_energy_history",
        }

    if not approximate_minimum_energy_history:
        return {
            "minimum_energy_problem": minimum_energy_problem,
            "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
            "minimum_energy_logs": minimum_energy_logs,
            "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
            "minimum_energy_history": minimum_energy_history,
            "approximate_minimum_energy_history": [],
            "uncontrolled_state_trajectory": None,
            "minimum_energy_state_trajectory": None,
            "approximate_minimum_energy_state_trajectory": None,
            "feedback_linearization_state_trajectory": None,
            "uncontrolled_energy": 0.0,
            "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
            "approximate_minimum_energy_control_energy": np.inf,
            "feedback_linearization_control_energy": np.inf,
            "failed": True,
            "failure_reason": "empty approximate_minimum_energy_history",
        }

    minimum_energy_control = minimum_energy_history[-1].u
    approximate_minimum_energy_control = approximate_minimum_energy_history[-1].u
    feedback_linearization_control = create_feedback_linearization_control(
        system,
        initial_state,
        target_state,
        initial_time,
        terminal_time,
    )

    saveat = dfx.SaveAt(ts=synthesis_problem.record_ts)

    uncontrolled_solution = solve_nominal_ivp(
        system=system,
        x0=initial_state,
        t0=initial_time,
        t1=terminal_time,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_interface,
    )
    minimum_energy_trajectory = solve_controlled_ivp(
        system=system,
        x0=initial_state,
        u_fn=minimum_energy_control,
        t0=initial_time,
        t1=terminal_time,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_interface,
    )
    approximate_minimum_energy_trajectory = solve_controlled_ivp(
        system=system,
        x0=initial_state,
        u_fn=approximate_minimum_energy_control,
        t0=initial_time,
        t1=terminal_time,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_interface,
    )
    feedback_linearization_trajectory = solve_controlled_ivp(
        system=system,
        x0=initial_state,
        u_fn=feedback_linearization_control,
        t0=initial_time,
        t1=terminal_time,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_interface,
    )

    feedback_linearization_energy = float(
        integrate_control_energy(
            feedback_linearization_control,
            synthesis_problem.integration_ts,
            synthesis_problem.integrator,
        )
    )

    return {
        "minimum_energy_problem": minimum_energy_problem,
        "approximate_minimum_energy_problem": approximate_minimum_energy_problem,
        "minimum_energy_logs": minimum_energy_logs,
        "approximate_minimum_energy_logs": approximate_minimum_energy_logs,
        "minimum_energy_history": minimum_energy_history,
        "approximate_minimum_energy_history": approximate_minimum_energy_history,
        "uncontrolled_state_trajectory": uncontrolled_solution.ys,
        "minimum_energy_state_trajectory": minimum_energy_trajectory.ys,
        "approximate_minimum_energy_state_trajectory": (
            approximate_minimum_energy_trajectory.ys
        ),
        "feedback_linearization_state_trajectory": (
            feedback_linearization_trajectory.ys
        ),
        "uncontrolled_energy": 0.0,
        "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
        "approximate_minimum_energy_control_energy": float(
            approximate_minimum_energy_history[-1].energy
        ),
        "feedback_linearization_control_energy": feedback_linearization_energy,
        "failed": False,
        "failure_reason": "",
    }


def simulate_minimum_energy(
    system,
    synthesis_problem,
    initial_state,
    target_state,
    initial_time,
    terminal_time,
    ode_solver_interface,
):
    minimum_energy_solution = solve_almost_gramian_synthesis(
        control_obj=synthesis_problem
    )
    minimum_energy_problem, minimum_energy_logs, minimum_energy_history, _ = (
        minimum_energy_solution
    )

    if not minimum_energy_history:
        return {
            "minimum_energy_problem": minimum_energy_problem,
            "minimum_energy_logs": minimum_energy_logs,
            "minimum_energy_history": [],
            "minimum_energy_state_trajectory": None,
            "minimum_energy_control_energy": np.inf,
            "failed": True,
            "failure_reason": "empty minimum_energy_history",
        }

    minimum_energy_control = minimum_energy_history[-1].u
    saveat = dfx.SaveAt(ts=synthesis_problem.record_ts)

    minimum_energy_trajectory = solve_controlled_ivp(
        system=system,
        x0=initial_state,
        u_fn=minimum_energy_control,
        t0=initial_time,
        t1=terminal_time,
        saveat=saveat,
        ODE_solver_wrapper=ode_solver_interface,
    )

    ys = minimum_energy_trajectory.ys

    if ys is None:
        return {
            "minimum_energy_problem": minimum_energy_problem,
            "minimum_energy_logs": minimum_energy_logs,
            "minimum_energy_history": minimum_energy_history,
            "minimum_energy_state_trajectory": None,
            "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
            "failed": True,
            "failure_reason": "controlled_ivp returned ys=None",
        }

    try:
        if len(ys) == 0:
            return {
                "minimum_energy_problem": minimum_energy_problem,
                "minimum_energy_logs": minimum_energy_logs,
                "minimum_energy_history": minimum_energy_history,
                "minimum_energy_state_trajectory": None,
                "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
                "failed": True,
                "failure_reason": "controlled_ivp returned empty ys",
            }
    except TypeError:
        pass

    ys_array = np.asarray(ys)
    if not np.all(np.isfinite(ys_array)):
        return {
            "minimum_energy_problem": minimum_energy_problem,
            "minimum_energy_logs": minimum_energy_logs,
            "minimum_energy_history": minimum_energy_history,
            "minimum_energy_state_trajectory": None,
            "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
            "failed": True,
            "failure_reason": "controlled_ivp returned non-finite ys",
        }

    return {
        "minimum_energy_problem": minimum_energy_problem,
        "minimum_energy_logs": minimum_energy_logs,
        "minimum_energy_history": minimum_energy_history,
        "minimum_energy_state_trajectory": ys,
        "minimum_energy_control_energy": float(minimum_energy_history[-1].energy),
        "failed": False,
        "failure_reason": "",
    }
