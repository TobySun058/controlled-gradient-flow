from __future__ import annotations

import jax.numpy as jnp

from controlled_gradient_flow.core.baseline_methods import (
    compute_gradient_descent_trajectory,
    compute_momentum_gradient_descent_trajectory,
    compute_stochastic_gradient_descent_trajectory,
)
from controlled_gradient_flow.core.visualization import (
    plot_comparative_trajectory_map,
    plot_reduced_control_comparison_map,
)
from controlled_gradient_flow.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from controlled_gradient_flow.core.objective import (
    create_default_objective_parameters,
    create_uncontrolled_gradient_flow,
)


def main(softmin_sharpness: float = 0.5, terminal_time: float = 0.75):
    parameters = create_default_objective_parameters(softmin_sharpness)
    initial_state = jnp.array([2.0, 6.0])
    target_state = jnp.array([4.5, 1.5])
    initial_time = 0.0

    system = build_control_affine_gradient_flow_system(
        create_uncontrolled_gradient_flow(parameters),
        state_dimension=2,
    )
    ode_solver_interface = create_ode_solver_interface()
    synthesis_problem = create_control_synthesis_problem(
        system=system,
        initial_state=initial_state,
        target_state=target_state,
        initial_time=initial_time,
        terminal_time=terminal_time,
        ode_solver_interface=ode_solver_interface,
    )

    comparison_results = simulate_control_strategy_comparison(
        system,
        synthesis_problem,
        initial_state,
        target_state,
        initial_time,
        terminal_time,
        ode_solver_interface,
    )
    gradient_descent_trajectory = compute_gradient_descent_trajectory(
        initial_state,
        parameters,
        step_size=0.10,
        iteration_count=140,
    )
    momentum_gradient_descent_trajectory = (
        compute_momentum_gradient_descent_trajectory(
            initial_state,
            parameters,
            step_size=0.07,
            momentum_coefficient=0.90,
            iteration_count=140,
        )
    )
    stochastic_gradient_descent_trajectory = (
        compute_stochastic_gradient_descent_trajectory(
            initial_state,
            parameters,
            step_size=0.10,
            noise_standard_deviation=0.15,
            iteration_count=140,
            random_seed=0,
        )
    )

    plot_comparative_trajectory_map(
        parameters=parameters,
        initial_state=initial_state,
        target_state=target_state,
        uncontrolled_state_trajectory=comparison_results[
            "uncontrolled_state_trajectory"
        ],
        minimum_energy_state_trajectory=comparison_results[
            "minimum_energy_state_trajectory"
        ],
        approximate_minimum_energy_state_trajectory=comparison_results[
            "approximate_minimum_energy_state_trajectory"
        ],
        feedback_linearization_state_trajectory=comparison_results[
            "feedback_linearization_state_trajectory"
        ],
        gradient_descent_trajectory=gradient_descent_trajectory,
        momentum_gradient_descent_trajectory=momentum_gradient_descent_trajectory,
        stochastic_gradient_descent_trajectory=stochastic_gradient_descent_trajectory,
        filename=(
            f"comparative_trajectory_map_softmin_sharpness_{softmin_sharpness:.2f}.png"
        ),
    )

    plot_reduced_control_comparison_map(
        parameters=parameters,
        initial_state=initial_state,
        target_state=target_state,
        minimum_energy_state_trajectory=comparison_results[
            "minimum_energy_state_trajectory"
        ],
        approximate_minimum_energy_state_trajectory=comparison_results[
            "approximate_minimum_energy_state_trajectory"
        ],
        feedback_linearization_state_trajectory=comparison_results[
            "feedback_linearization_state_trajectory"
        ],
        filename=(
            f"reduced_control_comparison_map_softmin_sharpness_{softmin_sharpness:.2f}.png"
        ),
    )


if __name__ == "__main__":
    main()