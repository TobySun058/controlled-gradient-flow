from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

from granmian_synthesis.core.baseline_methods import (
    compute_gradient_descent_trajectory,
    compute_momentum_gradient_descent_trajectory,
    compute_stochastic_gradient_descent_trajectory,
)
from granmian_synthesis.core.visualization import (
    plot_comparative_trajectory_map,
    plot_reduced_control_comparison_map,
)
from granmian_synthesis.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from granmian_synthesis.core.objective import (
    create_default_objective_parameters,
    create_uncontrolled_gradient_flow,
    evaluate_smooth_minimum_objective,
)


OUTPUT_DIRECTORY = Path("data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_terminal_objective_value(terminal_state, parameters):
    return float(
        evaluate_smooth_minimum_objective(jnp.array(terminal_state), parameters)
    )


def format_method_summary(
    method_name: str,
    terminal_state,
    target_state,
    objective_value: float,
    control_energy=None,
):
    terminal_error = compute_terminal_error(terminal_state, target_state)

    lines = [
        f"{method_name}",
        f"  terminal state   = [{float(terminal_state[0]): .6f}, {float(terminal_state[1]): .6f}]",
        f"  terminal error   = {terminal_error:.8f}",
        f"  terminal objective = {objective_value:.8f}",
    ]

    if control_energy is None:
        lines.append("  control energy   = N/A")
    else:
        lines.append(f"  control energy   = {float(control_energy):.8f}")

    return "\n".join(lines)


def create_summary_row(
    softmin_sharpness: float,
    method_name: str,
    terminal_state,
    target_state,
    parameters,
    control_energy=None,
):
    terminal_error = compute_terminal_error(terminal_state, target_state)
    terminal_objective = compute_terminal_objective_value(terminal_state, parameters)

    return {
        "softmin_sharpness": float(softmin_sharpness),
        "method": method_name,
        "terminal_state_x1": float(terminal_state[0]),
        "terminal_state_x2": float(terminal_state[1]),
        "terminal_error": float(terminal_error),
        "terminal_objective": float(terminal_objective),
        "control_energy": np.nan if control_energy is None else float(control_energy),
    }


def main():
    initial_state = jnp.array([2.0, 6.0])
    target_state = jnp.array([4.5, 1.5])
    initial_time = 0.0
    terminal_time = 0.75

    softmin_sharpness_values = np.arange(0.5, 2.0 + 1e-12, 0.25)

    all_summary_rows = []
    all_summary_text_blocks = []

    for softmin_sharpness in softmin_sharpness_values:
        print(f"\nRunning softmin sharpness = {softmin_sharpness:.2f}")

        parameters = create_default_objective_parameters(float(softmin_sharpness))
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
                "comparative_trajectory_map_"
                f"softmin_sharpness_{softmin_sharpness:.2f}.png"
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
                "reduced_control_comparison_map_"
                f"softmin_sharpness_{softmin_sharpness:.2f}.png"
            ),
        )

        uncontrolled_terminal_state = comparison_results[
            "uncontrolled_state_trajectory"
        ][-1]
        minimum_energy_terminal_state = comparison_results[
            "minimum_energy_state_trajectory"
        ][-1]
        approximate_minimum_energy_terminal_state = comparison_results[
            "approximate_minimum_energy_state_trajectory"
        ][-1]
        feedback_linearization_terminal_state = comparison_results[
            "feedback_linearization_state_trajectory"
        ][-1]
        gradient_descent_terminal_state = gradient_descent_trajectory[-1]
        momentum_gradient_descent_terminal_state = momentum_gradient_descent_trajectory[-1]
        stochastic_gradient_descent_terminal_state = stochastic_gradient_descent_trajectory[-1]

        summary_blocks = []
        summary_blocks.append(
            f"\n==================== SUMMARY: softmin_sharpness = {softmin_sharpness:.2f} ====================\n"
        )

        summary_blocks.append(
            format_method_summary(
                "Uncontrolled dynamics",
                uncontrolled_terminal_state,
                target_state,
                compute_terminal_objective_value(uncontrolled_terminal_state, parameters),
                comparison_results["uncontrolled_energy"],
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Minimum-energy synthesis",
                minimum_energy_terminal_state,
                target_state,
                compute_terminal_objective_value(minimum_energy_terminal_state, parameters),
                comparison_results["minimum_energy_control_energy"],
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Approximate minimum-energy synthesis",
                approximate_minimum_energy_terminal_state,
                target_state,
                compute_terminal_objective_value(
                    approximate_minimum_energy_terminal_state, parameters
                ),
                comparison_results["approximate_minimum_energy_control_energy"],
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Feedback linearization",
                feedback_linearization_terminal_state,
                target_state,
                compute_terminal_objective_value(
                    feedback_linearization_terminal_state, parameters
                ),
                comparison_results["feedback_linearization_control_energy"],
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Gradient descent",
                gradient_descent_terminal_state,
                target_state,
                compute_terminal_objective_value(gradient_descent_terminal_state, parameters),
                None,
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Momentum gradient descent",
                momentum_gradient_descent_terminal_state,
                target_state,
                compute_terminal_objective_value(
                    momentum_gradient_descent_terminal_state, parameters
                ),
                None,
            )
        )
        summary_blocks.append(
            format_method_summary(
                "Stochastic gradient descent",
                stochastic_gradient_descent_terminal_state,
                target_state,
                compute_terminal_objective_value(
                    stochastic_gradient_descent_terminal_state, parameters
                ),
                None,
            )
        )

        summary_blocks.append(
            "\n=========================================================================\n"
        )

        summary_text = "\n\n".join(summary_blocks)
        print(summary_text)
        all_summary_text_blocks.append(summary_text)

        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Uncontrolled dynamics",
                uncontrolled_terminal_state,
                target_state,
                parameters,
                comparison_results["uncontrolled_energy"],
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Minimum-energy synthesis",
                minimum_energy_terminal_state,
                target_state,
                parameters,
                comparison_results["minimum_energy_control_energy"],
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Approximate minimum-energy synthesis",
                approximate_minimum_energy_terminal_state,
                target_state,
                parameters,
                comparison_results["approximate_minimum_energy_control_energy"],
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Feedback linearization",
                feedback_linearization_terminal_state,
                target_state,
                parameters,
                comparison_results["feedback_linearization_control_energy"],
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Gradient descent",
                gradient_descent_terminal_state,
                target_state,
                parameters,
                None,
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Momentum gradient descent",
                momentum_gradient_descent_terminal_state,
                target_state,
                parameters,
                None,
            )
        )
        all_summary_rows.append(
            create_summary_row(
                softmin_sharpness,
                "Stochastic gradient descent",
                stochastic_gradient_descent_terminal_state,
                target_state,
                parameters,
                None,
            )
        )

    summary_dataframe = pd.DataFrame(all_summary_rows)
    summary_dataframe.to_csv(
        OUTPUT_DIRECTORY / "softmin_parameter_sweep_summary.csv",
        index=False,
    )

    with open(OUTPUT_DIRECTORY / "softmin_parameter_sweep_summary.txt", "w") as file:
        file.write("\n\n".join(all_summary_text_blocks))

    print("\nParameter sweep completed.")
    print(f"Saved CSV summary to: {OUTPUT_DIRECTORY / 'softmin_parameter_sweep_summary.csv'}")
    print(f"Saved text summary to: {OUTPUT_DIRECTORY / 'softmin_parameter_sweep_summary.txt'}")


if __name__ == "__main__":
    main()