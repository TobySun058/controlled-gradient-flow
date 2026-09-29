from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from controlled_gradient_flow.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from controlled_gradient_flow.core.ml_loss_simple import (
    create_synthetic_linear_regression_parameters,
    create_linear_regression_gradient_flow,
    evaluate_linear_regression_loss,
    solve_closed_form_linear_regression_minimizer,
)


OUTPUT_DIRECTORY = Path("results/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_initial_target_distance(initial_state, target_state):
    return float(jnp.linalg.norm(jnp.array(initial_state) - jnp.array(target_state)))


def evaluate_objective_on_grid(parameters, x_limits=(-1, 5), y_limits=(-1, 5), grid_resolution=250):
    x_coordinates = jnp.linspace(x_limits[0], x_limits[1], grid_resolution)
    y_coordinates = jnp.linspace(y_limits[0], y_limits[1], grid_resolution)
    x_mesh, y_mesh = jnp.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    evaluation_points = jnp.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)

    objective_values = jax.vmap(
        lambda state: evaluate_linear_regression_loss(state, parameters)
    )(evaluation_points).reshape((grid_resolution, grid_resolution))

    return np.array(x_mesh), np.array(y_mesh), np.array(objective_values)


def plot_case(
    parameters,
    initial_state,
    target_state,
    uncontrolled_state_trajectory,
    minimum_energy_state_trajectory,
    approximate_minimum_energy_state_trajectory,
    feedback_linearization_state_trajectory,
    filename,
    x_limits=(-1, 5),
    y_limits=(-1, 5),
):
    x_mesh, y_mesh, objective_values = evaluate_objective_on_grid(
        parameters,
        x_limits=x_limits,
        y_limits=y_limits,
        grid_resolution=250,
    )

    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.contour(x_mesh, y_mesh, objective_values, levels=30, linewidths=0.8)

    axis.plot(
        np.array(uncontrolled_state_trajectory[:, 0]),
        np.array(uncontrolled_state_trajectory[:, 1]),
        linewidth=2.0,
        label="Uncontrolled dynamics",
    )
    axis.plot(
        np.array(minimum_energy_state_trajectory[:, 0]),
        np.array(minimum_energy_state_trajectory[:, 1]),
        linewidth=2.0,
        label="Minimum-energy synthesis",
    )
    axis.plot(
        np.array(approximate_minimum_energy_state_trajectory[:, 0]),
        np.array(approximate_minimum_energy_state_trajectory[:, 1]),
        linewidth=2.0,
        label="Approximate minimum-energy synthesis",
    )
    axis.plot(
        np.array(feedback_linearization_state_trajectory[:, 0]),
        np.array(feedback_linearization_state_trajectory[:, 1]),
        linewidth=2.0,
        linestyle="--",
        label="Feedback linearization",
    )

    axis.scatter([float(initial_state[0])], [float(initial_state[1])], marker="o", s=60, label="Initial state")
    axis.scatter([float(target_state[0])], [float(target_state[1])], marker="*", s=140, label="Target state")

    axis.set_xlabel("w1")
    axis.set_ylabel("w2")
    axis.set_title("Synthetic ML loss control experiment")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_case(case_label, initial_state, target_state, ml_parameters):
    vector_field = create_linear_regression_gradient_flow(ml_parameters)

    system = build_control_affine_gradient_flow_system(
        vector_field,
        state_dimension=2,
    )

    initial_time = 0.0
    terminal_time = 0.75
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

    minimum_energy_terminal_state = comparison_results["minimum_energy_state_trajectory"][-1]
    minimum_energy_control_energy = comparison_results["minimum_energy_control_energy"]

    initial_target_distance = compute_initial_target_distance(initial_state, target_state)
    energy_per_distance = (
        float(minimum_energy_control_energy) / initial_target_distance
        if initial_target_distance > 0.0
        else np.nan
    )

    plot_case(
        parameters=ml_parameters,
        initial_state=initial_state,
        target_state=target_state,
        uncontrolled_state_trajectory=comparison_results["uncontrolled_state_trajectory"],
        minimum_energy_state_trajectory=comparison_results["minimum_energy_state_trajectory"],
        approximate_minimum_energy_state_trajectory=comparison_results[
            "approximate_minimum_energy_state_trajectory"
        ],
        feedback_linearization_state_trajectory=comparison_results[
            "feedback_linearization_state_trajectory"
        ],
        filename=f"synthetic_ml_control_{case_label}.png",
    )

    return {
        "case_label": case_label,
        "initial_x1": float(initial_state[0]),
        "initial_x2": float(initial_state[1]),
        "target_x1": float(target_state[0]),
        "target_x2": float(target_state[1]),
        "terminal_x1": float(minimum_energy_terminal_state[0]),
        "terminal_x2": float(minimum_energy_terminal_state[1]),
        "initial_target_distance": float(initial_target_distance),
        "terminal_error": compute_terminal_error(minimum_energy_terminal_state, target_state),
        "target_loss": float(evaluate_linear_regression_loss(target_state, ml_parameters)),
        "terminal_loss": float(evaluate_linear_regression_loss(minimum_energy_terminal_state, ml_parameters)),
        "minimum_energy_control_energy": float(minimum_energy_control_energy),
        "energy_per_distance": float(energy_per_distance),
    }


def main():
    ml_parameters = create_synthetic_linear_regression_parameters(
        sample_count=200,
        noise_standard_deviation=0.10,
        random_seed=0,
    )

    initial_state = jnp.array([1.0, 1.0])
    designated_target = jnp.array([4.0, 4.0])
    minimum_target = solve_closed_form_linear_regression_minimizer(ml_parameters)

    results = []
    results.append(
        run_case(
            case_label="designated_target",
            initial_state=initial_state,
            target_state=designated_target,
            ml_parameters=ml_parameters,
        )
    )
    results.append(
        run_case(
            case_label="local_minimum_target",
            initial_state=initial_state,
            target_state=minimum_target,
            ml_parameters=ml_parameters,
        )
    )

    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIRECTORY / "synthetic_ml_control_summary.csv"
    txt_path = OUTPUT_DIRECTORY / "synthetic_ml_control_summary.txt"

    results_df.to_csv(csv_path, index=False)
    with open(txt_path, "w") as file:
        file.write(results_df.to_string(index=False))

    print("\n================ SYNTHETIC ML CONTROL SUMMARY ================\n")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV summary to: {csv_path}")
    print(f"Saved text summary to: {txt_path}")


if __name__ == "__main__":
    main()