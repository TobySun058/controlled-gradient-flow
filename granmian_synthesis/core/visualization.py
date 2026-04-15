from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from granmian_synthesis.core.objective import evaluate_smooth_minimum_objective

OUTPUT_DIRECTORY = Path("plots")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def save_figure_to_disk(figure, filename: str, dpi: int = 300):
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def evaluate_objective_on_grid(parameters, x_limits, y_limits, grid_resolution):
    x_coordinates = jnp.linspace(x_limits[0], x_limits[1], grid_resolution)
    y_coordinates = jnp.linspace(y_limits[0], y_limits[1], grid_resolution)
    x_mesh, y_mesh = jnp.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    evaluation_points = jnp.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)
    objective_values = jax.vmap(
        lambda state: evaluate_smooth_minimum_objective(state, parameters)
    )(evaluation_points).reshape((grid_resolution, grid_resolution))
    return np.array(x_mesh), np.array(y_mesh), np.array(objective_values)


def plot_comparative_trajectory_map(
    parameters,
    initial_state,
    target_state,
    uncontrolled_state_trajectory,
    minimum_energy_state_trajectory,
    approximate_minimum_energy_state_trajectory,
    feedback_linearization_state_trajectory,
    gradient_descent_trajectory,
    momentum_gradient_descent_trajectory,
    stochastic_gradient_descent_trajectory,
    x_limits=(-2, 8),
    y_limits=(-2, 10),
    grid_resolution=250,
    contour_level_count=28,
    filename="comparative_trajectory_map.png",
):
    x_mesh, y_mesh, objective_values = evaluate_objective_on_grid(
        parameters,
        x_limits,
        y_limits,
        grid_resolution,
    )

    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.contour(x_mesh, y_mesh, objective_values, levels=contour_level_count, linewidths=0.8)

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
        label="Feedback linearization",
    )
    axis.plot(
        np.array(gradient_descent_trajectory[:, 0]),
        np.array(gradient_descent_trajectory[:, 1]),
        linewidth=1.8,
        linestyle="--",
        label="Gradient descent",
    )
    axis.plot(
        np.array(momentum_gradient_descent_trajectory[:, 0]),
        np.array(momentum_gradient_descent_trajectory[:, 1]),
        linewidth=1.8,
        linestyle="--",
        label="Momentum gradient descent",
    )
    axis.plot(
        np.array(stochastic_gradient_descent_trajectory[:, 0]),
        np.array(stochastic_gradient_descent_trajectory[:, 1]),
        linewidth=1.8,
        linestyle=":",
        label="Stochastic gradient descent",
    )

    axis.scatter([float(initial_state[0])], [float(initial_state[1])], marker="o", s=60, label="Initial state")
    axis.scatter([float(target_state[0])], [float(target_state[1])], marker="*", s=140, label="Target state")

    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(
        f"Comparative trajectory map (softmin sharpness = {parameters.softmin_sharpness:.2f}, T = 0.75)"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    save_figure_to_disk(figure, filename)


def plot_reduced_control_comparison_map(
    parameters,
    initial_state,
    target_state,
    minimum_energy_state_trajectory,
    approximate_minimum_energy_state_trajectory,
    feedback_linearization_state_trajectory,
    x_limits=(-2, 8),
    y_limits=(-2, 10),
    grid_resolution=250,
    contour_level_count=28,
    filename="reduced_control_comparison_map.png",
):
    x_mesh, y_mesh, objective_values = evaluate_objective_on_grid(
        parameters,
        x_limits,
        y_limits,
        grid_resolution,
    )

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    axis.contour(x_mesh, y_mesh, objective_values, levels=contour_level_count, linewidths=0.8)
    axis.plot(
        np.array(minimum_energy_state_trajectory[:, 0]),
        np.array(minimum_energy_state_trajectory[:, 1]),
        linewidth=2.2,
        label="Minimum-energy synthesis",
    )
    axis.plot(
        np.array(approximate_minimum_energy_state_trajectory[:, 0]),
        np.array(approximate_minimum_energy_state_trajectory[:, 1]),
        linewidth=2.2,
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
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(
        f"Reduced control comparison map (softmin sharpness = {parameters.softmin_sharpness:.2f}, T = 0.75)"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure_to_disk(figure, filename)