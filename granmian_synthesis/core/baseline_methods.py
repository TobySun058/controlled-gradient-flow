from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from granmian_synthesis.core.objective import evaluate_smooth_minimum_gradient


def compute_gradient_descent_trajectory(
    initial_state,
    parameters,
    step_size: float = 0.10,
    iteration_count: int = 140,
):
    trajectory = np.zeros((iteration_count + 1, 2))
    trajectory[0] = np.array(initial_state, dtype=float)
    for iteration in range(iteration_count):
        gradient = np.array(
            evaluate_smooth_minimum_gradient(jnp.array(trajectory[iteration]), parameters)
        )
        trajectory[iteration + 1] = trajectory[iteration] - step_size * gradient
    return trajectory


def compute_momentum_gradient_descent_trajectory(
    initial_state,
    parameters,
    step_size: float = 0.07,
    momentum_coefficient: float = 0.90,
    iteration_count: int = 140,
):
    trajectory = np.zeros((iteration_count + 1, 2))
    trajectory[0] = np.array(initial_state, dtype=float)

    initial_gradient = np.array(
        evaluate_smooth_minimum_gradient(jnp.array(trajectory[0]), parameters)
    )
    trajectory[1] = trajectory[0] - step_size * initial_gradient

    for iteration in range(1, iteration_count):
        gradient = np.array(
            evaluate_smooth_minimum_gradient(jnp.array(trajectory[iteration]), parameters)
        )
        trajectory[iteration + 1] = (
            trajectory[iteration]
            - step_size * gradient
            + momentum_coefficient * (trajectory[iteration] - trajectory[iteration - 1])
        )
    return trajectory


def compute_stochastic_gradient_descent_trajectory(
    initial_state,
    parameters,
    step_size: float = 0.10,
    noise_standard_deviation: float = 0.15,
    iteration_count: int = 140,
    random_seed: int = 0,
):
    random_number_generator = np.random.default_rng(random_seed)
    trajectory = np.zeros((iteration_count + 1, 2))
    trajectory[0] = np.array(initial_state, dtype=float)

    for iteration in range(iteration_count):
        gradient = np.array(
            evaluate_smooth_minimum_gradient(jnp.array(trajectory[iteration]), parameters)
        )
        stochastic_perturbation = random_number_generator.normal(
            0.0,
            noise_standard_deviation,
            size=2,
        )
        trajectory[iteration + 1] = trajectory[iteration] - step_size * (
            gradient + stochastic_perturbation
        )
    return trajectory