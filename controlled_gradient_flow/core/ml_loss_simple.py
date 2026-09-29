from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

Array = jnp.ndarray
VectorField = Callable[[Array], Array]


@dataclass(frozen=True)
class SyntheticLinearRegressionParameters:
    features: Array   # shape (N, 2)
    targets: Array    # shape (N,)


def create_synthetic_linear_regression_parameters(
    sample_count: int = 200,
    noise_standard_deviation: float = 0.10,
    random_seed: int = 0,
) -> SyntheticLinearRegressionParameters:
    key = jax.random.PRNGKey(random_seed)
    key_x, key_noise = jax.random.split(key)

    features = jax.random.normal(key_x, shape=(sample_count, 2))

    true_parameter = jnp.array([2.0, -1.5])
    noise = noise_standard_deviation * jax.random.normal(
        key_noise, shape=(sample_count,)
    )
    targets = features @ true_parameter + noise

    return SyntheticLinearRegressionParameters(
        features=features,
        targets=targets,
    )


def evaluate_linear_regression_loss(
    parameter: Array,
    parameters: SyntheticLinearRegressionParameters,
) -> Array:
    predictions = parameters.features @ parameter
    residuals = predictions - parameters.targets
    return 0.5 * jnp.mean(residuals**2)


def evaluate_linear_regression_gradient(
    parameter: Array,
    parameters: SyntheticLinearRegressionParameters,
) -> Array:
    loss_function = lambda p: evaluate_linear_regression_loss(p, parameters)
    return jax.grad(loss_function)(parameter)


def create_linear_regression_gradient_flow(
    parameters: SyntheticLinearRegressionParameters,
) -> VectorField:
    return lambda state: -evaluate_linear_regression_gradient(state, parameters)


def solve_closed_form_linear_regression_minimizer(
    parameters: SyntheticLinearRegressionParameters,
) -> Array:
    x = parameters.features
    y = parameters.targets
    gram_matrix = x.T @ x
    right_hand_side = x.T @ y
    return jnp.linalg.solve(gram_matrix, right_hand_side)