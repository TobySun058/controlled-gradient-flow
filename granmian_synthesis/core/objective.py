from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

Array = jnp.ndarray
VectorField = Callable[[Array], Array]


@dataclass(frozen=True)
class SmoothMinimumObjectiveParameters:
    centers: Array
    quadratic_matrices: Array
    offsets: Array
    softmin_sharpness: float


def evaluate_quadratic_component(
    state: Array,
    component_index: int,
    parameters: SmoothMinimumObjectiveParameters,
) -> Array:
    displacement = state - parameters.centers[component_index]
    return 0.5 * (
        displacement @ parameters.quadratic_matrices[component_index] @ displacement
    ) + parameters.offsets[component_index]


def evaluate_quadratic_component_gradient(
    state: Array,
    component_index: int,
    parameters: SmoothMinimumObjectiveParameters,
) -> Array:
    return parameters.quadratic_matrices[component_index] @ (
        state - parameters.centers[component_index]
    )


def evaluate_smooth_minimum_objective(
    state: Array,
    parameters: SmoothMinimumObjectiveParameters,
) -> Array:
    quadratic_values = jnp.stack(
        [
            evaluate_quadratic_component(state, component_index, parameters)
            for component_index in range(3)
        ],
        axis=0,
    )
    minimum_value = jnp.min(quadratic_values)
    normalized_exponential_sum = jnp.sum(
        jnp.exp(-parameters.softmin_sharpness * (quadratic_values - minimum_value))
    )
    return minimum_value - (1.0 / parameters.softmin_sharpness) * jnp.log(
        normalized_exponential_sum
    )


def evaluate_smooth_minimum_gradient(
    state: Array,
    parameters: SmoothMinimumObjectiveParameters,
) -> Array:
    quadratic_values = jnp.stack(
        [
            evaluate_quadratic_component(state, component_index, parameters)
            for component_index in range(3)
        ],
        axis=0,
    )
    minimum_value = jnp.min(quadratic_values)
    unnormalized_weights = jnp.exp(
        -parameters.softmin_sharpness * (quadratic_values - minimum_value)
    )
    normalized_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    component_gradients = jnp.stack(
        [
            evaluate_quadratic_component_gradient(state, component_index, parameters)
            for component_index in range(3)
        ],
        axis=0,
    )
    return jnp.sum(normalized_weights[:, None] * component_gradients, axis=0)


def create_uncontrolled_gradient_flow(
    parameters: SmoothMinimumObjectiveParameters,
) -> VectorField:
    return lambda state: -evaluate_smooth_minimum_gradient(state, parameters)


def create_default_objective_parameters(
    softmin_sharpness: float,
) -> SmoothMinimumObjectiveParameters:
    return SmoothMinimumObjectiveParameters(
        centers=jnp.array([[0.0, 0.0], [4.0, 2.0], [1.0, 6.0]]),
        quadratic_matrices=jnp.array([jnp.eye(2), jnp.eye(2), jnp.eye(2)]),
        offsets=jnp.array([2.0, 6.0, 8.0]),
        softmin_sharpness=float(softmin_sharpness),
    )