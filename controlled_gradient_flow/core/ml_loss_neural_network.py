from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
VectorField = Callable[[Array], Array]


@dataclass(frozen=True)
class LargeNetworkDataset:
    features: Array   # shape (N, d)
    labels: Array     # shape (N,)


@dataclass(frozen=True)
class NetworkArchitecture:
    layer_sizes: Tuple[int, ...]


@dataclass(frozen=True)
class FlatParameterMetadata:
    shapes: Tuple[Tuple[int, ...], ...]
    sizes: Tuple[int, ...]


def create_synthetic_binary_classification_dataset(
    sample_count_per_class: int = 200,
    random_seed: int = 0,
) -> LargeNetworkDataset:
    key = jax.random.PRNGKey(random_seed)
    key_a, key_b = jax.random.split(key)

    class0 = jax.random.normal(key_a, shape=(sample_count_per_class, 2)) + jnp.array(
        [-2.0, -2.0]
    )
    class1 = jax.random.normal(key_b, shape=(sample_count_per_class, 2)) + jnp.array(
        [2.0, 2.0]
    )

    features = jnp.concatenate([class0, class1], axis=0)
    labels = jnp.concatenate(
        [
            jnp.zeros(sample_count_per_class),
            jnp.ones(sample_count_per_class),
        ],
        axis=0,
    )

    return LargeNetworkDataset(features=features, labels=labels)


def initialize_mlp_parameters(
    architecture: NetworkArchitecture,
    random_seed: int = 0,
    weight_scale: float = 0.10,
) -> List[Tuple[Array, Array]]:
    keys = jax.random.split(
        jax.random.PRNGKey(random_seed),
        num=len(architecture.layer_sizes) - 1,
    )

    parameters = []
    for key, input_dim, output_dim in zip(
        keys,
        architecture.layer_sizes[:-1],
        architecture.layer_sizes[1:],
    ):
        weight_key, bias_key = jax.random.split(key)
        weights = weight_scale * jax.random.normal(weight_key, shape=(input_dim, output_dim))
        biases = jnp.zeros((output_dim,))
        parameters.append((weights, biases))

    return parameters


def mlp_forward(parameters: List[Tuple[Array, Array]], inputs: Array) -> Array:
    activations = inputs
    for layer_index, (weights, biases) in enumerate(parameters):
        outputs = activations @ weights + biases
        if layer_index < len(parameters) - 1:
            activations = jax.nn.tanh(outputs)
        else:
            activations = outputs
    return activations.squeeze(-1)


def evaluate_binary_classification_loss(
    parameters: List[Tuple[Array, Array]],
    dataset: LargeNetworkDataset,
    l2_regularization: float = 1e-4,
) -> Array:
    logits = mlp_forward(parameters, dataset.features)
    data_loss = jnp.mean(
        jnp.maximum(logits, 0.0)
        - logits * dataset.labels
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )

    regularization_loss = 0.5 * l2_regularization * sum(
        jnp.sum(weights**2) + jnp.sum(biases**2)
        for weights, biases in parameters
    )

    return data_loss + regularization_loss


def flatten_parameter_pytree(
    parameters: List[Tuple[Array, Array]],
) -> Tuple[Array, FlatParameterMetadata]:
    leaves = []
    shapes = []
    sizes = []

    for weights, biases in parameters:
        for array in (weights, biases):
            leaves.append(array.reshape(-1))
            shapes.append(array.shape)
            sizes.append(array.size)

    flat_vector = jnp.concatenate(leaves, axis=0)
    metadata = FlatParameterMetadata(
        shapes=tuple(shapes),
        sizes=tuple(sizes),
    )
    return flat_vector, metadata


def unflatten_parameter_vector(
    flat_vector: Array,
    architecture: NetworkArchitecture,
    metadata: FlatParameterMetadata,
) -> List[Tuple[Array, Array]]:
    del architecture  # metadata already carries what we need

    arrays = []
    start_index = 0
    for shape, size in zip(metadata.shapes, metadata.sizes):
        stop_index = start_index + size
        arrays.append(flat_vector[start_index:stop_index].reshape(shape))
        start_index = stop_index

    parameter_list = []
    for i in range(0, len(arrays), 2):
        parameter_list.append((arrays[i], arrays[i + 1]))

    return parameter_list


def evaluate_flat_binary_classification_loss(
    flat_parameter: Array,
    dataset: LargeNetworkDataset,
    architecture: NetworkArchitecture,
    metadata: FlatParameterMetadata,
    l2_regularization: float = 1e-4,
) -> Array:
    parameters = unflatten_parameter_vector(flat_parameter, architecture, metadata)
    return evaluate_binary_classification_loss(
        parameters,
        dataset,
        l2_regularization=l2_regularization,
    )


def evaluate_flat_binary_classification_gradient(
    flat_parameter: Array,
    dataset: LargeNetworkDataset,
    architecture: NetworkArchitecture,
    metadata: FlatParameterMetadata,
    l2_regularization: float = 1e-4,
) -> Array:
    loss_function = lambda p: evaluate_flat_binary_classification_loss(
        p,
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )
    return jax.grad(loss_function)(flat_parameter)


def create_large_network_gradient_flow(
    dataset: LargeNetworkDataset,
    architecture: NetworkArchitecture,
    metadata: FlatParameterMetadata,
    l2_regularization: float = 1e-4,
) -> VectorField:
    return lambda state: -evaluate_flat_binary_classification_gradient(
        state,
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )


def create_default_large_network_setup(
    random_seed: int = 0,
):
    dataset = create_synthetic_binary_classification_dataset(
        sample_count_per_class=200,
        random_seed=random_seed,
    )
    architecture = NetworkArchitecture(layer_sizes=(2, 4, 4, 1))
    parameter_list = initialize_mlp_parameters(
        architecture,
        random_seed=random_seed,
    )
    flat_parameter, metadata = flatten_parameter_pytree(parameter_list)

    return {
        "dataset": dataset,
        "architecture": architecture,
        "initial_parameter_vector": flat_parameter,
        "metadata": metadata,
    }