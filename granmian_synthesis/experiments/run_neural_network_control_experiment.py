from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from pathlib import Path

from granmian_synthesis.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from granmian_synthesis.core.ml_loss_neural_network import (
    create_default_large_network_setup,
    create_large_network_gradient_flow,
    evaluate_flat_binary_classification_loss,
)


OUTPUT_DIRECTORY = Path("granmian_synthesis/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_initial_target_distance(initial_state, target_state):
    return float(jnp.linalg.norm(jnp.array(initial_state) - jnp.array(target_state)))


def train_to_local_minimum(
    initial_state,
    dataset,
    architecture,
    metadata,
    l2_regularization: float = 1e-4,
    step_size: float = 1e-2,
    iteration_count: int = 2000,
):
    loss_function = lambda parameter: evaluate_flat_binary_classification_loss(
        parameter,
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )
    gradient_function = jax.grad(loss_function)

    parameter = jnp.array(initial_state, dtype=float)

    for _ in range(iteration_count):
        parameter = parameter - step_size * gradient_function(parameter)

    return parameter


def run_case(
    case_label,
    initial_state,
    target_state,
    dataset,
    architecture,
    metadata,
    l2_regularization: float = 1e-4,
    initial_time: float = 0.0,
    terminal_time: float = 0.75,
):
    vector_field = create_large_network_gradient_flow(
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )

    system = build_control_affine_gradient_flow_system(
        vector_field,
        state_dimension=initial_state.shape[0],
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

    uncontrolled_terminal_state = comparison_results["uncontrolled_state_trajectory"][-1]
    minimum_energy_terminal_state = comparison_results["minimum_energy_state_trajectory"][-1]
    approximate_minimum_energy_terminal_state = comparison_results[
        "approximate_minimum_energy_state_trajectory"
    ][-1]
    feedback_linearization_terminal_state = comparison_results[
        "feedback_linearization_state_trajectory"
    ][-1]

    initial_target_distance = compute_initial_target_distance(initial_state, target_state)
    minimum_energy_control_energy = float(comparison_results["minimum_energy_control_energy"])

    energy_per_distance = (
        minimum_energy_control_energy / initial_target_distance
        if initial_target_distance > 0.0
        else np.nan
    )

    return {
        "case_label": case_label,
        "dimension": int(initial_state.shape[0]),
        "initial_target_distance": float(initial_target_distance),

        "uncontrolled_terminal_error": compute_terminal_error(
            uncontrolled_terminal_state, target_state
        ),
        "minimum_energy_terminal_error": compute_terminal_error(
            minimum_energy_terminal_state, target_state
        ),
        "approximate_minimum_energy_terminal_error": compute_terminal_error(
            approximate_minimum_energy_terminal_state, target_state
        ),
        "feedback_linearization_terminal_error": compute_terminal_error(
            feedback_linearization_terminal_state, target_state
        ),

        "initial_loss": float(
            evaluate_flat_binary_classification_loss(
                initial_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),
        "target_loss": float(
            evaluate_flat_binary_classification_loss(
                target_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),
        "uncontrolled_terminal_loss": float(
            evaluate_flat_binary_classification_loss(
                uncontrolled_terminal_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),
        "minimum_energy_terminal_loss": float(
            evaluate_flat_binary_classification_loss(
                minimum_energy_terminal_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),
        "approximate_minimum_energy_terminal_loss": float(
            evaluate_flat_binary_classification_loss(
                approximate_minimum_energy_terminal_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),
        "feedback_linearization_terminal_loss": float(
            evaluate_flat_binary_classification_loss(
                feedback_linearization_terminal_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        ),

        "minimum_energy_control_energy": minimum_energy_control_energy,
        "approximate_minimum_energy_control_energy": float(
            comparison_results["approximate_minimum_energy_control_energy"]
        ),
        "feedback_linearization_control_energy": float(
            comparison_results["feedback_linearization_control_energy"]
        ),
        "energy_per_distance": float(energy_per_distance),
    }


def print_case_summary(case_name, result):
    print(f"\n==================== {case_name} ====================\n")
    print(f"dimension                              = {result['dimension']}")
    print(f"initial-target distance                = {result['initial_target_distance']:.8f}")
    print(f"initial loss                           = {result['initial_loss']:.8f}")
    print(f"target loss                            = {result['target_loss']:.8f}")
    print()
    print(f"uncontrolled terminal error            = {result['uncontrolled_terminal_error']:.8f}")
    print(f"minimum-energy terminal error          = {result['minimum_energy_terminal_error']:.8f}")
    print(f"approx minimum-energy terminal error   = {result['approximate_minimum_energy_terminal_error']:.8f}")
    print(f"feedback linearization terminal error  = {result['feedback_linearization_terminal_error']:.8f}")
    print()
    print(f"uncontrolled terminal loss             = {result['uncontrolled_terminal_loss']:.8f}")
    print(f"minimum-energy terminal loss           = {result['minimum_energy_terminal_loss']:.8f}")
    print(f"approx minimum-energy terminal loss    = {result['approximate_minimum_energy_terminal_loss']:.8f}")
    print(f"feedback linearization terminal loss   = {result['feedback_linearization_terminal_loss']:.8f}")
    print()
    print(f"minimum-energy control energy          = {result['minimum_energy_control_energy']:.8f}")
    print(f"approx minimum-energy control energy   = {result['approximate_minimum_energy_control_energy']:.8f}")
    print(f"feedback linearization control energy  = {result['feedback_linearization_control_energy']:.8f}")
    print(f"energy / distance                      = {result['energy_per_distance']:.8f}")


def main():
    setup = create_default_large_network_setup(random_seed=0)

    dataset = setup["dataset"]
    architecture = setup["architecture"]
    metadata = setup["metadata"]
    reference_parameter = setup["initial_parameter_vector"]

    parameter_dimension = reference_parameter.shape[0]

    initial_state = jnp.ones((parameter_dimension,))
    designated_target = 4.0 * jnp.ones((parameter_dimension,))

    minimum_target = train_to_local_minimum(
        initial_state=initial_state,
        dataset=dataset,
        architecture=architecture,
        metadata=metadata,
        l2_regularization=1e-4,
        step_size=1e-2,
        iteration_count=2000,
    )

    results = []

    designated_result = run_case(
        case_label="designated_target",
        initial_state=initial_state,
        target_state=designated_target,
        dataset=dataset,
        architecture=architecture,
        metadata=metadata,
        l2_regularization=1e-4,
        initial_time=0.0,
        terminal_time=0.75,
    )
    results.append(designated_result)
    print_case_summary("DESIGNATED TARGET CASE", designated_result)

    minimum_result = run_case(
        case_label="local_minimum_target",
        initial_state=initial_state,
        target_state=minimum_target,
        dataset=dataset,
        architecture=architecture,
        metadata=metadata,
        l2_regularization=1e-4,
        initial_time=0.0,
        terminal_time=0.75,
    )
    results.append(minimum_result)
    print_case_summary("LOCAL MINIMUM TARGET CASE", minimum_result)

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIRECTORY / "large_network_control_summary.csv"
    txt_path = OUTPUT_DIRECTORY / "large_network_control_summary.txt"

    results_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as file:
        file.write(results_df.to_string(index=False))

    print("\n===============================================================\n")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV summary to: {csv_path}")
    print(f"Saved text summary to: {txt_path}")


if __name__ == "__main__":
    main()