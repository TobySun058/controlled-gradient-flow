from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

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


OUTPUT_DIRECTORY = Path("granmian_synthesis/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_initial_target_distance(initial_state, target_state):
    return float(jnp.linalg.norm(jnp.array(initial_state) - jnp.array(target_state)))


def compute_terminal_objective_value(terminal_state, parameters):
    return float(
        evaluate_smooth_minimum_objective(jnp.array(terminal_state), parameters)
    )


def run_minimum_energy_case(
    initial_state,
    target_state,
    softmin_sharpness=1.25,
    initial_time=0.0,
    terminal_time=0.75,
):
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

    minimum_energy_trajectory = comparison_results["minimum_energy_state_trajectory"]
    minimum_energy_terminal_state = minimum_energy_trajectory[-1]
    minimum_energy_control_energy = comparison_results["minimum_energy_control_energy"]

    initial_target_distance = compute_initial_target_distance(
        initial_state, target_state
    )

    energy_per_distance = (
        float(minimum_energy_control_energy) / initial_target_distance
        if initial_target_distance > 0.0
        else np.nan
    )

    return {
        "initial_x1": float(initial_state[0]),
        "initial_x2": float(initial_state[1]),
        "target_x1": float(target_state[0]),
        "target_x2": float(target_state[1]),
        "terminal_x1": float(minimum_energy_terminal_state[0]),
        "terminal_x2": float(minimum_energy_terminal_state[1]),
        "initial_target_distance": float(initial_target_distance),
        "terminal_error": compute_terminal_error(
            minimum_energy_terminal_state, target_state
        ),
        "terminal_objective": compute_terminal_objective_value(
            minimum_energy_terminal_state, parameters
        ),
        "minimum_energy_control_energy": float(minimum_energy_control_energy),
        "energy_per_distance": float(energy_per_distance),
    }


def main():
    softmin_sharpness = 1.25
    initial_time = 0.0
    terminal_time = 0.75

    test_cases = [
        {
            "label": "original_case",
            "initial_state": jnp.array([2.0, 6.0]),
            "target_state": jnp.array([4.5, 1.5]),
        },
        {
            "label": "upper_left_basin_to_lower_basin",
            "initial_state": jnp.array([1.0, 6.0]),
            "target_state": jnp.array([0.5, 0.0]),
        },
        {
            "label": "upper_left_basin_to_right_basin",
            "initial_state": jnp.array([1.0, 6.0]),
            "target_state": jnp.array([4.0, 2.0]),
        },
        {
            "label": "lower_basin_to_upper_left_basin",
            "initial_state": jnp.array([0.5, 0.0]),
            "target_state": jnp.array([1.0, 6.0]),
        },
        {
            "label": "lower_basin_to_right_steep_region",
            "initial_state": jnp.array([0.5, 0.0]),
            "target_state": jnp.array([6.5, 8.0]),
        },
        {
            "label": "saddle_to_lower_basin",
            "initial_state": jnp.array([3.5, 4.0]),
            "target_state": jnp.array([4.0, 2.0]),
        },
        {
            "label": "saddle_to_upper_left_basin",
            "initial_state": jnp.array([3.5, 4.0]),
            "target_state": jnp.array([1.0, 6.0]),
        },
        {
            "label": "right_steep_region_to_upper_left_basin",
            "initial_state": jnp.array([6.5, 8.0]),
            "target_state": jnp.array([1.0, 6.0]),
        },
        {
            "label": "right_steep_region_to_lower_basin",
            "initial_state": jnp.array([6.5, 8.0]),
            "target_state": jnp.array([0.5, 0.0]),
        },
    ]

    all_rows = []

    print("\n==================== MINIMUM-ENERGY LANDSCAPE STUDY ====================\n")
    print(f"softmin_sharpness = {softmin_sharpness:.2f}")
    print(f"terminal_time     = {terminal_time:.2f}\n")

    for case in test_cases:
        result = run_minimum_energy_case(
            initial_state=case["initial_state"],
            target_state=case["target_state"],
            softmin_sharpness=softmin_sharpness,
            initial_time=initial_time,
            terminal_time=terminal_time,
        )

        row = {"label": case["label"], **result}
        all_rows.append(row)

        print(f"Case: {case['label']}")
        print(
            f"  initial state     = [{result['initial_x1']:.4f}, {result['initial_x2']:.4f}]"
        )
        print(
            f"  target state      = [{result['target_x1']:.4f}, {result['target_x2']:.4f}]"
        )
        print(
            f"  terminal state    = [{result['terminal_x1']:.4f}, {result['terminal_x2']:.4f}]"
        )
        print(
            f"  init-target dist  = {result['initial_target_distance']:.8f}"
        )
        print(f"  terminal error    = {result['terminal_error']:.8f}")
        print(f"  objective value   = {result['terminal_objective']:.8f}")
        print(
            f"  control energy    = {result['minimum_energy_control_energy']:.8f}"
        )
        print(
            f"  energy / distance = {result['energy_per_distance']:.8f}"
        )
        print()

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(
        by="minimum_energy_control_energy", ascending=True
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIRECTORY / "minimum_energy_landscape_study.csv"
    txt_path = OUTPUT_DIRECTORY / "minimum_energy_landscape_study.txt"

    results_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write("Minimum-energy landscape study\n")
        f.write(f"softmin_sharpness = {softmin_sharpness:.2f}\n")
        f.write(f"terminal_time     = {terminal_time:.2f}\n\n")
        f.write(results_df.to_string(index=False))

    print("=======================================================================")
    print("\nSorted results by minimum energy:\n")
    print(
        results_df[
            [
                "label",
                "initial_x1",
                "initial_x2",
                "target_x1",
                "target_x2",
                "initial_target_distance",
                "minimum_energy_control_energy",
                "energy_per_distance",
                "terminal_error",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV summary to: {csv_path}")
    print(f"Saved text summary to: {txt_path}")


if __name__ == "__main__":
    main()