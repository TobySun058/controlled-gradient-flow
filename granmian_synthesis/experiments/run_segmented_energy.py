from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

from granmian_synthesis.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_minimum_energy,
)

from granmian_synthesis.core.objective import (
    create_default_objective_parameters,
    create_uncontrolled_gradient_flow,
    evaluate_smooth_minimum_objective,
)


OUTPUT_DIRECTORY = Path("granmian_synthesis/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def compute_distance(point_a, point_b):
    return float(jnp.linalg.norm(jnp.array(point_a) - jnp.array(point_b)))


def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_terminal_objective_value(terminal_state, parameters):
    return float(
        evaluate_smooth_minimum_objective(jnp.array(terminal_state), parameters)
    )


def generate_segment_points(initial_state, target_state, segment_count):
    initial_state = jnp.array(initial_state, dtype=float)
    target_state = jnp.array(target_state, dtype=float)

    interpolation_values = jnp.linspace(0.0, 1.0, segment_count + 1)
    points = [
        initial_state + alpha * (target_state - initial_state)
        for alpha in interpolation_values
    ]
    return points


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

    # lighter ODE settings
    ode_solver_interface = create_ode_solver_interface(
        relative_tolerance=1e-5,
        absolute_tolerance=1e-5,
        maximum_steps=200000,
        initial_step_size=1e-3,
    )

    # lighter almost-Gramian problem
    synthesis_problem = create_control_synthesis_problem(
        system=system,
        initial_state=initial_state,
        target_state=target_state,
        initial_time=initial_time,
        terminal_time=terminal_time,
        ode_solver_interface=ode_solver_interface,
        integration_samples=3000,
        maximum_iterations=4,
        minimum_iterations=1,
        record_samples=200,
        interpolation_samples=1000,
    )

    minimum_energy_results = simulate_minimum_energy(
        system,
        synthesis_problem,
        initial_state,
        target_state,
        initial_time,
        terminal_time,
        ode_solver_interface,
    )

    minimum_energy_trajectory = minimum_energy_results["minimum_energy_state_trajectory"]
    minimum_energy_terminal_state = minimum_energy_trajectory[-1]
    minimum_energy_control_energy = float(
        minimum_energy_results["minimum_energy_control_energy"]
    )

    segment_distance = compute_distance(initial_state, target_state)

    segment_energy_derivative = (
        minimum_energy_control_energy / segment_distance
        if segment_distance > 0.0
        else np.nan
    )

    return {
        "initial_x1": float(initial_state[0]),
        "initial_x2": float(initial_state[1]),
        "target_x1": float(target_state[0]),
        "target_x2": float(target_state[1]),
        "terminal_x1": float(minimum_energy_terminal_state[0]),
        "terminal_x2": float(minimum_energy_terminal_state[1]),
        "segment_distance": float(segment_distance),
        "terminal_error": compute_terminal_error(
            minimum_energy_terminal_state, target_state
        ),
        "terminal_objective": compute_terminal_objective_value(
            minimum_energy_terminal_state, parameters
        ),
        "segment_energy": float(minimum_energy_control_energy),
        "segment_energy_derivative": float(segment_energy_derivative),
    }

def main():
    softmin_sharpness = 1.25
    initial_time = 0.0
    terminal_time = 0.75

    # Lower-left corner to upper-right corner
    global_initial_state = jnp.array([0.5, 0.0])
    global_target_state = jnp.array([6.5, 8.0])

    # Number of small segments
    segment_count = 50

    segment_points = generate_segment_points(
        global_initial_state,
        global_target_state,
        segment_count,
    )

    all_rows = []
    cumulative_energy = 0.0
    cumulative_distance = 0.0

    total_straight_line_distance = compute_distance(
        global_initial_state, global_target_state
    )

    print("\n================ SEGMENTED MINIMUM-ENERGY STUDY ================\n")
    print(f"softmin_sharpness           = {softmin_sharpness:.2f}")
    print(f"terminal_time per segment   = {terminal_time:.2f}")
    print(f"global initial state        = [{float(global_initial_state[0]):.4f}, {float(global_initial_state[1]):.4f}]")
    print(f"global target state         = [{float(global_target_state[0]):.4f}, {float(global_target_state[1]):.4f}]")
    print(f"total straight-line distance= {total_straight_line_distance:.8f}")
    print(f"segment_count               = {segment_count}\n")

    for segment_index in range(segment_count):
        segment_initial_state = segment_points[segment_index]
        segment_target_state = segment_points[segment_index + 1]

        result = run_minimum_energy_case(
            initial_state=segment_initial_state,
            target_state=segment_target_state,
            softmin_sharpness=softmin_sharpness,
            initial_time=initial_time,
            terminal_time=terminal_time,
        )

        cumulative_energy += result["segment_energy"]
        cumulative_distance += result["segment_distance"]

        row = {
            "segment_index": segment_index,
            "segment_start_fraction": float(segment_index / segment_count),
            "segment_end_fraction": float((segment_index + 1) / segment_count),
            "cumulative_distance": float(cumulative_distance),
            "cumulative_energy": float(cumulative_energy),
            **result,
        }
        all_rows.append(row)

        print(f"Segment {segment_index:02d}")
        print(
            f"  start state              = [{result['initial_x1']:.4f}, {result['initial_x2']:.4f}]"
        )
        print(
            f"  end state                = [{result['target_x1']:.4f}, {result['target_x2']:.4f}]"
        )
        print(f"  segment distance         = {result['segment_distance']:.8f}")
        print(f"  segment energy           = {result['segment_energy']:.8f}")
        print(
            f"  segment energy derivative= {result['segment_energy_derivative']:.8f}"
        )
        print(f"  terminal error           = {result['terminal_error']:.8f}")
        print(f"  cumulative energy        = {cumulative_energy:.8f}")
        print()

    results_df = pd.DataFrame(all_rows)

    summary_row = {
        "segment_index": "TOTAL",
        "segment_start_fraction": 0.0,
        "segment_end_fraction": 1.0,
        "cumulative_distance": float(cumulative_distance),
        "cumulative_energy": float(cumulative_energy),
        "initial_x1": float(global_initial_state[0]),
        "initial_x2": float(global_initial_state[1]),
        "target_x1": float(global_target_state[0]),
        "target_x2": float(global_target_state[1]),
        "terminal_x1": np.nan,
        "terminal_x2": np.nan,
        "segment_distance": float(total_straight_line_distance),
        "terminal_error": np.nan,
        "terminal_objective": np.nan,
        "segment_energy": float(cumulative_energy),
        "segment_energy_derivative": (
            float(cumulative_energy / cumulative_distance)
            if cumulative_distance > 0.0
            else np.nan
        ),
    }

    summary_df = pd.DataFrame([summary_row])

    csv_path = OUTPUT_DIRECTORY / "segmented_minimum_energy_study.csv"
    txt_path = OUTPUT_DIRECTORY / "segmented_minimum_energy_study.txt"

    results_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write("Segmented minimum-energy study\n")
        f.write(f"softmin_sharpness = {softmin_sharpness:.2f}\n")
        f.write(f"terminal_time per segment = {terminal_time:.2f}\n")
        f.write(
            f"global initial state = [{float(global_initial_state[0]):.4f}, {float(global_initial_state[1]):.4f}]\n"
        )
        f.write(
            f"global target state  = [{float(global_target_state[0]):.4f}, {float(global_target_state[1]):.4f}]\n"
        )
        f.write(f"segment_count = {segment_count}\n\n")
        f.write("Per-segment results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nOverall summary:\n")
        f.write(summary_df.to_string(index=False))

    print("================================================================")
    print("\nPer-segment summary:\n")
    print(
        results_df[
            [
                "segment_index",
                "initial_x1",
                "initial_x2",
                "target_x1",
                "target_x2",
                "segment_distance",
                "segment_energy",
                "segment_energy_derivative",
                "terminal_error",
                "cumulative_energy",
            ]
        ].to_string(index=False)
    )

    print("\nOverall summary:\n")
    print(summary_df.to_string(index=False))

    print(f"\nSaved CSV summary to: {csv_path}")
    print(f"Saved text summary to: {txt_path}")


if __name__ == "__main__":
    main()