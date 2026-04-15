from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granmian_synthesis.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from granmian_synthesis.core.ml_loss_neural_network import (
    NetworkArchitecture,
    create_large_network_gradient_flow,
    create_synthetic_binary_classification_dataset,
    evaluate_flat_binary_classification_loss,
    flatten_parameter_pytree,
    initialize_mlp_parameters,
)


OUTPUT_DIRECTORY = Path("granmian_synthesis/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def create_small_network_setup(
    dataset_seed: int = 0,
    initialization_seed: int = 0,
    sample_count_per_class: int = 120,
    weight_scale: float = 0.20,
):
    """
    Small network:
        input(2) -> hidden(3) -> output(1)

    Parameter count:
        2*3 + 3 + 3*1 + 1 = 13
    """
    dataset = create_synthetic_binary_classification_dataset(
        sample_count_per_class=sample_count_per_class,
        random_seed=dataset_seed,
    )
    architecture = NetworkArchitecture(layer_sizes=(2, 3, 1))

    parameter_list = initialize_mlp_parameters(
        architecture,
        random_seed=initialization_seed,
        weight_scale=weight_scale,
    )
    flat_parameter, metadata = flatten_parameter_pytree(parameter_list)

    return {
        "dataset": dataset,
        "architecture": architecture,
        "metadata": metadata,
        "initial_parameter_vector": flat_parameter,
    }


def evaluate_loss(
    state,
    dataset,
    architecture,
    metadata,
    l2_regularization: float = 1e-4,
) -> float:
    return float(
        evaluate_flat_binary_classification_loss(
            jnp.array(state),
            dataset,
            architecture,
            metadata,
            l2_regularization=l2_regularization,
        )
    )


def compute_distance(point_a, point_b) -> float:
    return float(jnp.linalg.norm(jnp.array(point_a) - jnp.array(point_b)))


def sample_unit_directions(key, dimension: int, direction_count: int):
    raw = jax.random.normal(key, shape=(direction_count, dimension))
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    return raw / jnp.maximum(norms, 1e-12)


def generate_ray_points(initial_state, direction, segment_length: float, segment_count: int):
    initial_state = jnp.array(initial_state, dtype=float)
    direction = jnp.array(direction, dtype=float)

    points = [
        initial_state + float(segment_index) * segment_length * direction
        for segment_index in range(segment_count + 1)
    ]
    return points


def create_system(
    dataset,
    architecture,
    metadata,
    l2_regularization: float = 1e-4,
):
    vector_field = create_large_network_gradient_flow(
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )

    state_dimension = int(sum(metadata.sizes))

    system = build_control_affine_gradient_flow_system(
        vector_field,
        state_dimension=state_dimension,
    )
    ode_solver_interface = create_ode_solver_interface()

    return system, ode_solver_interface


def run_minimum_energy_segment(
    initial_state,
    target_state,
    system,
    ode_solver_interface,
    initial_time: float = 0.0,
    terminal_time: float = 0.40,
):
    """
    Compute the minimum-energy control for one small segment only.
    """
    initial_state = jnp.array(initial_state, dtype=float)
    target_state = jnp.array(target_state, dtype=float)

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
    minimum_energy_control_energy = float(comparison_results["minimum_energy_control_energy"])

    segment_distance = compute_distance(initial_state, target_state)
    terminal_error = compute_distance(minimum_energy_terminal_state, target_state)

    return {
        "terminal_state": minimum_energy_terminal_state,
        "segment_distance": float(segment_distance),
        "segment_energy": float(minimum_energy_control_energy),
        "terminal_error": float(terminal_error),
        "energy_per_distance": (
            float(minimum_energy_control_energy / segment_distance)
            if segment_distance > 0.0
            else np.nan
        ),
    }


def make_plots(results_df: pd.DataFrame):
    """
    Create:
      1) segment energy vs segment index
      2) cumulative energy vs radius
      3) heatmap of segment energy
      4) heatmap of energy per distance
    """
    if results_df.empty:
        return

    # Plot 1: segment energy for each direction
    plt.figure(figsize=(10, 6))
    for direction_index, group in results_df.groupby("direction_index"):
        group = group.sort_values("segment_index")
        plt.plot(
            group["segment_index"],
            group["segment_energy"],
            marker="o",
            label=f"dir {direction_index}",
        )
    plt.xlabel("Segment index")
    plt.ylabel("Segment energy")
    plt.title("Per-segment minimum control energy along each direction")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "small_network_directional_segment_energy.png",
        dpi=200,
    )
    plt.close()

    # Plot 2: cumulative energy vs radius
    plt.figure(figsize=(10, 6))
    for direction_index, group in results_df.groupby("direction_index"):
        group = group.sort_values("segment_index")
        plt.plot(
            group["segment_end_radius"],
            group["cumulative_energy"],
            marker="o",
            label=f"dir {direction_index}",
        )
    plt.xlabel("Radius")
    plt.ylabel("Cumulative energy")
    plt.title("Cumulative minimum control energy along each direction")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "small_network_directional_cumulative_energy.png",
        dpi=200,
    )
    plt.close()

    # Plot 3: heatmap of segment energy
    energy_matrix = (
        results_df.pivot(
            index="direction_index",
            columns="segment_index",
            values="segment_energy",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(energy_matrix.values, aspect="auto")
    plt.colorbar(label="Segment energy")
    plt.xlabel("Segment index")
    plt.ylabel("Direction index")
    plt.title("Heatmap of per-segment minimum control energy")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "small_network_directional_segment_energy_heatmap.png",
        dpi=200,
    )
    plt.close()

    # Plot 4: heatmap of energy per distance
    epd_matrix = (
        results_df.pivot(
            index="direction_index",
            columns="segment_index",
            values="energy_per_distance",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(epd_matrix.values, aspect="auto")
    plt.colorbar(label="Energy / distance")
    plt.xlabel("Segment index")
    plt.ylabel("Direction index")
    plt.title("Heatmap of energy per distance")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "small_network_directional_energy_per_distance_heatmap.png",
        dpi=200,
    )
    plt.close()


def main():
    dataset_seed = 0
    initialization_seed = 3

    l2_regularization = 1e-4
    direction_count = 3
    segment_count = 3
    segment_length = 0.35
    terminal_time = 0.40

    setup = create_small_network_setup(
        dataset_seed=dataset_seed,
        initialization_seed=initialization_seed,
        sample_count_per_class=120,
        weight_scale=0.20,
    )

    dataset = setup["dataset"]
    architecture = setup["architecture"]
    metadata = setup["metadata"]
    initial_state = setup["initial_parameter_vector"]

    initial_loss = evaluate_loss(
        initial_state,
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )

    system, ode_solver_interface = create_system(
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )

    key = jax.random.PRNGKey(123)
    directions = sample_unit_directions(
        key,
        initial_state.shape[0],
        direction_count,
    )

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    print("\n================ DIRECTIONAL SEGMENT ENERGY STUDY ================\n")
    print(f"parameter dimension     = {int(initial_state.shape[0])}")
    print(f"initial loss            = {initial_loss:.8f}")
    print(f"direction_count         = {direction_count}")
    print(f"segment_count           = {segment_count}")
    print(f"segment_length          = {segment_length:.4f}")
    print(f"terminal_time/segment   = {terminal_time:.4f}")
    print()

    for direction_index, direction in enumerate(directions):
        ray_points = generate_ray_points(
            initial_state=initial_state,
            direction=direction,
            segment_length=segment_length,
            segment_count=segment_count,
        )

        cumulative_energy = 0.0
        cumulative_distance = 0.0

        print(f"Direction {direction_index:02d}")

        for segment_index in range(segment_count):
            segment_initial_state = ray_points[segment_index]
            segment_target_state = ray_points[segment_index + 1]

            start_radius = float(segment_index * segment_length)
            end_radius = float((segment_index + 1) * segment_length)

            segment_start_loss = evaluate_loss(
                segment_initial_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
            segment_target_loss = evaluate_loss(
                segment_target_state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )

            result = run_minimum_energy_segment(
                initial_state=segment_initial_state,
                target_state=segment_target_state,
                system=system,
                ode_solver_interface=ode_solver_interface,
                initial_time=0.0,
                terminal_time=terminal_time,
            )

            cumulative_energy += result["segment_energy"]
            cumulative_distance += result["segment_distance"]

            row = {
                "direction_index": int(direction_index),
                "segment_index": int(segment_index),
                "segment_start_radius": start_radius,
                "segment_end_radius": end_radius,
                "cumulative_distance": float(cumulative_distance),
                "cumulative_energy": float(cumulative_energy),
                "segment_start_loss": float(segment_start_loss),
                "segment_target_loss": float(segment_target_loss),
                "segment_distance": float(result["segment_distance"]),
                "segment_energy": float(result["segment_energy"]),
                "energy_per_distance": float(result["energy_per_distance"]),
                "terminal_error": float(result["terminal_error"]),
            }
            all_rows.append(row)

            print(
                f"  segment {segment_index:02d} | "
                f"radius [{start_radius:.2f}, {end_radius:.2f}] | "
                f"energy = {result['segment_energy']:.8f} | "
                f"energy/distance = {result['energy_per_distance']:.8f} | "
                f"terminal error = {result['terminal_error']:.8f}"
            )

        direction_rows = [
            row for row in all_rows if row["direction_index"] == direction_index
        ]
        total_energy = sum(row["segment_energy"] for row in direction_rows)
        mean_energy = float(np.mean([row["segment_energy"] for row in direction_rows]))
        max_energy = float(np.max([row["segment_energy"] for row in direction_rows]))
        mean_epd = float(np.mean([row["energy_per_distance"] for row in direction_rows]))

        summary_rows.append(
            {
                "direction_index": int(direction_index),
                "final_radius": float(segment_count * segment_length),
                "total_energy": float(total_energy),
                "mean_segment_energy": mean_energy,
                "max_segment_energy": max_energy,
                "mean_energy_per_distance": mean_epd,
            }
        )

        print(
            f"  total energy = {total_energy:.8f} | "
            f"mean segment energy = {mean_energy:.8f} | "
            f"max segment energy = {max_energy:.8f}"
        )
        print()

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_csv = OUTPUT_DIRECTORY / "small_network_directional_segment_energy.csv"
    summary_csv = OUTPUT_DIRECTORY / "small_network_directional_segment_energy_summary.csv"
    results_txt = OUTPUT_DIRECTORY / "small_network_directional_segment_energy.txt"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(results_txt, "w") as f:
        f.write("Directional segment energy study\n\n")
        f.write(f"parameter dimension = {int(initial_state.shape[0])}\n")
        f.write(f"initial loss = {initial_loss:.8f}\n")
        f.write(f"direction_count = {direction_count}\n")
        f.write(f"segment_count = {segment_count}\n")
        f.write(f"segment_length = {segment_length:.4f}\n")
        f.write(f"terminal_time/segment = {terminal_time:.4f}\n\n")
        f.write("Per-segment results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nDirection summary:\n")
        f.write(summary_df.to_string(index=False))

    make_plots(results_df)

    print("\n================ DIRECTION SUMMARY ================\n")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(" ", results_csv)
    print(" ", summary_csv)
    print(" ", results_txt)
    print(" ", OUTPUT_DIRECTORY / "small_network_directional_segment_energy.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_directional_cumulative_energy.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_directional_segment_energy_heatmap.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_directional_energy_per_distance_heatmap.png")


if __name__ == "__main__":
    main()