from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controlled_gradient_flow.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_control_strategy_comparison,
)
from controlled_gradient_flow.core.ml_loss_neural_network import (
    NetworkArchitecture,
    create_large_network_gradient_flow,
    create_synthetic_binary_classification_dataset,
    evaluate_flat_binary_classification_gradient,
    evaluate_flat_binary_classification_loss,
    flatten_parameter_pytree,
    initialize_mlp_parameters,
)

OUTPUT_DIRECTORY = Path("results/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def create_small_network_setup(
    dataset_seed: int = 0,
    initialization_seed: int = 0,
    sample_count_per_class: int = 120,
    weight_scale: float = 0.20,
):
    """
    Smaller network than your default large setup:
        input(2) -> hidden(3) -> output(1)
    Total parameter dimension = 2*3 + 3 + 3*1 + 1 = 13
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


def compute_distance(x, y) -> float:
    return float(jnp.linalg.norm(jnp.array(x) - jnp.array(y)))


def run_gradient_descent(
    initial_state,
    dataset,
    architecture,
    metadata,
    l2_regularization: float = 1e-4,
    step_size: float = 1e-2,
    iteration_count: int = 1200,
):
    state = jnp.array(initial_state, dtype=float)
    loss_history = [
        evaluate_loss(
            state,
            dataset,
            architecture,
            metadata,
            l2_regularization=l2_regularization,
        )
    ]

    for _ in range(iteration_count):
        gradient = evaluate_flat_binary_classification_gradient(
            state,
            dataset,
            architecture,
            metadata,
            l2_regularization=l2_regularization,
        )
        state = state - step_size * gradient
        loss_history.append(
            evaluate_loss(
                state,
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )
        )

    return {
        "terminal_state": state,
        "terminal_loss": float(loss_history[-1]),
        "loss_history": loss_history,
    }


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


def run_minimum_energy_move(
    initial_state,
    target_state,
    system,
    ode_solver_interface,
    initial_time: float = 0.0,
    terminal_time: float = 0.40,
):
    """
    Uses only the minimum-energy trajectory from the comparison output.
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

    terminal_state = comparison_results["minimum_energy_state_trajectory"][-1]
    control_energy = float(comparison_results["minimum_energy_control_energy"])
    travel_distance = compute_distance(initial_state, target_state)
    terminal_error = compute_distance(terminal_state, target_state)

    return {
        "terminal_state": terminal_state,
        "control_energy": control_energy,
        "travel_distance": travel_distance,
        "energy_per_distance": (
            control_energy / travel_distance if travel_distance > 0.0 else np.nan
        ),
        "terminal_error": terminal_error,
    }


def sample_unit_directions(key, dimension: int, direction_count: int):
    raw = jax.random.normal(key, shape=(direction_count, dimension))
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    return raw / jnp.maximum(norms, 1e-12)


def directional_energy_scan(
    initial_state,
    baseline_gd_loss: float,
    dataset,
    architecture,
    metadata,
    system,
    ode_solver_interface,
    l2_regularization: float = 1e-4,
    gd_step_size: float = 1e-2,
    gd_relax_steps: int = 700,
    direction_count: int = 16,
    radius_list=(0.20, 0.40, 0.70, 1.00, 1.40),
    terminal_time: float = 0.40,
    score_penalty: float = 0.15,
    random_seed: int = 0,
):
    """
    From the same initial point x0:
      - sample directions d_k
      - probe targets x0 + r d_k
      - compute minimum energy
      - relax with GD
      - summarize each direction
    """
    initial_state = jnp.array(initial_state, dtype=float)
    key = jax.random.PRNGKey(random_seed)
    directions = sample_unit_directions(
        key,
        initial_state.shape[0],
        direction_count,
    )

    probe_records: List[Dict] = []

    for direction_index, direction in enumerate(directions):
        for radius in radius_list:
            target_state = initial_state + float(radius) * direction

            try:
                move = run_minimum_energy_move(
                    initial_state=initial_state,
                    target_state=target_state,
                    system=system,
                    ode_solver_interface=ode_solver_interface,
                    terminal_time=terminal_time,
                )
            except Exception:
                continue

            controlled_loss = evaluate_loss(
                move["terminal_state"],
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
            )

            relaxed = run_gradient_descent(
                move["terminal_state"],
                dataset,
                architecture,
                metadata,
                l2_regularization=l2_regularization,
                step_size=gd_step_size,
                iteration_count=gd_relax_steps,
            )

            relaxed_loss = float(relaxed["terminal_loss"])
            improvement_over_baseline = float(baseline_gd_loss - relaxed_loss)

            probe_records.append(
                {
                    "direction_index": int(direction_index),
                    "radius": float(radius),
                    "control_energy": float(move["control_energy"]),
                    "travel_distance": float(move["travel_distance"]),
                    "energy_per_distance": float(move["energy_per_distance"]),
                    "terminal_error": float(move["terminal_error"]),
                    "controlled_loss": float(controlled_loss),
                    "relaxed_loss": float(relaxed_loss),
                    "improvement_over_baseline": float(improvement_over_baseline),
                    "move_terminal_state": move["terminal_state"],
                    "relaxed_terminal_state": relaxed["terminal_state"],
                }
            )

    if not probe_records:
        return [], [], None

    direction_summary_rows: List[Dict] = []

    for direction_index in range(direction_count):
        rows = [r for r in probe_records if r["direction_index"] == direction_index]
        if not rows:
            continue

        rows_sorted = sorted(rows, key=lambda x: x["radius"])
        best_row = min(rows_sorted, key=lambda x: x["relaxed_loss"])

        radii = np.array([r["radius"] for r in rows_sorted], dtype=float)
        epd = np.array([r["energy_per_distance"] for r in rows_sorted], dtype=float)

        if len(radii) >= 2 and np.all(np.isfinite(epd)):
            energy_slope = float(np.polyfit(radii, epd, deg=1)[0])
        else:
            energy_slope = np.nan

        best_energy = float(best_row["control_energy"])
        best_relaxed_loss = float(best_row["relaxed_loss"])
        best_improvement = float(best_row["improvement_over_baseline"])

        score = (
            best_improvement / max(best_energy, 1e-10)
            - score_penalty * max(energy_slope, 0.0)
        )

        direction_summary_rows.append(
            {
                "direction_index": int(direction_index),
                "best_radius": float(best_row["radius"]),
                "best_relaxed_loss": best_relaxed_loss,
                "best_control_energy": best_energy,
                "best_energy_per_distance": float(best_row["energy_per_distance"]),
                "best_improvement_over_baseline": best_improvement,
                "energy_slope": float(energy_slope),
                "score": float(score),
                "beats_baseline_gd": bool(best_relaxed_loss < baseline_gd_loss),
                "best_probe_row": best_row,
            }
        )

    candidates_beating_gd = [
        row for row in direction_summary_rows if row["beats_baseline_gd"]
    ]

    if candidates_beating_gd:
        chosen_direction = max(candidates_beating_gd, key=lambda x: x["score"])
    else:
        chosen_direction = None

    return probe_records, direction_summary_rows, chosen_direction


def make_plots(
    probe_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    baseline_gd_loss: float,
    chosen_direction_index: Optional[int],
):
    if not direction_df.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(
            direction_df["direction_index"],
            direction_df["score"],
            marker="o",
        )
        plt.xlabel("Direction index")
        plt.ylabel("Direction score")
        plt.title("Directional energy score")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIRECTORY / "small_network_direction_scores.png",
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            direction_df["direction_index"],
            direction_df["best_relaxed_loss"],
            marker="o",
        )
        plt.axhline(
            baseline_gd_loss,
            linestyle="--",
            label="Baseline GD loss",
        )
        plt.xlabel("Direction index")
        plt.ylabel("Best relaxed loss")
        plt.title("Best loss found along each direction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIRECTORY / "small_network_direction_best_losses.png",
            dpi=200,
        )
        plt.close()

    if not probe_df.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            probe_df["control_energy"],
            probe_df["relaxed_loss"],
            alpha=0.7,
        )
        plt.xlabel("Minimum control energy")
        plt.ylabel("Relaxed final loss")
        plt.title("All probes: energy vs relaxed loss")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIRECTORY / "small_network_probe_energy_vs_relaxed_loss.png",
            dpi=200,
        )
        plt.close()

    if chosen_direction_index is not None and not probe_df.empty:
        chosen_df = probe_df[probe_df["direction_index"] == chosen_direction_index]
        chosen_df = chosen_df.sort_values("radius")

        if not chosen_df.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(
                chosen_df["radius"],
                chosen_df["energy_per_distance"],
                marker="o",
                label="Energy / distance",
            )
            plt.xlabel("Radius")
            plt.ylabel("Energy / distance")
            plt.title("Chosen direction: energy pattern along the ray")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIRECTORY / "small_network_chosen_direction_energy_curve.png",
                dpi=200,
            )
            plt.close()


def main():
    dataset_seed = 0
    initialization_seed = 3

    l2_regularization = 1e-4
    gd_step_size = 1e-2
    baseline_gd_steps = 1400
    gd_relax_steps = 700

    direction_count = 16
    radius_list = (0.20, 0.40, 0.70, 1.00, 1.40)
    terminal_time = 0.40
    score_penalty = 0.15

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

    baseline_gd = run_gradient_descent(
        initial_state,
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
        step_size=gd_step_size,
        iteration_count=baseline_gd_steps,
    )
    baseline_gd_loss = float(baseline_gd["terminal_loss"])

    system, ode_solver_interface = create_system(
        dataset,
        architecture,
        metadata,
        l2_regularization=l2_regularization,
    )

    probe_records, direction_summary_rows, chosen_direction = directional_energy_scan(
        initial_state=initial_state,
        baseline_gd_loss=baseline_gd_loss,
        dataset=dataset,
        architecture=architecture,
        metadata=metadata,
        system=system,
        ode_solver_interface=ode_solver_interface,
        l2_regularization=l2_regularization,
        gd_step_size=gd_step_size,
        gd_relax_steps=gd_relax_steps,
        direction_count=direction_count,
        radius_list=radius_list,
        terminal_time=terminal_time,
        score_penalty=score_penalty,
        random_seed=123,
    )

    if chosen_direction is not None:
        chosen_probe = chosen_direction["best_probe_row"]
        final_state = jnp.array(chosen_probe["relaxed_terminal_state"])
        guided_final_loss = float(chosen_probe["relaxed_loss"])
        chosen_direction_index = int(chosen_direction["direction_index"])
        chosen_radius = float(chosen_direction["best_radius"])
        chosen_energy = float(chosen_direction["best_control_energy"])
        guided_beats_gd = bool(guided_final_loss < baseline_gd_loss)
    else:
        final_state = baseline_gd["terminal_state"]
        guided_final_loss = baseline_gd_loss
        chosen_direction_index = None
        chosen_radius = np.nan
        chosen_energy = 0.0
        guided_beats_gd = False

    probe_rows_for_csv = []
    for row in probe_records:
        clean_row = dict(row)
        clean_row.pop("move_terminal_state", None)
        clean_row.pop("relaxed_terminal_state", None)
        probe_rows_for_csv.append(clean_row)

    direction_rows_for_csv = []
    for row in direction_summary_rows:
        clean_row = dict(row)
        clean_row.pop("best_probe_row", None)
        direction_rows_for_csv.append(clean_row)

    summary_df = pd.DataFrame(
        [
            {
                "dimension": int(initial_state.shape[0]),
                "initial_loss": float(initial_loss),
                "baseline_gd_loss": float(baseline_gd_loss),
                "guided_final_loss": float(guided_final_loss),
                "guided_minus_gd": float(guided_final_loss - baseline_gd_loss),
                "guided_beats_gd": bool(guided_beats_gd),
                "chosen_direction_index": chosen_direction_index,
                "chosen_radius": chosen_radius,
                "chosen_control_energy": float(chosen_energy),
            }
        ]
    )

    probe_df = pd.DataFrame(probe_rows_for_csv)
    direction_df = pd.DataFrame(direction_rows_for_csv)

    summary_csv = OUTPUT_DIRECTORY / "small_network_directional_scan_summary.csv"
    probe_csv = OUTPUT_DIRECTORY / "small_network_directional_scan_probes.csv"
    direction_csv = OUTPUT_DIRECTORY / "small_network_directional_scan_directions.csv"

    summary_df.to_csv(summary_csv, index=False)
    probe_df.to_csv(probe_csv, index=False)
    direction_df.to_csv(direction_csv, index=False)

    make_plots(
        probe_df=probe_df,
        direction_df=direction_df,
        baseline_gd_loss=baseline_gd_loss,
        chosen_direction_index=chosen_direction_index,
    )

    print("\n================ DIRECTIONAL ENERGY SCAN ================\n")
    print(f"parameter dimension           = {int(initial_state.shape[0])}")
    print(f"initial loss                  = {initial_loss:.8f}")
    print(f"baseline GD final loss        = {baseline_gd_loss:.8f}")
    print(f"guided final loss             = {guided_final_loss:.8f}")
    print(f"guided minus GD               = {guided_final_loss - baseline_gd_loss:.8f}")
    print(f"guided beats baseline GD      = {guided_beats_gd}")
    print(f"chosen direction index        = {chosen_direction_index}")
    print(f"chosen radius                 = {chosen_radius}")
    print(f"chosen control energy         = {chosen_energy:.8f}")

    if not direction_df.empty:
        print("\nTop directions by score:\n")
        print(
            direction_df.sort_values("score", ascending=False)
            .head(8)
            .to_string(index=False)
        )

    print("\nSaved files:")
    print(" ", summary_csv)
    print(" ", probe_csv)
    print(" ", direction_csv)
    print(" ", OUTPUT_DIRECTORY / "small_network_direction_scores.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_direction_best_losses.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_probe_energy_vs_relaxed_loss.png")
    print(" ", OUTPUT_DIRECTORY / "small_network_chosen_direction_energy_curve.png")


if __name__ == "__main__":
    main()