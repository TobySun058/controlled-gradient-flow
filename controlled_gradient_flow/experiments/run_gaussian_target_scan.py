from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

from controlled_gradient_flow.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_minimum_energy,
)
from controlled_gradient_flow.core.objective import (
    create_default_objective_parameters,
    create_uncontrolled_gradient_flow,
)


OUTPUT_DIRECTORY = Path("results/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIRECTORY / "minimum_energy_gaussian_target_scan.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "minimum_energy_gaussian_target_scan_summary.txt"


def sample_targets_around_known_minimizer(known_minimizer, D, sample_count, rng):
    known_minimizer = np.asarray(known_minimizer, dtype=float)
    noise = rng.normal(loc=0.0, scale=D, size=(sample_count, 2))
    return known_minimizer[None, :] + noise


def save_results(all_rows):
    results_df = pd.DataFrame(all_rows)

    # Save full raw results every time
    results_df.to_csv(CSV_PATH, index=False)

    # Save updated summary every time
    if len(results_df) > 0:
        summary_df = (
            results_df.groupby("D")
            .agg(
                minimum_energy_mean=("minimum_energy", "mean"),
                minimum_energy_std=("minimum_energy", "std"),
                minimum_energy_min=("minimum_energy", "min"),
                minimum_energy_max=("minimum_energy", "max"),
                terminal_error_mean=("terminal_error", "mean"),
                terminal_error_max=("terminal_error", "max"),
            )
            .reset_index()
        )
    else:
        summary_df = pd.DataFrame()

    with open(SUMMARY_PATH, "w") as f:
        f.write("minimum-energy Gaussian target scan\n\n")
        f.write("Summary by D:\n")
        if len(summary_df) > 0:
            f.write(summary_df.to_string(index=False))
        else:
            f.write("No successful samples yet.")
        f.write("\n\nFull sample results:\n")
        if len(results_df) > 0:
            f.write(results_df.to_string(index=False))
        else:
            f.write("No successful samples yet.")

    print(f"Progress saved to: {CSV_PATH}")
    print(f"Summary saved to:  {SUMMARY_PATH}")


def main():
    softmin_sharpness = 1.25
    initial_time = 0.0
    terminal_time = 0.75

    X0 = jnp.array([2.0, 6.0])
    X1 = np.array([0.0, 0.0], dtype=float)

    D_values = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
    samples_per_D = 10

    rng = np.random.default_rng(0)
    all_rows = []

    parameters = create_default_objective_parameters(float(softmin_sharpness))
    system = build_control_affine_gradient_flow_system(
        create_uncontrolled_gradient_flow(parameters),
        state_dimension=2,
    )
    ode_solver_interface = create_ode_solver_interface()

    print("\n================ MINIMUM-ENERGY GAUSSIAN TARGET SCAN ================\n")
    print(f"softmin_sharpness = {softmin_sharpness:.2f}")
    print(f"terminal_time     = {terminal_time:.2f}")
    print(f"fixed X0          = [{float(X0[0]):.4f}, {float(X0[1]):.4f}]")
    print(f"known argmin X1   = [{float(X1[0]):.4f}, {float(X1[1]):.4f}]\n")

    # Save an empty file at the beginning so the path exists immediately
    save_results(all_rows)

    for D in D_values:
        sampled_targets = sample_targets_around_known_minimizer(X1, D, samples_per_D, rng)

        print(f"--- Standard deviation D = {D:.4f} ---")

        for sample_index, target in enumerate(sampled_targets):
            print(f"Starting sample {sample_index:02d} for D = {D:.4f}")

            target_state = jnp.array(target)

            try:
                synthesis_problem = create_control_synthesis_problem(
                    system=system,
                    initial_state=X0,
                    target_state=target_state,
                    initial_time=initial_time,
                    terminal_time=terminal_time,
                    ode_solver_interface=ode_solver_interface,
                )

                results = simulate_minimum_energy(
                    system,
                    synthesis_problem,
                    X0,
                    target_state,
                    initial_time,
                    terminal_time,
                    ode_solver_interface,
                )

                terminal_state = results["minimum_energy_state_trajectory"][-1]
                terminal_error = float(jnp.linalg.norm(terminal_state - target_state))
                minimum_energy = float(results["minimum_energy_control_energy"])

                row = {
                    "D": float(D),
                    "sample_index": int(sample_index),
                    "target_guess_x1": float(target[0]),
                    "target_guess_x2": float(target[1]),
                    "terminal_x1": float(terminal_state[0]),
                    "terminal_x2": float(terminal_state[1]),
                    "terminal_error": terminal_error,
                    "minimum_energy": minimum_energy,
                }
                all_rows.append(row)

                print(
                    f"sample {sample_index:02d} | "
                    f"X_guess = [{target[0]: .4f}, {target[1]: .4f}] | "
                    f"energy = {minimum_energy:.8f} | "
                    f"err = {terminal_error:.3e}"
                )

                # Save immediately after each successful sample
                save_results(all_rows)

            except Exception as e:
                print(f"Sample {sample_index:02d} at D={D:.4f} failed: {e}")
                # Save whatever was already collected
                save_results(all_rows)

            finally:
                # Help memory a little
                if "results" in locals():
                    del results
                if "synthesis_problem" in locals():
                    del synthesis_problem
                gc.collect()

        print()

    print("================ FINAL SUMMARY ================\n")
    if len(all_rows) > 0:
        final_df = pd.DataFrame(all_rows)
        final_summary = (
            final_df.groupby("D")
            .agg(
                minimum_energy_mean=("minimum_energy", "mean"),
                minimum_energy_std=("minimum_energy", "std"),
                minimum_energy_min=("minimum_energy", "min"),
                minimum_energy_max=("minimum_energy", "max"),
                terminal_error_mean=("terminal_error", "mean"),
                terminal_error_max=("terminal_error", "max"),
            )
            .reset_index()
        )
        print(final_summary.to_string(index=False))
    else:
        print("No successful samples were completed.")

    print(f"\nSaved CSV summary to: {CSV_PATH}")
    print(f"Saved text summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()