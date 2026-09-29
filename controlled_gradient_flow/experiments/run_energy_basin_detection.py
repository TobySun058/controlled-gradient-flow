from __future__ import annotations

import gc
import heapq
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controlled_gradient_flow.core.controlled_dynamics import (
    build_control_affine_gradient_flow_system,
    create_control_synthesis_problem,
    create_ode_solver_interface,
    simulate_minimum_energy,
)
from controlled_gradient_flow.core.objective import (
    create_default_objective_parameters,
    create_uncontrolled_gradient_flow,
    evaluate_smooth_minimum_objective,
)


OUTPUT_DIRECTORY = Path("results/data")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIRECTORY / "energy_basin_atlas_checkpoint.pkl"
PROGRESS_PATH = OUTPUT_DIRECTORY / "energy_basin_atlas_progress.json"
PARTIAL_EDGES_CSV = OUTPUT_DIRECTORY / "energy_basin_atlas_partial_edges.csv"
PARTIAL_NODES_CSV = OUTPUT_DIRECTORY / "energy_basin_atlas_partial_nodes.csv"
PARTIAL_SUMMARY_TXT = OUTPUT_DIRECTORY / "energy_basin_atlas_partial_summary.txt"
PARTIAL_COMPARISON_PNG = OUTPUT_DIRECTORY / "energy_basin_atlas_partial_comparison.png"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def compute_terminal_error(terminal_state, target_state):
    return float(jnp.linalg.norm(jnp.array(terminal_state) - jnp.array(target_state)))


def compute_initial_target_distance(initial_state, target_state):
    return float(jnp.linalg.norm(jnp.array(initial_state) - jnp.array(target_state)))


def compute_terminal_objective_value(terminal_state, parameters):
    return float(
        evaluate_smooth_minimum_objective(jnp.array(terminal_state), parameters)
    )


def create_landscape_context(softmin_sharpness: float = 1.25):
    parameters = create_default_objective_parameters(float(softmin_sharpness))
    system = build_control_affine_gradient_flow_system(
        create_uncontrolled_gradient_flow(parameters),
        state_dimension=2,
    )
    ode_solver_interface = create_ode_solver_interface()
    return parameters, system, ode_solver_interface


def point_from_grid(xs: np.ndarray, ys: np.ndarray, i: int, j: int):
    return jnp.array([xs[i], ys[j]], dtype=float)


# -----------------------------------------------------------------------------
# Minimum-energy edge solve
# -----------------------------------------------------------------------------

def run_minimum_energy_case_with_context(
    initial_state,
    target_state,
    parameters,
    system,
    ode_solver_interface,
    initial_time: float = 0.0,
    terminal_time: float = 1.0,
):
    synthesis_problem = create_control_synthesis_problem(
        system=system,
        initial_state=initial_state,
        target_state=target_state,
        initial_time=initial_time,
        terminal_time=terminal_time,
        ode_solver_interface=ode_solver_interface,
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

    initial_target_distance = compute_initial_target_distance(initial_state, target_state)

    if (
        minimum_energy_results.get("failed", False)
        or minimum_energy_results.get("minimum_energy_state_trajectory") is None
    ):
        return {
            "initial_x1": float(initial_state[0]),
            "initial_x2": float(initial_state[1]),
            "target_x1": float(target_state[0]),
            "target_x2": float(target_state[1]),
            "terminal_x1": np.nan,
            "terminal_x2": np.nan,
            "initial_target_distance": float(initial_target_distance),
            "terminal_error": np.nan,
            "terminal_objective": np.nan,
            "minimum_energy_control_energy": np.nan,
            "energy_per_distance": np.nan,
            "failed": True,
        }

    minimum_energy_trajectory = minimum_energy_results["minimum_energy_state_trajectory"]
    minimum_energy_terminal_state = jnp.array(minimum_energy_trajectory[-1], dtype=float)
    minimum_energy_control_energy = float(
        minimum_energy_results["minimum_energy_control_energy"]
    )

    energy_per_distance = (
        minimum_energy_control_energy / initial_target_distance
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
        "failed": False,
    }


# -----------------------------------------------------------------------------
# Grid utilities
# -----------------------------------------------------------------------------

def four_neighbors(i: int, j: int, nx: int, ny: int):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            yield ni, nj


def eight_neighbors(i: int, j: int, nx: int, ny: int):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                yield ni, nj


def smooth_field(field: np.ndarray, passes: int = 3) -> np.ndarray:
    smoothed = field.copy()
    ny, nx = smoothed.shape

    for _ in range(passes):
        updated = smoothed.copy()
        for j in range(ny):
            for i in range(nx):
                values = []
                center = smoothed[j, i]
                if np.isfinite(center):
                    values.append(center)
                for ni, nj in eight_neighbors(i, j, nx, ny):
                    value = smoothed[nj, ni]
                    if np.isfinite(value):
                        values.append(value)
                updated[j, i] = float(np.mean(values)) if values else np.nan
        smoothed = updated

    return smoothed


def compute_boundary_mask(region_map: np.ndarray) -> np.ndarray:
    ny, nx = region_map.shape
    boundary = np.zeros_like(region_map, dtype=float)

    for j in range(ny):
        for i in range(nx):
            current = region_map[j, i]
            if current < 0:
                continue
            for ni, nj in four_neighbors(i, j, nx, ny):
                other = region_map[nj, ni]
                if other >= 0 and other != current:
                    boundary[j, i] = 1.0
                    break
    return boundary


def assign_energy_basins(local_energy_field: np.ndarray):
    ny, nx = local_energy_field.shape
    sink_map: Dict[Tuple[int, int], Tuple[int, int] | None] = {}

    def descend(i: int, j: int):
        if (i, j) in sink_map:
            return sink_map[(i, j)]

        current_value = local_energy_field[j, i]
        if not np.isfinite(current_value):
            sink_map[(i, j)] = None
            return None

        best_i, best_j = i, j
        best_value = current_value

        for ni, nj in eight_neighbors(i, j, nx, ny):
            neighbor_value = local_energy_field[nj, ni]
            if np.isfinite(neighbor_value) and neighbor_value < best_value - 1e-12:
                best_value = neighbor_value
                best_i, best_j = ni, nj

        if best_i == i and best_j == j:
            sink_map[(i, j)] = (i, j)
            return (i, j)

        sink = descend(best_i, best_j)
        sink_map[(i, j)] = sink
        return sink

    raw_sinks = []
    for j in range(ny):
        for i in range(nx):
            raw_sinks.append(descend(i, j))

    unique_sinks = sorted(
        {sink for sink in raw_sinks if sink is not None},
        key=lambda p: (p[1], p[0]),
    )
    basin_lookup = {sink: basin_id for basin_id, sink in enumerate(unique_sinks)}

    basin_map = np.full((ny, nx), -1, dtype=int)
    for j in range(ny):
        for i in range(nx):
            sink = sink_map[(i, j)]
            if sink is not None:
                basin_map[j, i] = basin_lookup[sink]

    return basin_map, unique_sinks


def compute_radial_profile(radius_map: np.ndarray, value_map: np.ndarray, bin_count: int = 10):
    flat_radius = radius_map.ravel()
    flat_value = value_map.ravel()

    valid = np.isfinite(flat_value)
    flat_radius = flat_radius[valid]
    flat_value = flat_value[valid]

    if flat_value.size == 0:
        return pd.DataFrame(
            columns=[
                "radius_left",
                "radius_right",
                "radius_mid",
                "mean_value",
                "std_value",
                "count",
            ]
        )

    radius_max = float(np.max(flat_radius))
    bins = np.linspace(0.0, radius_max, bin_count + 1)
    rows = []

    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        mask = (flat_radius >= left) & (flat_radius < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "radius_left": float(left),
                "radius_right": float(right),
                "radius_mid": float(0.5 * (left + right)),
                "mean_value": float(np.mean(flat_value[mask])),
                "std_value": float(np.std(flat_value[mask])),
                "count": int(np.sum(mask)),
            }
        )

    return pd.DataFrame(rows)


def dijkstra_minimum_energy(
    center_index: Tuple[int, int],
    edge_metrics: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]],
    nx: int,
    ny: int,
):
    cumulative_energy = np.full((ny, nx), np.inf, dtype=float)
    predecessor_i = np.full((ny, nx), -1, dtype=int)
    predecessor_j = np.full((ny, nx), -1, dtype=int)

    center_i, center_j = center_index
    cumulative_energy[center_j, center_i] = 0.0

    heap = [(0.0, center_i, center_j)]

    while heap:
        current_cost, i, j = heapq.heappop(heap)
        if current_cost > cumulative_energy[j, i] + 1e-12:
            continue

        for ni, nj in four_neighbors(i, j, nx, ny):
            edge_key = ((i, j), (ni, nj))
            edge = edge_metrics.get(edge_key)
            if edge is None:
                continue
            step_cost = edge["minimum_energy_control_energy"]

            if edge.get("failed", False) or not np.isfinite(step_cost):
                continue

            candidate_cost = current_cost + step_cost

            if candidate_cost < cumulative_energy[nj, ni] - 1e-12:
                cumulative_energy[nj, ni] = candidate_cost
                predecessor_i[nj, ni] = i
                predecessor_j[nj, ni] = j
                heapq.heappush(heap, (candidate_cost, ni, nj))

    cumulative_energy[~np.isfinite(cumulative_energy)] = np.nan
    return cumulative_energy, predecessor_i, predecessor_j


# -----------------------------------------------------------------------------
# Checkpoint / cache helpers
# -----------------------------------------------------------------------------

def _config_signature(
    center_state,
    half_width_x,
    half_width_y,
    grid_size_x,
    grid_size_y,
    softmin_sharpness,
    initial_time,
    terminal_time,
    smoothing_passes,
):
    return {
        "center_state": [float(center_state[0]), float(center_state[1])],
        "half_width_x": float(half_width_x),
        "half_width_y": float(half_width_y),
        "grid_size_x": int(grid_size_x),
        "grid_size_y": int(grid_size_y),
        "softmin_sharpness": float(softmin_sharpness),
        "initial_time": float(initial_time),
        "terminal_time": float(terminal_time),
        "smoothing_passes": int(smoothing_passes),
    }


def _build_partial_atlas_from_edges(
    xs,
    ys,
    center_state,
    center_index,
    parameters,
    edge_metrics,
    directed_edge_rows,
    softmin_sharpness,
    terminal_time,
    smoothing_passes,
):
    nx = len(xs)
    ny = len(ys)
    center_x = float(center_state[0])
    center_y = float(center_state[1])

    objective_map = np.zeros((ny, nx), dtype=float)
    radius_map = np.zeros((ny, nx), dtype=float)
    for j in range(ny):
        for i in range(nx):
            point = point_from_grid(xs, ys, i, j)
            objective_map[j, i] = float(
                evaluate_smooth_minimum_objective(point, parameters)
            )
            radius_map[j, i] = float(
                np.linalg.norm(np.array([xs[i] - center_x, ys[j] - center_y]))
            )

    local_mean_epd = np.full((ny, nx), np.nan, dtype=float)
    local_min_epd = np.full((ny, nx), np.nan, dtype=float)
    local_mean_energy = np.full((ny, nx), np.nan, dtype=float)
    success_rate = np.zeros((ny, nx), dtype=float)

    total_directed_edges = 0
    failure_count = 0
    for j in range(ny):
        for i in range(nx):
            neighbors = list(four_neighbors(i, j, nx, ny))
            total_directed_edges += len(neighbors)
            available = []
            for ni, nj in neighbors:
                row = edge_metrics.get(((i, j), (ni, nj)))
                if row is None:
                    continue
                if row.get("failed", False):
                    failure_count += 1
                available.append(row)

            valid = [
                row
                for row in available
                if (not row["failed"])
                and np.isfinite(row["energy_per_distance"])
                and np.isfinite(row["minimum_energy_control_energy"])
            ]

            success_rate[j, i] = float(len(valid)) / float(len(neighbors)) if neighbors else np.nan

            if valid:
                local_mean_epd[j, i] = float(
                    np.mean([row["energy_per_distance"] for row in valid])
                )
                local_min_epd[j, i] = float(
                    np.min([row["energy_per_distance"] for row in valid])
                )
                local_mean_energy[j, i] = float(
                    np.mean([row["minimum_energy_control_energy"] for row in valid])
                )

    cumulative_energy, predecessor_i, predecessor_j = dijkstra_minimum_energy(
        center_index=center_index,
        edge_metrics=edge_metrics,
        nx=nx,
        ny=ny,
    )

    smoothed_local_mean_epd = smooth_field(local_mean_epd, passes=smoothing_passes)
    basin_map, basin_seeds = assign_energy_basins(smoothed_local_mean_epd)
    boundary_mask = compute_boundary_mask(basin_map)

    node_rows: List[Dict] = []
    for j in range(ny):
        for i in range(nx):
            node_rows.append(
                {
                    "i": i,
                    "j": j,
                    "x1": float(xs[i]),
                    "x2": float(ys[j]),
                    "objective": float(objective_map[j, i]),
                    "radius_from_center": float(radius_map[j, i]),
                    "cumulative_minimum_energy": float(cumulative_energy[j, i]) if np.isfinite(cumulative_energy[j, i]) else np.nan,
                    "local_mean_energy": float(local_mean_energy[j, i]) if np.isfinite(local_mean_energy[j, i]) else np.nan,
                    "local_mean_energy_per_distance": float(local_mean_epd[j, i]) if np.isfinite(local_mean_epd[j, i]) else np.nan,
                    "local_min_energy_per_distance": float(local_min_epd[j, i]) if np.isfinite(local_min_epd[j, i]) else np.nan,
                    "smoothed_local_mean_energy_per_distance": float(smoothed_local_mean_epd[j, i]) if np.isfinite(smoothed_local_mean_epd[j, i]) else np.nan,
                    "success_rate": float(success_rate[j, i]),
                    "basin_id": int(basin_map[j, i]),
                    "predecessor_i": int(predecessor_i[j, i]),
                    "predecessor_j": int(predecessor_j[j, i]),
                }
            )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(directed_edge_rows)

    basin_rows: List[Dict] = []
    for basin_id, (seed_i, seed_j) in enumerate(basin_seeds):
        mask = basin_map == basin_id
        if not np.any(mask):
            continue
        basin_rows.append(
            {
                "basin_id": int(basin_id),
                "seed_i": int(seed_i),
                "seed_j": int(seed_j),
                "seed_x1": float(xs[seed_i]),
                "seed_x2": float(ys[seed_j]),
                "seed_smoothed_local_mean_epd": float(smoothed_local_mean_epd[seed_j, seed_i]),
                "node_count": int(np.sum(mask)),
                "mean_objective": float(np.nanmean(objective_map[mask])),
                "mean_local_mean_epd": float(np.nanmean(local_mean_epd[mask])),
                "mean_cumulative_energy": float(np.nanmean(cumulative_energy[mask])),
                "min_cumulative_energy": float(np.nanmin(cumulative_energy[mask])),
                "max_cumulative_energy": float(np.nanmax(cumulative_energy[mask])),
                "mean_success_rate": float(np.nanmean(success_rate[mask])),
            }
        )

    basins_df = pd.DataFrame(basin_rows)
    if not basins_df.empty:
        basins_df = basins_df.sort_values(
            by=["seed_smoothed_local_mean_epd", "node_count"],
            ascending=[True, False],
        ).reset_index(drop=True)

    radial_profile_df = compute_radial_profile(radius_map, cumulative_energy, bin_count=12)
    local_epd_profile_df = compute_radial_profile(radius_map, local_mean_epd, bin_count=12)

    return {
        "parameters": parameters,
        "center_state": np.array([center_x, center_y]),
        "center_index": center_index,
        "xs": xs,
        "ys": ys,
        "objective_map": objective_map,
        "radius_map": radius_map,
        "local_mean_energy": local_mean_energy,
        "local_mean_epd": local_mean_epd,
        "smoothed_local_mean_epd": smoothed_local_mean_epd,
        "success_rate": success_rate,
        "cumulative_energy": cumulative_energy,
        "basin_map": basin_map,
        "basin_seeds": basin_seeds,
        "boundary_mask": boundary_mask,
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "basins_df": basins_df,
        "radial_profile_df": radial_profile_df,
        "local_epd_profile_df": local_epd_profile_df,
        "softmin_sharpness": softmin_sharpness,
        "terminal_time": terminal_time,
        "failure_count": failure_count,
        "total_directed_edges": total_directed_edges,
    }


def save_partial_outputs(atlas):
    atlas["edges_df"].to_csv(PARTIAL_EDGES_CSV, index=False)
    atlas["nodes_df"].to_csv(PARTIAL_NODES_CSV, index=False)
    with open(PARTIAL_SUMMARY_TXT, "w") as f:
        f.write("Partial Energy Basin Atlas\n\n")
        f.write(f"center_state = {atlas['center_state'].tolist()}\n")
        f.write(f"softmin_sharpness = {atlas['softmin_sharpness']:.4f}\n")
        f.write(f"terminal_time = {atlas['terminal_time']:.4f}\n")
        f.write(f"grid_shape = {len(atlas['ys'])} x {len(atlas['xs'])}\n")
        f.write(f"basin_count = {len(atlas['basin_seeds'])}\n")
        f.write(
            f"failure_edges = {atlas['failure_count']} / {atlas['total_directed_edges']}\n\n"
        )
        f.write("Basin summary:\n")
        if not atlas["basins_df"].empty:
            f.write(atlas["basins_df"].to_string(index=False))
        else:
            f.write("No valid basins found yet.")
    try:
        make_comparison_figure(atlas, PARTIAL_COMPARISON_PNG)
    except Exception:
        pass


def _save_checkpoint(
    config_sig,
    xs,
    ys,
    center_index,
    edge_metrics,
    directed_edge_rows,
    edge_counter,
    failure_count,
):
    payload = {
        "config_signature": config_sig,
        "xs": xs,
        "ys": ys,
        "center_index": center_index,
        "edge_metrics": edge_metrics,
        "directed_edge_rows": directed_edge_rows,
        "edge_counter": int(edge_counter),
        "failure_count": int(failure_count),
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(payload, f)

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "edge_counter": int(edge_counter),
                "failure_count": int(failure_count),
                "checkpoint_path": str(CHECKPOINT_PATH),
                "partial_edges_csv": str(PARTIAL_EDGES_CSV),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _load_checkpoint(config_sig):
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        with open(CHECKPOINT_PATH, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    if payload.get("config_signature") != config_sig:
        return None
    return payload


def maybe_clear_caches():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def add_seed_markers(ax, atlas):
    xs = atlas["xs"]
    ys = atlas["ys"]
    for basin_id, (seed_i, seed_j) in enumerate(atlas["basin_seeds"]):
        ax.scatter(xs[seed_i], ys[seed_j], marker="x", s=80, color="black")
        ax.text(xs[seed_i], ys[seed_j], f"B{basin_id}", color="black")


def percentile_limits(field: np.ndarray, lower: float = 5.0, upper: float = 95.0):
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        return None, None
    return float(np.percentile(valid, lower)), float(np.percentile(valid, upper))


def make_comparison_figure(atlas, file_path: Path):
    xs = atlas["xs"]
    ys = atlas["ys"]
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    center_x, center_y = atlas["center_state"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    contour = ax.contourf(xs, ys, atlas["objective_map"], levels=25)
    fig.colorbar(contour, ax=ax, label="Objective")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    add_seed_markers(ax, atlas)
    ax.set_title("Objective landscape with center and basin seeds")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[0, 1]
    local_plot = np.ma.masked_invalid(atlas["smoothed_local_mean_epd"])
    vmin, vmax = percentile_limits(atlas["smoothed_local_mean_epd"])
    im = ax.imshow(
        local_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Smoothed local mean energy / distance")
    ax.contour(xs, ys, atlas["boundary_mask"], levels=[0.5], linewidths=1.0, colors="purple")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    add_seed_markers(ax, atlas)
    ax.set_title("Local steering difficulty field")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[0, 2]
    im = ax.imshow(
        atlas["success_rate"],
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, label="Outgoing edge success rate")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    add_seed_markers(ax, atlas)
    ax.set_title("Solver success-rate field")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[1, 0]
    cumulative_plot = np.ma.masked_invalid(atlas["cumulative_energy"])
    vmin, vmax = percentile_limits(atlas["cumulative_energy"])
    im = ax.imshow(
        cumulative_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Cumulative minimum energy from center")
    ax.contour(xs, ys, atlas["boundary_mask"], levels=[0.5], linewidths=1.0, colors="purple")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    ax.set_title("Outward energy atlas via Dijkstra expansion")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[1, 1]
    basin_plot = np.ma.masked_where(atlas["basin_map"] < 0, atlas["basin_map"])
    im = ax.imshow(
        basin_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Energy basin id")
    ax.contour(xs, ys, atlas["boundary_mask"], levels=[0.5], linewidths=1.0, colors="purple")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    add_seed_markers(ax, atlas)
    ax.set_title("Energy-basin segmentation")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[1, 2]
    contour = ax.contourf(xs, ys, atlas["objective_map"], levels=25)
    fig.colorbar(contour, ax=ax, label="Objective")
    ax.contour(xs, ys, atlas["boundary_mask"], levels=[0.5], linewidths=1.3, colors="white")
    ax.scatter(center_x, center_y, marker="*", s=140, color="deepskyblue")
    add_seed_markers(ax, atlas)
    ax.set_title("Objective with estimated basin boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    fig.suptitle("Energy Basin Atlas", fontsize=16)
    fig.tight_layout()
    fig.savefig(file_path, dpi=220)
    plt.close(fig)


def make_radial_profile_figure(atlas, file_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    radial_df = atlas["radial_profile_df"]
    local_df = atlas["local_epd_profile_df"]

    if not radial_df.empty:
        ax.plot(
            radial_df["radius_mid"],
            radial_df["mean_value"],
            marker="o",
            label="mean cumulative energy",
        )

    ax.set_xlabel("Radius from center")
    ax.set_ylabel("Cumulative minimum energy")
    ax.set_title("Outward growth of steering cost")

    ax2 = ax.twinx()
    if not local_df.empty:
        ax2.plot(
            local_df["radius_mid"],
            local_df["mean_value"],
            marker="s",
            label="mean local energy/distance",
        )
    ax2.set_ylabel("Local mean energy / distance")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(file_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Final save outputs
# -----------------------------------------------------------------------------

def save_outputs(atlas):
    nodes_csv = OUTPUT_DIRECTORY / "energy_basin_atlas_nodes.csv"
    edges_csv = OUTPUT_DIRECTORY / "energy_basin_atlas_directed_edges.csv"
    basins_csv = OUTPUT_DIRECTORY / "energy_basin_atlas_basins.csv"
    radial_csv = OUTPUT_DIRECTORY / "energy_basin_atlas_radial_profile.csv"
    local_csv = OUTPUT_DIRECTORY / "energy_basin_atlas_local_epd_profile.csv"
    text_path = OUTPUT_DIRECTORY / "energy_basin_atlas_summary.txt"

    comparison_png = OUTPUT_DIRECTORY / "energy_basin_atlas_comparison.png"
    radial_png = OUTPUT_DIRECTORY / "energy_basin_atlas_radial_profile.png"

    atlas["nodes_df"].to_csv(nodes_csv, index=False)
    atlas["edges_df"].to_csv(edges_csv, index=False)
    atlas["basins_df"].to_csv(basins_csv, index=False)
    atlas["radial_profile_df"].to_csv(radial_csv, index=False)
    atlas["local_epd_profile_df"].to_csv(local_csv, index=False)

    with open(text_path, "w") as f:
        f.write("Energy Basin Atlas\n\n")
        f.write(f"center_state = {atlas['center_state'].tolist()}\n")
        f.write(f"softmin_sharpness = {atlas['softmin_sharpness']:.4f}\n")
        f.write(f"terminal_time = {atlas['terminal_time']:.4f}\n")
        f.write(f"grid_shape = {len(atlas['ys'])} x {len(atlas['xs'])}\n")
        f.write(f"basin_count = {len(atlas['basin_seeds'])}\n")
        f.write(
            f"failure_edges = {atlas['failure_count']} / {atlas['total_directed_edges']}\n\n"
        )
        f.write("Basin summary:\n")
        if not atlas["basins_df"].empty:
            f.write(atlas["basins_df"].to_string(index=False))
        else:
            f.write("No valid basins found.")
        f.write("\n\nNode sample:\n")
        f.write(atlas["nodes_df"].head(20).to_string(index=False))

    make_comparison_figure(atlas, comparison_png)
    make_radial_profile_figure(atlas, radial_png)

    print("\nSaved files:")
    print(" ", nodes_csv)
    print(" ", edges_csv)
    print(" ", basins_csv)
    print(" ", radial_csv)
    print(" ", local_csv)
    print(" ", text_path)
    print(" ", comparison_png)
    print(" ", radial_png)


# -----------------------------------------------------------------------------
# Atlas construction with checkpointing
# -----------------------------------------------------------------------------

def build_energy_basin_atlas(
    center_state=jnp.array([3.5, 4.0]),
    half_width_x: float = 2.5,
    half_width_y: float = 2.5,
    grid_size_x: int = 17,
    grid_size_y: int = 17,
    softmin_sharpness: float = 1.25,
    initial_time: float = 0.0,
    terminal_time: float = 1.0,
    smoothing_passes: int = 3,
    checkpoint_every: int = 25,
    cache_clear_every: int = 25,
    resume_from_checkpoint: bool = True,
):
    parameters, system, ode_solver_interface = create_landscape_context(softmin_sharpness)

    center_x = float(center_state[0])
    center_y = float(center_state[1])

    xs = np.linspace(center_x - half_width_x, center_x + half_width_x, grid_size_x)
    ys = np.linspace(center_y - half_width_y, center_y + half_width_y, grid_size_y)
    nx = len(xs)
    ny = len(ys)

    center_i = int(np.argmin(np.abs(xs - center_x)))
    center_j = int(np.argmin(np.abs(ys - center_y)))
    center_index = (center_i, center_j)

    config_sig = _config_signature(
        center_state,
        half_width_x,
        half_width_y,
        grid_size_x,
        grid_size_y,
        softmin_sharpness,
        initial_time,
        terminal_time,
        smoothing_passes,
    )

    print("\n==================== ENERGY BASIN ATLAS ====================\n")
    print(f"center state       = [{center_x:.4f}, {center_y:.4f}]")
    print(f"grid size          = {grid_size_x} x {grid_size_y}")
    print(f"x range            = [{xs[0]:.4f}, {xs[-1]:.4f}]")
    print(f"y range            = [{ys[0]:.4f}, {ys[-1]:.4f}]")
    print(f"softmin_sharpness  = {softmin_sharpness:.2f}")
    print(f"terminal_time      = {terminal_time:.2f}")
    print(f"checkpoint_every   = {checkpoint_every}")
    print(f"cache_clear_every  = {cache_clear_every}\n")

    total_directed_edges = 0
    for j in range(ny):
        for i in range(nx):
            total_directed_edges += sum(1 for _ in four_neighbors(i, j, nx, ny))

    edge_metrics: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]] = {}
    directed_edge_rows: List[Dict] = []
    edge_counter = 0
    failure_count = 0

    if resume_from_checkpoint:
        payload = _load_checkpoint(config_sig)
        if payload is not None:
            edge_metrics = payload.get("edge_metrics", {})
            directed_edge_rows = payload.get("directed_edge_rows", [])
            edge_counter = int(payload.get("edge_counter", len(edge_metrics)))
            failure_count = int(payload.get("failure_count", 0))
            print(
                f"Resuming from checkpoint: {edge_counter}/{total_directed_edges} directed edges, "
                f"failures={failure_count}"
            )

    try:
        for j in range(ny):
            for i in range(nx):
                initial_state = point_from_grid(xs, ys, i, j)
                for ni, nj in four_neighbors(i, j, nx, ny):
                    edge_key = ((i, j), (ni, nj))
                    if edge_key in edge_metrics:
                        continue

                    target_state = point_from_grid(xs, ys, ni, nj)
                    result = run_minimum_energy_case_with_context(
                        initial_state=initial_state,
                        target_state=target_state,
                        parameters=parameters,
                        system=system,
                        ode_solver_interface=ode_solver_interface,
                        initial_time=initial_time,
                        terminal_time=terminal_time,
                    )

                    if result["failed"]:
                        failure_count += 1

                    edge_metrics[edge_key] = result
                    directed_edge_rows.append(
                        {
                            "source_i": i,
                            "source_j": j,
                            "source_x1": float(xs[i]),
                            "source_x2": float(ys[j]),
                            "target_i": ni,
                            "target_j": nj,
                            "target_x1": float(xs[ni]),
                            "target_x2": float(ys[nj]),
                            "minimum_energy_control_energy": result["minimum_energy_control_energy"],
                            "energy_per_distance": result["energy_per_distance"],
                            "terminal_error": result["terminal_error"],
                            "failed": result["failed"],
                        }
                    )

                    edge_counter += 1
                    if edge_counter % 10 == 0 or edge_counter == total_directed_edges:
                        print(
                            f"computed directed edges: {edge_counter}/{total_directed_edges} | "
                            f"failures: {failure_count}"
                        )

                    do_checkpoint = (
                        edge_counter % checkpoint_every == 0
                        or edge_counter == total_directed_edges
                    )
                    if do_checkpoint:
                        partial_atlas = _build_partial_atlas_from_edges(
                            xs=xs,
                            ys=ys,
                            center_state=center_state,
                            center_index=center_index,
                            parameters=parameters,
                            edge_metrics=edge_metrics,
                            directed_edge_rows=directed_edge_rows,
                            softmin_sharpness=softmin_sharpness,
                            terminal_time=terminal_time,
                            smoothing_passes=smoothing_passes,
                        )
                        save_partial_outputs(partial_atlas)
                        _save_checkpoint(
                            config_sig=config_sig,
                            xs=xs,
                            ys=ys,
                            center_index=center_index,
                            edge_metrics=edge_metrics,
                            directed_edge_rows=directed_edge_rows,
                            edge_counter=edge_counter,
                            failure_count=failure_count,
                        )

                    if edge_counter % cache_clear_every == 0:
                        maybe_clear_caches()

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint before exit...")
        partial_atlas = _build_partial_atlas_from_edges(
            xs=xs,
            ys=ys,
            center_state=center_state,
            center_index=center_index,
            parameters=parameters,
            edge_metrics=edge_metrics,
            directed_edge_rows=directed_edge_rows,
            softmin_sharpness=softmin_sharpness,
            terminal_time=terminal_time,
            smoothing_passes=smoothing_passes,
        )
        save_partial_outputs(partial_atlas)
        _save_checkpoint(
            config_sig=config_sig,
            xs=xs,
            ys=ys,
            center_index=center_index,
            edge_metrics=edge_metrics,
            directed_edge_rows=directed_edge_rows,
            edge_counter=edge_counter,
            failure_count=failure_count,
        )
        raise
    except Exception:
        print("\nException occurred. Saving checkpoint before re-raising...")
        partial_atlas = _build_partial_atlas_from_edges(
            xs=xs,
            ys=ys,
            center_state=center_state,
            center_index=center_index,
            parameters=parameters,
            edge_metrics=edge_metrics,
            directed_edge_rows=directed_edge_rows,
            softmin_sharpness=softmin_sharpness,
            terminal_time=terminal_time,
            smoothing_passes=smoothing_passes,
        )
        save_partial_outputs(partial_atlas)
        _save_checkpoint(
            config_sig=config_sig,
            xs=xs,
            ys=ys,
            center_index=center_index,
            edge_metrics=edge_metrics,
            directed_edge_rows=directed_edge_rows,
            edge_counter=edge_counter,
            failure_count=failure_count,
        )
        raise

    atlas = _build_partial_atlas_from_edges(
        xs=xs,
        ys=ys,
        center_state=center_state,
        center_index=center_index,
        parameters=parameters,
        edge_metrics=edge_metrics,
        directed_edge_rows=directed_edge_rows,
        softmin_sharpness=softmin_sharpness,
        terminal_time=terminal_time,
        smoothing_passes=smoothing_passes,
    )

    _save_checkpoint(
        config_sig=config_sig,
        xs=xs,
        ys=ys,
        center_index=center_index,
        edge_metrics=edge_metrics,
        directed_edge_rows=directed_edge_rows,
        edge_counter=edge_counter,
        failure_count=failure_count,
    )
    save_partial_outputs(atlas)
    maybe_clear_caches()

    return atlas


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    atlas = build_energy_basin_atlas(
        center_state=jnp.array([3.5, 4.0]),
        half_width_x=2.5,
        half_width_y=2.5,
        grid_size_x=17,
        grid_size_y=17,
        softmin_sharpness=1.25,
        initial_time=0.0,
        terminal_time=1.0,
        smoothing_passes=3,
        checkpoint_every=25,
        cache_clear_every=25,
        resume_from_checkpoint=True,
    )

    print("\n==================== BASIN SUMMARY ====================\n")
    if atlas["basins_df"].empty:
        print("No valid basins found.")
    else:
        print(atlas["basins_df"].to_string(index=False))

    print(
        f"\nFailure edges: {atlas['failure_count']} / {atlas['total_directed_edges']}"
    )

    save_outputs(atlas)


if __name__ == "__main__":
    main()
