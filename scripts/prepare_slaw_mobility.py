#!/usr/bin/env python3
# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Generate reproducible SLAW client-to-ES association sequences.

The script invokes BonnMotion, samples continuous positions, maps them to a
fixed regular ES coverage grid, and saves one association matrix per mobility
seed.  Every HFL method can then read the same .npz file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mobility import map_positions_to_regular_grid, regular_grid_centers


DEFAULT_SEEDS = (2025, 2026, 2027)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare offline SLAW client-to-ES association matrices."
    )
    parser.add_argument("--num_of_clients", type=int, default=200)
    parser.add_argument("--num_of_edges", type=int, default=10)
    parser.add_argument("--round_num", type=int, default=500)
    parser.add_argument("--edge_epoch", type=int, default=2)
    parser.add_argument(
        "--activity_mode",
        choices=("global", "home_pair", "home_block"),
        default="global",
        help=(
            "global uses the full area; home_pair maps every SLAW trace into "
            "a persistent two-adjacent-ES activity region; home_block maps it "
            "into a persistent 2x2 four-ES activity region"
        ),
    )
    parser.add_argument(
        "--history_steps",
        type=int,
        default=0,
        help=(
            "extra association states generated before formal training; the "
            "prefix is stored separately for mobility-correlated data preparation"
        ),
    )
    parser.add_argument(
        "--mobility_seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="independent walker seeds; all seeds share one spatial landscape",
    )
    parser.add_argument("--sample_interval", type=float, default=60.0)
    parser.add_argument("--area_width", type=float, default=1000.0)
    parser.add_argument("--area_height", type=float, default=1000.0)
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_columns", type=int, default=5)
    parser.add_argument("--ignore_seconds", type=float, default=3600.0)
    parser.add_argument("--num_waypoints", type=int, default=1000)
    parser.add_argument("--min_pause", type=float, default=10.0)
    parser.add_argument("--max_pause", type=float, default=50.0)
    parser.add_argument("--levy_exponent", type=float, default=1.0)
    parser.add_argument("--hurst", type=float, default=0.75)
    parser.add_argument("--distance_weight", type=float, default=3.0)
    parser.add_argument("--cluster_range", type=float, default=50.0)
    parser.add_argument("--cluster_ratio", type=int, default=5)
    parser.add_argument("--waypoint_ratio", type=int, default=5)
    parser.add_argument(
        "--bonnmotion_bin",
        type=Path,
        default=None,
        help="path to BonnMotion's bin/bm (auto-detected under ~/tools if omitted)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/mobility/slaw")
    )
    return parser.parse_args()


def find_bonnmotion_bin(requested):
    candidates = []
    if requested is not None:
        candidates.append(Path(requested))
    if os.environ.get("BONNMOTION_HOME"):
        candidates.append(Path(os.environ["BONNMOTION_HOME"]) / "bin" / "bm")
    candidates.append(Path.home() / "tools" / "bonnmotion-3.0.1" / "bin" / "bm")
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "BonnMotion executable not found. Pass --bonnmotion_bin or install "
        "BonnMotion under ~/tools/bonnmotion-3.0.1."
    )


def run_checked(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def parse_one_file(path, num_clients, num_steps, interval):
    positions = np.full((num_clients, num_steps, 2), np.nan, dtype=np.float64)
    seen = np.zeros((num_clients, num_steps), dtype=bool)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) != 6:
            raise ValueError("unexpected TheONE header in {}".format(path))
        for line_number, line in enumerate(handle, start=2):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "malformed TheONE row {} in {}".format(line_number, path)
                )
            time_value, node_value, x_value, y_value = map(float, parts)
            node = int(node_value)
            step = int(round(time_value / interval))
            if not (0 <= node < num_clients and 0 <= step < num_steps):
                continue
            if seen[node, step]:
                raise ValueError(
                    "duplicate position for node {} step {} in {}".format(
                        node, step, path
                    )
                )
            positions[node, step] = (x_value, y_value)
            seen[node, step] = True
    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError("{} sampled positions are missing from {}".format(missing, path))
    return positions


def generate_positions(args, bm, seed, work_dir, fixed_waypoints):
    num_steps = args.history_steps + args.round_num * args.edge_epoch
    duration = num_steps * args.sample_interval
    scenario = work_dir / "slaw_seed{}".format(seed)
    command = [
        bm,
        "-f",
        scenario,
        "SLAW",
        "-n",
        args.num_of_clients,
        "-d",
        duration,
        "-i",
        args.ignore_seconds,
        "-x",
        args.area_width,
        "-y",
        args.area_height,
        "-R",
        seed,
        "-p",
        args.min_pause,
        "-P",
        args.max_pause,
        "-b",
        args.levy_exponent,
        "-h",
        args.hurst,
        "-l",
        args.distance_weight,
        "-r",
        args.cluster_range,
        "-Q",
        args.cluster_ratio,
        "-W",
        args.waypoint_ratio,
    ]
    if fixed_waypoints is None:
        command.extend(["-w", args.num_waypoints])
    else:
        command.extend(["-F", fixed_waypoints])
    run_checked(command)
    run_checked([bm, "TheONEFile", "-f", scenario, "-l", args.sample_interval])
    positions = parse_one_file(
        Path(str(scenario) + ".one"),
        args.num_of_clients,
        num_steps,
        args.sample_interval,
    )
    generated_waypoints = Path(str(scenario) + "_waypoints.csv")
    return positions, generated_waypoints


def mobility_statistics(associations, num_edges):
    transitions = associations[:, 1:] != associations[:, :-1]
    handover_ratio = float(transitions.mean()) if transitions.size else 0.0
    dwell_lengths = []
    for sequence in associations:
        if sequence.size == 0:
            continue
        change_points = np.flatnonzero(sequence[1:] != sequence[:-1]) + 1
        dwell_lengths.extend(np.diff(np.r_[0, change_points, sequence.size]).tolist())
    occupancy = np.bincount(
        associations.reshape(-1), minlength=num_edges
    ).astype(float)
    occupancy /= occupancy.sum()
    return {
        "handover_ratio": handover_ratio,
        "stay_ratio": 1.0 - handover_ratio,
        "mean_dwell_steps": float(np.mean(dwell_lengths)),
        "mean_distinct_es": float(
            np.mean([np.unique(row).size for row in associations])
        ),
        "occupancy_share": occupancy.tolist(),
        "occupancy_cv": float(occupancy.std() / occupancy.mean()),
    }


def map_to_home_pair_regions(
    positions,
    area_width,
    area_height,
    rows,
    columns,
    seed,
):
    """Scale and translate SLAW paths into persistent horizontal ES pairs."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (num_clients, steps, 2)")
    if int(columns) < 2:
        raise ValueError("home_pair activity requires at least two grid columns")

    num_clients = positions.shape[0]
    num_edges = int(rows) * int(columns)
    home_edges = np.arange(num_clients, dtype=np.int64) % num_edges
    rng = np.random.default_rng(int(seed))
    rng.shuffle(home_edges)

    home_rows = home_edges // int(columns)
    home_columns = home_edges % int(columns)
    neighbor_columns = np.where(
        home_columns < int(columns) - 1,
        home_columns + 1,
        home_columns - 1,
    )
    neighbor_edges = home_rows * int(columns) + neighbor_columns
    activity_edges = np.stack((home_edges, neighbor_edges), axis=1)

    upper = np.nextafter(1.0, 0.0)
    normalized_x = np.clip(positions[..., 0] / float(area_width), 0.0, upper)
    normalized_y = np.clip(positions[..., 1] / float(area_height), 0.0, upper)
    left_columns = np.minimum(home_columns, neighbor_columns)
    cell_width = float(area_width) / int(columns)
    cell_height = float(area_height) / int(rows)

    mapped = np.empty_like(positions, dtype=np.float64)
    mapped[..., 0] = (
        left_columns[:, None] + 2.0 * normalized_x
    ) * cell_width
    mapped[..., 1] = (
        home_rows[:, None] + normalized_y
    ) * cell_height
    x_upper = np.nextafter(
        (left_columns + 2) * cell_width,
        left_columns * cell_width,
    )
    y_upper = np.nextafter(
        (home_rows + 1) * cell_height,
        home_rows * cell_height,
    )
    mapped[..., 0] = np.minimum(mapped[..., 0], x_upper[:, None])
    mapped[..., 1] = np.minimum(mapped[..., 1], y_upper[:, None])
    return mapped, home_edges, activity_edges


def map_to_home_block_regions(
    positions,
    area_width,
    area_height,
    rows,
    columns,
    seed,
):
    """Scale and translate SLAW paths into persistent 2x2 ES blocks."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (num_clients, steps, 2)")
    if int(rows) < 2 or int(columns) < 2:
        raise ValueError("home_block activity requires at least a 2x2 ES grid")

    rows = int(rows)
    columns = int(columns)
    num_clients = positions.shape[0]
    num_edges = rows * columns
    home_edges = np.arange(num_clients, dtype=np.int64) % num_edges
    rng = np.random.default_rng(int(seed))
    rng.shuffle(home_edges)

    home_rows = home_edges // columns
    home_columns = home_edges % columns
    top_rows = np.minimum(home_rows, rows - 2)
    left_columns = np.minimum(home_columns, columns - 2)
    activity_edges = np.empty((num_clients, 4), dtype=np.int64)
    for client_id in range(num_clients):
        block = [
            (top_rows[client_id] + row_offset) * columns
            + left_columns[client_id]
            + column_offset
            for row_offset in range(2)
            for column_offset in range(2)
        ]
        home = int(home_edges[client_id])
        activity_edges[client_id] = [home] + [edge for edge in block if edge != home]

    upper = np.nextafter(1.0, 0.0)
    normalized_x = np.clip(positions[..., 0] / float(area_width), 0.0, upper)
    normalized_y = np.clip(positions[..., 1] / float(area_height), 0.0, upper)
    cell_width = float(area_width) / columns
    cell_height = float(area_height) / rows

    mapped = np.empty_like(positions, dtype=np.float64)
    mapped[..., 0] = (
        left_columns[:, None] + 2.0 * normalized_x
    ) * cell_width
    mapped[..., 1] = (
        top_rows[:, None] + 2.0 * normalized_y
    ) * cell_height
    x_upper = np.nextafter(
        (left_columns + 2) * cell_width,
        left_columns * cell_width,
    )
    y_upper = np.nextafter(
        (top_rows + 2) * cell_height,
        top_rows * cell_height,
    )
    mapped[..., 0] = np.minimum(mapped[..., 0], x_upper[:, None])
    mapped[..., 1] = np.minimum(mapped[..., 1], y_upper[:, None])
    return mapped, home_edges, activity_edges


def main():
    args = parse_args()
    if args.num_of_clients <= 0 or args.num_of_edges <= 0:
        raise ValueError("num_of_clients and num_of_edges must be positive")
    if args.grid_rows <= 0 or args.grid_columns <= 0:
        raise ValueError("grid_rows and grid_columns must be positive")
    if args.grid_rows * args.grid_columns != args.num_of_edges:
        raise ValueError(
            "grid_rows * grid_columns must equal num_of_edges; found {} * {} "
            "!= {}".format(
                args.grid_rows, args.grid_columns, args.num_of_edges
            )
        )
    if args.round_num <= 0 or args.edge_epoch <= 0 or args.sample_interval <= 0:
        raise ValueError("round counts and sample_interval must be positive")
    if args.history_steps < 0:
        raise ValueError("history_steps must be non-negative")
    if args.activity_mode in ("home_pair", "home_block") and args.history_steps:
        raise ValueError("fixed local activity modes do not require history_steps")
    if not args.mobility_seeds:
        raise ValueError("at least one mobility seed is required")

    bm = find_bonnmotion_bin(args.bonnmotion_bin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_steps = args.round_num * args.edge_epoch
    total_steps = args.history_steps + training_steps
    landscape_seed = int(args.mobility_seeds[0])
    landscape_path = args.output_dir / "slaw_landscape_seed{}_waypoints.csv".format(
        landscape_seed
    )

    positions_by_seed = {}
    with tempfile.TemporaryDirectory(prefix="slaw_bonnmotion_") as temporary:
        work_dir = Path(temporary)
        for index, seed in enumerate(args.mobility_seeds):
            positions, generated_waypoints = generate_positions(
                args,
                bm,
                int(seed),
                work_dir,
                None if index == 0 else landscape_path,
            )
            positions_by_seed[int(seed)] = positions
            if index == 0:
                if not generated_waypoints.is_file():
                    raise FileNotFoundError(
                        "BonnMotion did not write {}".format(generated_waypoints)
                    )
                shutil.copy2(generated_waypoints, landscape_path)

        es_positions = regular_grid_centers(
            args.area_width,
            args.area_height,
            args.grid_rows,
            args.grid_columns,
        )
        placement = "fixed {}x{} regular grid".format(
            args.grid_columns, args.grid_rows
        )

        es_path = args.output_dir / "slaw_es_positions_m{}_e{}_seed{}.npy".format(
            args.num_of_clients, args.num_of_edges, landscape_seed
        )
        np.save(es_path, es_positions.astype(np.float64))

        for seed, positions in positions_by_seed.items():
            home_edges = None
            activity_edges = None
            if args.activity_mode == "home_pair":
                positions, home_edges, activity_edges = map_to_home_pair_regions(
                    positions,
                    args.area_width,
                    args.area_height,
                    args.grid_rows,
                    args.grid_columns,
                    seed,
                )
            elif args.activity_mode == "home_block":
                positions, home_edges, activity_edges = map_to_home_block_regions(
                    positions,
                    args.area_width,
                    args.area_height,
                    args.grid_rows,
                    args.grid_columns,
                    seed,
                )
            all_associations = map_positions_to_regular_grid(
                positions,
                args.area_width,
                args.area_height,
                args.grid_rows,
                args.grid_columns,
            )
            if activity_edges is not None:
                valid = np.any(
                    all_associations[:, :, None] == activity_edges[:, None, :],
                    axis=2,
                )
                if not valid.all():
                    raise AssertionError("a trajectory left its fixed activity region")
            history_associations = all_associations[:, :args.history_steps]
            associations = all_associations[
                :, args.history_steps:args.history_steps + training_steps
            ]
            history_positions = positions[:, :args.history_steps]
            training_positions = positions[
                :, args.history_steps:args.history_steps + training_steps
            ]
            stats = mobility_statistics(associations, args.num_of_edges)
            metadata = {
                "model": "SLAW",
                "mobility_seed": seed,
                "landscape_seed": landscape_seed,
                "num_clients": args.num_of_clients,
                "num_edges": args.num_of_edges,
                "num_steps": training_steps,
                "history_steps": args.history_steps,
                "total_generated_steps": total_steps,
                "round_num": args.round_num,
                "edge_epoch": args.edge_epoch,
                "sample_interval_seconds": args.sample_interval,
                "area_width_m": args.area_width,
                "area_height_m": args.area_height,
                "grid_rows": args.grid_rows,
                "grid_columns": args.grid_columns,
                "cell_width_m": args.area_width / args.grid_columns,
                "cell_height_m": args.area_height / args.grid_rows,
                "ignore_seconds": args.ignore_seconds,
                "num_waypoints": args.num_waypoints,
                "min_pause_seconds": args.min_pause,
                "max_pause_seconds": args.max_pause,
                "levy_exponent": args.levy_exponent,
                "hurst": args.hurst,
                "distance_weight": args.distance_weight,
                "cluster_range_m": args.cluster_range,
                "cluster_ratio_divisor": args.cluster_ratio,
                "waypoint_ratio_divisor": args.waypoint_ratio,
                "association_rule": "fixed non-overlapping regular grid cell",
                "es_placement": placement,
                "activity_mode": args.activity_mode,
                "activity_region_size": (
                    int(activity_edges.shape[1])
                    if activity_edges is not None
                    else args.num_of_edges
                ),
                "statistics": stats,
            }
            if args.activity_mode == "home_pair":
                filename = "slaw_m{}_e{}_t{}_homepair_seed{}.npz".format(
                    args.num_of_clients,
                    args.num_of_edges,
                    training_steps,
                    seed,
                )
            elif args.activity_mode == "home_block":
                filename = "slaw_m{}_e{}_t{}_homeblock4_seed{}.npz".format(
                    args.num_of_clients,
                    args.num_of_edges,
                    training_steps,
                    seed,
                )
            elif args.history_steps:
                filename = "slaw_m{}_e{}_t{}_h{}_seed{}.npz".format(
                    args.num_of_clients,
                    args.num_of_edges,
                    training_steps,
                    args.history_steps,
                    seed,
                )
            else:
                filename = "slaw_m{}_e{}_t{}_seed{}.npz".format(
                    args.num_of_clients, args.num_of_edges, training_steps, seed
                )
            output = args.output_dir / filename
            payload = {
                "associations": associations,
                "history_associations": history_associations,
                "positions": training_positions.astype(np.float32),
                "history_positions": history_positions.astype(np.float32),
                "es_positions": es_positions.astype(np.float32),
                "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            }
            if home_edges is not None:
                payload["home_edges"] = home_edges.astype(np.int16)
                payload["activity_edges"] = activity_edges.astype(np.int16)
            np.savez_compressed(output, **payload)
            print("Saved {}".format(output))
            print(
                "  handover={:.4f}, mean dwell={:.2f}, distinct ES={:.2f}, "
                "occupancy CV={:.3f}".format(
                    stats["handover_ratio"],
                    stats["mean_dwell_steps"],
                    stats["mean_distinct_es"],
                    stats["occupancy_cv"],
                )
            )
    print("Saved fixed-grid ES positions to {}".format(es_path))
    print("Saved shared SLAW landscape to {}".format(landscape_path))


if __name__ == "__main__":
    main()
