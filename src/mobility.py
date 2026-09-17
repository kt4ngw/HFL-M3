# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Mobility inputs shared by all cloud trainers.

Precomputed mobility files intentionally contain only exogenous client--ES
associations.  The learning and mapping code should not know how the trajectory
was generated (SLAW today, a real trace in a future experiment).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np


def random_es_adjacency(num_edges, min_links=3, max_links=5, seed=2025):
    """Build the deterministic ES graph shared by mobility and HFL."""
    num_edges = int(num_edges)
    min_links = int(min_links)
    max_links = int(max_links)
    if not 0 <= min_links <= max_links < num_edges:
        raise ValueError(
            "ES degrees require 0 <= min_links <= max_links < num_edges"
        )

    rng = np.random.default_rng(int(seed))
    adjacency = np.zeros((num_edges, num_edges), dtype=np.int8)
    target_degree = rng.integers(min_links, max_links, size=num_edges)
    nodes = list(range(num_edges))
    for edge in nodes:
        while adjacency[edge].sum() < target_degree[edge]:
            candidates = [
                neighbor
                for neighbor in nodes
                if neighbor != edge and adjacency[edge, neighbor] == 0
            ]
            if not candidates:
                break
            neighbor = int(rng.choice(candidates))
            adjacency[edge, neighbor] = 1
            adjacency[neighbor, edge] = 1

    for edge in nodes:
        if adjacency[edge].sum() == 0:
            neighbor = int(rng.choice([node for node in nodes if node != edge]))
            adjacency[edge, neighbor] = 1
            adjacency[neighbor, edge] = 1

    np.fill_diagonal(adjacency, 0)
    return adjacency


def regular_grid_centers(area_width, area_height, rows, columns):
    """Return row-major ES centers for a fixed rectangular coverage grid."""
    area_width = float(area_width)
    area_height = float(area_height)
    rows = int(rows)
    columns = int(columns)
    if area_width <= 0 or area_height <= 0:
        raise ValueError("mobility-area dimensions must be positive")
    if rows <= 0 or columns <= 0:
        raise ValueError("grid rows and columns must be positive")

    cell_width = area_width / columns
    cell_height = area_height / rows
    return np.asarray(
        [
            ((column + 0.5) * cell_width, (row + 0.5) * cell_height)
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float64,
    )


def map_positions_to_regular_grid(
    positions, area_width, area_height, rows, columns
):
    """Map positions to non-overlapping fixed grid cells with row-major IDs."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim < 1 or positions.shape[-1] != 2:
        raise ValueError("positions must have final dimension 2")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain only finite coordinates")

    centers = regular_grid_centers(area_width, area_height, rows, columns)
    del centers  # Validate dimensions once; centers are not needed for mapping.
    area_width = float(area_width)
    area_height = float(area_height)
    rows = int(rows)
    columns = int(columns)
    x = positions[..., 0]
    y = positions[..., 1]
    tolerance = 1e-9
    if (
        np.any(x < -tolerance)
        or np.any(x > area_width + tolerance)
        or np.any(y < -tolerance)
        or np.any(y > area_height + tolerance)
    ):
        raise ValueError("positions fall outside the configured mobility area")

    column_ids = np.floor(x / (area_width / columns)).astype(np.int64)
    row_ids = np.floor(y / (area_height / rows)).astype(np.int64)
    column_ids = np.clip(column_ids, 0, columns - 1)
    row_ids = np.clip(row_ids, 0, rows - 1)
    return (row_ids * columns + column_ids).astype(np.int16)


def default_slaw_path(options):
    """Return an exact or sufficiently long prepared SLAW realization."""
    steps = int(options["round_num"]) * int(options["edge_epoch"])
    num_clients = int(options["num_of_clients"])
    num_edges = int(options["num_of_edges"])
    seed = int(options.get("mobility_seed", 2025))
    directory = Path(".") / "data" / "mobility" / "slaw"
    correlated = str(options.get("data_partition", "dirichlet")) == "mobcorr_strict"
    if correlated:
        filename = "slaw_m{}_e{}_t{}_homeblock4_seed{}.npz".format(
            num_clients, num_edges, steps, seed
        )
        pattern = "slaw_m{}_e{}_t*_homeblock4_seed{}.npz".format(
            num_clients, num_edges, seed
        )
        step_pattern = re.compile(r"_t(\d+)_homeblock4")
    else:
        filename = "slaw_m{}_e{}_t{}_seed{}.npz".format(
            num_clients, num_edges, steps, seed
        )
        pattern = "slaw_m{}_e{}_t*_seed{}.npz".format(
            num_clients, num_edges, seed
        )
        step_pattern = re.compile(r"_t(\d+)_seed")
    exact = directory / filename
    if exact.is_file():
        return str(exact)

    eligible = []
    for candidate in directory.glob(pattern):
        match = step_pattern.search(candidate.name)
        if match and int(match.group(1)) >= steps:
            eligible.append((int(match.group(1)), candidate))
    if eligible:
        return str(min(eligible, key=lambda item: item[0])[1])
    return str(exact)


class PrecomputedMobility:
    """Read and validate a client--ES association matrix once before training."""

    def __init__(self, path, num_clients, num_edges, num_steps):
        self.path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                "prepared mobility file does not exist: {}\n"
                "Run the matching prepare_*_mobility.py script first, or pass the correct "
                "--mobility_file.".format(self.path)
            )

        with np.load(self.path, allow_pickle=False) as saved:
            if "associations" not in saved:
                raise KeyError(
                    "mobility file {} has no 'associations' array".format(self.path)
                )
            associations = np.asarray(saved["associations"])
            metadata_json = (
                str(saved["metadata_json"].item())
                if "metadata_json" in saved
                else "{}"
            )
            adjacency = (
                np.asarray(saved["adjacency"])
                if "adjacency" in saved
                else None
            )
            transition_matrix = (
                np.asarray(saved["transition_matrix"])
                if "transition_matrix" in saved
                else None
            )

        expected_clients = int(num_clients)
        expected_steps = int(num_steps)
        if (
            associations.ndim != 2
            or associations.shape[0] != expected_clients
            or associations.shape[1] < expected_steps
        ):
            raise ValueError(
                "mobility associations have shape {}, expected ({}, T) with "
                "T >= {} for num_of_clients={} and round_num*edge_epoch={}".format(
                    associations.shape,
                    expected_clients,
                    expected_steps,
                    num_clients,
                    num_steps,
                )
            )
        # A long prepared realization can be reused by shorter experiments.
        associations = associations[:, :expected_steps]
        if not np.issubdtype(associations.dtype, np.integer):
            if not np.all(np.equal(associations, np.floor(associations))):
                raise ValueError("mobility associations must contain integer ES IDs")
            associations = associations.astype(np.int64)
        if associations.size:
            minimum = int(associations.min())
            maximum = int(associations.max())
            if minimum < 0 or maximum >= int(num_edges):
                raise ValueError(
                    "mobility ES IDs must be in [0, {}], found [{}, {}]".format(
                        int(num_edges) - 1, minimum, maximum
                    )
                )

        self.associations = associations.astype(np.int64, copy=False)
        self.adjacency = None
        if adjacency is not None:
            if adjacency.shape != (int(num_edges), int(num_edges)):
                raise ValueError(
                    "mobility adjacency has shape {}, expected ({}, {})".format(
                        adjacency.shape, int(num_edges), int(num_edges)
                    )
                )
            self.adjacency = adjacency.astype(np.int8, copy=False)
        self.transition_matrix = None
        if transition_matrix is not None:
            if transition_matrix.shape != (int(num_edges), int(num_edges)):
                raise ValueError(
                    "mobility transition matrix has shape {}, expected ({}, {})".format(
                        transition_matrix.shape, int(num_edges), int(num_edges)
                    )
                )
            self.transition_matrix = transition_matrix.astype(float, copy=False)
        try:
            self.metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid metadata_json in {}".format(self.path)) from exc

    def edges_at(self, step):
        step = int(step)
        if step < 0 or step >= self.associations.shape[1]:
            raise IndexError(
                "mobility step {} is outside prepared range [0, {})".format(
                    step, self.associations.shape[1]
                )
            )
        return self.associations[:, step]
