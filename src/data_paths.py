# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Canonical paths and cache tags for prepared federated partitions."""

from __future__ import annotations

import os
from pathlib import Path


PARTITION_DIRICHLET = "dirichlet"
PARTITION_MOBCORR_STRICT = "mobcorr_strict"


def partition_tag(options):
    """Return a stable identifier for the selected client-data partition."""
    if bool(options.get("pathe", False)):
        return "slice_{}".format(options["slice"])

    partition = str(options.get("data_partition", PARTITION_DIRICHLET)).lower()
    if partition == PARTITION_DIRICHLET:
        return "dir_{}".format(options["dirichlet"])
    if partition == PARTITION_MOBCORR_STRICT:
        mobility_seed = int(options.get("mobility_seed", 2025))
        return "mobcorr_strict_homeblock4_mseed{}".format(mobility_seed)
    raise ValueError("unsupported data_partition: {!r}".format(partition))


def federated_data_path(options, project_root="."):
    """Return the directory containing all per-client ``.npy`` files."""
    root = Path(project_root) / "data" / "federated_data"
    root /= str(options["dataset_name"])
    root /= "nc{}".format(int(options["num_of_clients"]))

    if bool(options.get("pathe", False)):
        return os.fspath(root / "slice{}".format(options["slice"]) / "pathe")

    partition = str(options.get("data_partition", PARTITION_DIRICHLET)).lower()
    if partition == PARTITION_DIRICHLET:
        return os.fspath(root / "dir{}".format(options["dirichlet"]))
    if partition == PARTITION_MOBCORR_STRICT:
        return os.fspath(root / partition_tag(options))
    raise ValueError("unsupported data_partition: {!r}".format(partition))


def validate_federated_data(path, num_clients):
    """Fail early when an explicitly prepared partition is incomplete."""
    path = Path(path)
    missing = [
        client_id
        for client_id in range(1, int(num_clients) + 1)
        if not (path / "client_{}".format(client_id) / "train_data.npy").is_file()
    ]
    test_file = path / "test_data" / "test_data.npy"
    if missing or not test_file.is_file():
        details = []
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            suffix = " ..." if len(missing) > 10 else ""
            details.append("missing clients: {}{}".format(preview, suffix))
        if not test_file.is_file():
            details.append("missing test data: {}".format(test_file))
        raise FileNotFoundError(
            "prepared federated partition is incomplete at {} ({})".format(
                path, "; ".join(details)
            )
        )
    return os.fspath(path)
