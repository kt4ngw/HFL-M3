#!/usr/bin/env python3
# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Generate federated client ``.npy`` files without starting training.

Run this script from any directory.  Raw datasets are read through the
project's existing ``GetDataSet`` implementation, so the produced files are
identical to the ones that ``main.py`` would create.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_ALIASES = {
    "fmnist": "fashionmnist",
    "fashion-mnist": "fashionmnist",
    "fashionmnist": "fashionmnist",
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
}

DEFAULT_NUM_CLASSES = {
    "fashionmnist": 10,
    "cifar10": 10,
    # This project trains CIFAR-100 with its 20 coarse labels.
    "cifar100": 20,
}

RAW_DATA_HINTS = {
    "fashionmnist": "data/FashionMNIST/raw/*.gz",
    "cifar10": "data/cifar-10-batches-py/",
    "cifar100": "data/cifar-100-python/",
}


def canonical_dataset_name(name):
    """Return the dataset spelling used in federated-data paths."""
    normalized = name.strip().lower()
    try:
        return DATASET_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_NUM_CLASSES))
        raise ValueError(
            f"unsupported dataset {name!r}; choose one of: {supported}"
        ) from exc


def federated_data_path(dataset_name, num_clients, dirichlet, pathe, slices):
    base = (
        PROJECT_ROOT
        / "data"
        / "federated_data"
        / dataset_name
        / f"nc{num_clients}"
    )
    if pathe:
        return base / f"slice{slices}" / "pathe"
    return base / f"dir{dirichlet}"


def inspect_split(output_dir, num_clients):
    """Validate generated files and return a compact dataset summary."""
    missing_clients = [
        client_id
        for client_id in range(1, num_clients + 1)
        if not (output_dir / f"client_{client_id}" / "train_data.npy").is_file()
    ]
    test_file = output_dir / "test_data" / "test_data.npy"

    if missing_clients or not test_file.is_file():
        details = []
        if missing_clients:
            preview = ", ".join(map(str, missing_clients[:10]))
            suffix = " ..." if len(missing_clients) > 10 else ""
            details.append(
                f"missing client train files: {preview}{suffix} "
                f"({len(missing_clients)} total)"
            )
        if not test_file.is_file():
            details.append(f"missing test file: {test_file}")
        raise RuntimeError("; ".join(details))

    train_rows = 0
    row_width = None
    for client_id in range(1, num_clients + 1):
        client_file = output_dir / f"client_{client_id}" / "train_data.npy"
        array = np.load(client_file, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] < 2:
            raise RuntimeError(f"invalid client array shape in {client_file}: {array.shape}")
        train_rows += array.shape[0]
        if row_width is None:
            row_width = array.shape[1]
        elif array.shape[1] != row_width:
            raise RuntimeError(
                f"inconsistent row width in {client_file}: "
                f"expected {row_width}, got {array.shape[1]}"
            )

    test_array = np.load(test_file, mmap_mode="r")
    if test_array.ndim != 2 or test_array.shape[1] != row_width:
        raise RuntimeError(
            f"invalid test array shape in {test_file}: {test_array.shape}; "
            f"expected (*, {row_width})"
        )

    return {
        "train_rows": train_rows,
        "test_rows": test_array.shape[0],
        "row_width": row_width,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-client federated .npy files only; do not prepare "
            "virtual groups or start model training."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="fashionmnist",
        help="fashionmnist (or fmnist), cifar10, or cifar100",
    )
    parser.add_argument("--num_of_clients", type=int, default=200)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="inferred from the dataset when omitted (CIFAR-100 uses 20)",
    )
    parser.add_argument("--dirichlet", type=float, default=0.05)
    parser.add_argument("--pathe", action="store_true")
    parser.add_argument("--slice", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="global seed; the existing Dirichlet splitter itself uses seed 2025",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        dataset_name = canonical_dataset_name(args.dataset_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.num_of_clients <= 0:
        raise SystemExit("--num_of_clients must be positive")
    if not args.pathe and args.dirichlet <= 0:
        raise SystemExit("--dirichlet must be positive")

    num_classes = args.num_classes or DEFAULT_NUM_CLASSES[dataset_name]
    output_dir = federated_data_path(
        dataset_name,
        args.num_of_clients,
        args.dirichlet,
        args.pathe,
        args.slice,
    )

    # A complete split needs no raw-data reload.  This also makes the command
    # a quick integrity check when it is run more than once.
    try:
        summary = inspect_split(output_dir, args.num_of_clients)
    except (FileNotFoundError, OSError, RuntimeError):
        summary = None

    if summary is None:
        options = {
            "dataset_name": dataset_name,
            "num_of_clients": args.num_of_clients,
            "num_classes": num_classes,
            "dirichlet": args.dirichlet,
            "pathe": args.pathe,
            "slice": args.slice,
            "seed": args.seed,
        }

        # GetDataSet writes paths relative to the project root.
        os.chdir(PROJECT_ROOT)
        np.random.seed(args.seed)
        try:
            from src.getdata import GetDataSet

            GetDataSet(options)
        except FileNotFoundError as exc:
            hint = RAW_DATA_HINTS[dataset_name]
            raise SystemExit(
                f"raw {dataset_name} data is incomplete. Expected {hint}\n{exc}"
            ) from exc

        try:
            summary = inspect_split(output_dir, args.num_of_clients)
        except (OSError, RuntimeError) as exc:
            raise SystemExit(f"federated-data generation failed: {exc}") from exc
        status = "Generated and verified"
    else:
        status = "Already complete; verified"

    print(f"{status}: {output_dir}")
    print(
        f"clients={args.num_of_clients}, train_samples={summary['train_rows']}, "
        f"test_samples={summary['test_rows']}, row_width={summary['row_width']}"
    )
    print("Next: run scripts/prepare_private_virtual_groups.py with matching data arguments.")


if __name__ == "__main__":
    main()
