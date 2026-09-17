#!/usr/bin/env python3
# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Emulate the one-time private Gram protocol and save virtual groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_paths import federated_data_path, validate_federated_data
from src.virtualset.private_gram import (
    load_client_label_histograms,
    make_groups_from_gram,
    validate_bfv_range,
)
from src.virtualset.private_gram_gpu import compute_private_gram_heongpu
from src.virtualset.set_maker import group_cache_path, save_prepared_groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare private-Gram virtual groups with BFV."
    )
    parser.add_argument(
        "--privacy_backend",
        choices=("heongpu_gpu",),
        default="heongpu_gpu",
    )
    parser.add_argument("--dataset_name", default="fashionmnist")
    parser.add_argument("--num_of_clients", type=int, default=200)
    parser.add_argument("--num_of_edges", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dirichlet", type=float, default=0.01)
    parser.add_argument(
        "--data_partition",
        choices=("dirichlet", "mobcorr_strict"),
        default="dirichlet",
    )
    parser.add_argument("--pathe", action="store_true")
    parser.add_argument("--slice", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mobility_seed", type=int, default=2025)
    parser.add_argument(
        "--poly_modulus_degree",
        type=int,
        default=8192,
        help="HEonGPU BFV polynomial degree",
    )
    parser.add_argument(
        "--plain_modulus",
        type=int,
        default=33832961,
        help=(
            "BFV batching prime; a backend-compatible safe default is selected "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--gpu_executable",
        default=None,
        help="path to private_gram_gpu; defaults to build/heongpu/private_gram_gpu",
    )
    parser.add_argument("--cuda_streams", type=int, default=16)
    parser.add_argument("--gpu_batch_size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    options = vars(args).copy()
    options["group_distribution"] = "private_gram"
    options["data_path"] = federated_data_path(options)
    validate_federated_data(options["data_path"], args.num_of_clients)
    cache_path = group_cache_path(options)
    if os.path.isfile(cache_path) and not args.force:
        print("Already complete; use --force to regenerate: {}".format(cache_path))
        return

    # Simulation harness only: each row below represents a histogram that is
    # computed locally by one client in the deployed protocol.  It is never
    # passed to the evaluator or written to the private-group artifact.
    histograms = load_client_label_histograms(options).astype(np.int64)
    sample_counts = histograms.sum(axis=1).astype(np.int64)
    validate_bfv_range(sample_counts, args.plain_modulus)

    gram, native_metadata = compute_private_gram_heongpu(
        histograms,
        executable=args.gpu_executable,
        poly_modulus_degree=args.poly_modulus_degree,
        plain_modulus=args.plain_modulus,
        cuda_streams=args.cuda_streams,
        batch_size=args.gpu_batch_size,
    )
    keygen_seconds = native_metadata["keygen_seconds"]
    encryption_seconds = native_metadata[
        "client_encryption_and_serialization_seconds"
    ]
    gram_seconds = native_metadata[
        "gram_evaluation_and_decryption_seconds"
    ]
    public_context_bytes = native_metadata["evaluator_context_bytes"]
    client_context_bytes = native_metadata["client_context_bytes"]
    encrypted_histogram_bytes = native_metadata[
        "encrypted_histogram_bytes"
    ]
    encrypted_histogram_max_bytes = int(np.ceil(
        encrypted_histogram_bytes / args.num_of_clients
    ))
    gram_entry_count = native_metadata["encrypted_gram_entries"]
    gram_ciphertext_bytes = None

    # This exact equality is available only to the experiment harness and is
    # not part of the deployed protocol.
    if not np.array_equal(gram, histograms @ histograms.T):
        raise AssertionError("decrypted BFV Gram matrix is incorrect")

    start = time.perf_counter()
    groups = make_groups_from_gram(
        gram,
        sample_counts,
        num_groups=args.num_of_edges,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    grouping_seconds = time.perf_counter() - start
    save_prepared_groups(options, groups)

    print("Class order: {}".format(list(range(args.num_classes))))
    group_scores = []
    for group_id in sorted(groups):
        group_histogram = histograms[groups[group_id]].sum(axis=0)
        group_proportions = group_histogram / group_histogram.sum()
        score = float(
            np.square(group_proportions - 1.0 / args.num_classes).sum()
        )
        group_scores.append(score)
        print(
            "G{} clients={} total={} counts={} props={} Q={:.8f}".format(
                group_id + 1,
                len(groups[group_id]),
                int(group_histogram.sum()),
                group_histogram.astype(np.int64).tolist(),
                np.round(group_proportions, 4).tolist(),
                score,
            )
        )

    metadata = {
        "backend": "HEonGPU-BFV",
        "group_distribution": "private_gram",
        "poly_modulus_degree": args.poly_modulus_degree,
        "plain_modulus": args.plain_modulus,
        "num_clients": args.num_of_clients,
        "num_classes": args.num_classes,
        "num_groups": args.num_of_edges,
        "keygen_seconds": keygen_seconds,
        "client_encryption_seconds": encryption_seconds,
        "gram_evaluation_and_decryption_seconds": gram_seconds,
        "grouping_seconds": grouping_seconds,
        "preprocessing_computation_seconds": (
            keygen_seconds
            + encryption_seconds
            + gram_seconds
            + grouping_seconds
        ),
        "mean_group_imbalance": float(np.mean(group_scores)),
        "max_group_imbalance": float(np.max(group_scores)),
        "public_context_bytes": public_context_bytes,
        "client_context_bytes": client_context_bytes,
        "encrypted_histogram_bytes": encrypted_histogram_bytes,
        "encrypted_histogram_max_bytes": encrypted_histogram_max_bytes,
        "encrypted_gram_entries": gram_entry_count,
        "encrypted_gram_stream_bytes": gram_ciphertext_bytes,
        "encrypted_gram_transport": "in-memory role emulation; evaluator-selector communication excluded",
        "released_to_hfl_server": "virtual-group memberships only",
        "threat_model": "honest-but-curious, non-colluding evaluator and selector",
    }
    if native_metadata:
        metadata["native_backend_runtime"] = native_metadata
    metadata_path = cache_path + ".privacy.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print("Saved private-Gram groups to {}".format(cache_path))
    print("Saved privacy/runtime metadata to {}".format(metadata_path))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if native_metadata:
        print(
            "HEonGPU runtime: keygen={:.3f}s, client encrypt+serialize={:.3f}s, "
            "Gram evaluate={:.3f}s, Gram decrypt={:.3f}s, protocol total={:.3f}s".format(
                native_metadata["keygen_seconds"],
                native_metadata[
                    "client_encryption_and_serialization_seconds"
                ],
                native_metadata["gram_evaluation_seconds"],
                native_metadata["gram_decryption_seconds"],
                native_metadata["total_gpu_protocol_seconds"],
            )
        )
    print(
        "Balance: mean Q={:.8f}, max Q={:.8f}".format(
            float(np.mean(group_scores)),
            float(np.max(group_scores)),
        )
    )


if __name__ == "__main__":
    main()
