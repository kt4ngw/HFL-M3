# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Gram-based grouping and range checks shared with the HEonGPU adapter."""

import os
import random

import numpy as np


def validate_bfv_range(sample_counts, plain_modulus):
    """Ensure every possible histogram dot product fits without wraparound."""
    sample_counts = np.asarray(sample_counts, dtype=np.int64)
    if sample_counts.ndim != 1 or sample_counts.size == 0:
        raise ValueError("sample_counts must be a non-empty vector")
    if np.any(sample_counts < 0):
        raise ValueError("sample_counts cannot contain negative values")
    max_dot_upper_bound = int(sample_counts.max()) ** 2
    if max_dot_upper_bound >= int(plain_modulus):
        raise ValueError(
            "plain_modulus={} is too small; it must exceed the public dot-product "
            "upper bound {}".format(plain_modulus, max_dot_upper_bound)
        )


def gram_group_cost(group, gram, sample_counts, num_classes):
    """Exact squared distance between a group's aggregate and the uniform target."""
    group = np.asarray(group, dtype=np.int64)
    total_samples = float(np.asarray(sample_counts)[group].sum())
    if total_samples <= 0:
        return float("inf")
    numerator = float(gram[np.ix_(group, group)].sum())
    return numerator / (total_samples * total_samples) - 1.0 / int(num_classes)


def make_groups_from_gram(gram, sample_counts, num_groups, num_classes, seed):
    """Reproduce the existing fixed-size greedy VSF using only Gram and counts."""
    gram = np.asarray(gram, dtype=np.int64)
    sample_counts = np.asarray(sample_counts, dtype=np.int64)
    num_clients = len(sample_counts)
    if gram.shape != (num_clients, num_clients):
        raise ValueError("gram and sample_counts dimensions do not match")
    if not 0 < int(num_groups) <= num_clients:
        raise ValueError("num_groups must be between one and num_clients")

    rng = random.Random(int(seed))
    remaining = list(range(num_clients))
    base_size, remainder = divmod(num_clients, int(num_groups))
    group_sizes = [
        base_size + (group_id < remainder)
        for group_id in range(int(num_groups))
    ]
    groups = {}
    for group_id, group_size in enumerate(group_sizes):
        first = rng.choice(remaining)
        group = [first]
        remaining.remove(first)
        while len(group) < group_size:
            best_client = min(
                remaining,
                key=lambda client_id: gram_group_cost(
                    group + [client_id],
                    gram,
                    sample_counts,
                    num_classes,
                ),
            )
            group.append(best_client)
            remaining.remove(best_client)
        groups[group_id] = group
    return groups


def load_client_label_histograms(options):
    """Read each client's local label histogram from the prepared partition.

    In the private protocol each client computes this locally and only ever
    uploads its BFV encryption; the plaintext is loaded here solely because the
    experiment harness emulates every client role in one process.
    """
    distributions = []
    for client_id in range(int(options['num_of_clients'])):
        train_dir = os.path.join(options['data_path'], f"client_{client_id + 1}")
        train_data = np.load(os.path.join(train_dir, 'train_data.npy'), mmap_mode='r')
        labels = np.asarray(train_data[:, -1], dtype=int)
        counts = np.bincount(labels, minlength=int(options.get('num_classes', 10)))
        distributions.append(counts)
    return np.asarray(distributions, dtype=float)
