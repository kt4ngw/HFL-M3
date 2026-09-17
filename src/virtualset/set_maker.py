# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import os
import pickle
import random

import numpy as np

from src.data_paths import partition_tag


def group_cache_path(options, filename='set.pkl'):
    """Return the canonical cache path without accessing client statistics."""
    split_name = (
        'pathology'
        if options['pathe']
        else str(options.get('data_partition', 'dirichlet'))
    )
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    cache_dir = os.path.join(
        project_root,
        'artifacts',
        'virtual_groups',
        'groups',
        split_name,
    )
    split = partition_tag(options)
    source = options.get('group_distribution', 'private_gram')
    group_seed = int(options.get('group_seed', options['seed']))
    suffix = (
        f"dn_{options['dataset_name']}"
        f"_noc_{options['num_of_clients']}"
        f"_noe_{options['num_of_edges']}"
        f"_{split}_src_{source}"
        f"_seed_{group_seed}"
    )
    return os.path.join(cache_dir, suffix + '_' + filename)


def load_prepared_groups(options):
    path = group_cache_path(options)
    try:
        with open(path, 'rb') as file:
            groups = pickle.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Virtual groups have not been prepared. Expected cache: " + path
        ) from error
    return groups, path


def save_prepared_groups(options, groups):
    path = group_cache_path(options)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(groups, file)
    return path


def balance(group_distribution, client_distribution):
    combined = np.asarray(group_distribution, dtype=float) + np.asarray(
        client_distribution, dtype=float
    )
    total = float(combined.sum())
    if total <= 0:
        return float("inf")
    target = 1.0 / len(combined)
    return float(np.square(combined / total - target).sum())


# Keep the old misspelled public name for compatibility with older scripts.
def banlance(D_distribution, clients_data_distribution):
    return balance(D_distribution, clients_data_distribution)


class Set_Maker:
    def __init__(
        self,
        clients_data_distribution,
        options,
        generate_if_missing=True,
        force=False,
    ):
        distribution = np.asarray(clients_data_distribution, dtype=float)
        expected_shape = (
            int(options['num_of_clients']),
            int(options.get('num_classes', 10)),
        )
        if distribution.shape != expected_shape:
            raise ValueError(
                f"client distribution shape {distribution.shape}, expected {expected_shape}"
            )

        self.clients_label_distribution = distribution
        self.options = options
        self.generate_if_missing = bool(generate_if_missing)
        self.force = bool(force)
        canonical_cache = group_cache_path(options)
        self.script_dir = os.path.dirname(canonical_cache)
        os.makedirs(self.script_dir, exist_ok=True)
        self.suffix = os.path.basename(canonical_cache)[:-len('_set.pkl')]
        self.load_or_generate_set()

    def get_best_set(self):
        num_clients = int(self.options['num_of_clients'])
        num_groups = int(self.options['num_of_edges'])
        if not 0 < num_groups <= num_clients:
            raise ValueError("num_of_edges must be between 1 and num_of_clients")

        rng = random.Random(
            int(self.options.get('group_seed', self.options['seed']))
        )
        clients_index = list(range(num_clients))
        base_size, remainder = divmod(num_clients, num_groups)
        target_sizes = [
            base_size + (group_id < remainder) for group_id in range(num_groups)
        ]

        groups = {}
        for group_id, target_size in enumerate(target_sizes):
            first = rng.choice(clients_index)
            group = [first]
            group_distribution = self.clients_label_distribution[first].copy()
            clients_index.remove(first)

            while len(group) < target_size:
                best_client = min(
                    clients_index,
                    key=lambda client_id: balance(
                        group_distribution,
                        self.clients_label_distribution[client_id],
                    ),
                )
                group.append(best_client)
                group_distribution += self.clients_label_distribution[best_client]
                clients_index.remove(best_client)
            groups[group_id] = group

        return groups

    def load_or_generate_set(self, filename='set.pkl'):
        self.cache_path = os.path.join(
            self.script_dir, self.suffix + '_' + filename
        )
        if not self.force:
            try:
                self.load_set(self.cache_path)
                print(f"Virtual groups loaded from {self.cache_path}")
                return
            except FileNotFoundError:
                pass

        if not self.generate_if_missing:
            raise FileNotFoundError(
                "Virtual groups have not been prepared. Run "
                "`python scripts/prepare_private_virtual_groups.py` with the "
                "same configuration before training. Expected cache: "
                + self.cache_path
            )
        self.G = self.get_best_set()
        self.save_set(self.cache_path)
        print(f"New virtual groups generated and saved to {self.cache_path}")

    def save_set(self, filename='set.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.G, file)

    def load_set(self, filename='set.pkl'):
        with open(filename, 'rb') as file:
            self.G = pickle.load(file)
