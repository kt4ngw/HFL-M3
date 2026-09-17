# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import numpy as np


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    np.random.seed(2025)
    # print(train_labels)
    n_classes = train_labels.max() + 1
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # print(class_idcs)
    # print("class_idcs", class_idcs)
    # print("n_classes", n_classes)
    client_idcs = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        random_class = np.random.choice(n_classes)
        random_sample = np.random.choice(class_idcs[random_class])
        client_idcs[i].append(random_sample)
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += idcs.tolist()

    client_idcs = [np.array(idcs) for idcs in client_idcs]
    result = []
    for i, idcs in enumerate(client_idcs):
        result.append(idcs.tolist())
    return client_idcs, result


import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque

def shard_split_noniid(labels: np.ndarray,
                       num_clients: int,
                        shards_per_client: int = 1,
                       num_classes: int = 10,
                       ensure_diff_classes_per_client: bool = True,
                       seed=2025) -> Dict[int, np.ndarray]:
    """Pathological split: cut every class into equal shards and hand each
    client ``shards_per_client`` shards, preferring distinct classes.

    Classes may have different sample counts. Returns client_id -> index array.
    """

    rng = np.random.default_rng(seed)

    total_shards = num_clients * shards_per_client
    assert total_shards % num_classes == 0, \
        f"total shards {total_shards} must be divisible by num_classes {num_classes}; adjust shards_per_client or num_clients"
    shards_per_class = total_shards // num_classes

    class_indices: List[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            raise ValueError(f"class {c} has no samples to shard")
        rng.shuffle(idx)
        class_indices.append(idx)

    shards: List[Tuple[int, np.ndarray]] = []  # (class_id, indices_of_shard)
    for c in range(num_classes):
        for shard in np.array_split(class_indices[c], shards_per_class):
            if shard.size == 0:
                raise ValueError(
                    f"class {c} has fewer samples than shards_per_class={shards_per_class}"
                )
            shards.append((c, shard))

    rng.shuffle(shards)
    clients: Dict[int, List[np.ndarray]] = {cid: [] for cid in range(num_clients)}

    pool_by_class = defaultdict(deque)
    for c, sl in shards:
        pool_by_class[c].append(sl)

    def pop_shard_from_class(cls: int):
        return pool_by_class[cls].popleft() if len(pool_by_class[cls]) > 0 else None

    for cid in range(num_clients):
        taken_classes = set()
        for k in range(shards_per_client):
            chosen = None
            if ensure_diff_classes_per_client:
                candidate_classes = [c for c in range(num_classes)
                                     if c not in taken_classes and len(pool_by_class[c]) > 0]
                if not candidate_classes:
                    candidate_classes = [c for c in range(num_classes) if len(pool_by_class[c]) > 0]
                if candidate_classes:
                    candidate_classes.sort(key=lambda x: -len(pool_by_class[x]))
                    chosen = candidate_classes[0]
            else:
                avail = [c for c in range(num_classes) if len(pool_by_class[c]) > 0]
                if avail:
                    avail.sort(key=lambda x: -len(pool_by_class[x]))
                    chosen = avail[0]

            if chosen is None:
                raise RuntimeError("shards exhausted before all clients were served; check the split parameters")

            sl = pop_shard_from_class(chosen)
            if sl is None:
                raise RuntimeError("internal error: selected an empty class")
            clients[cid].append(sl)
            taken_classes.add(chosen)

    client_indices = {cid: np.concatenate(shards_list, axis=0) for cid, shards_list in clients.items()}
    return client_indices
