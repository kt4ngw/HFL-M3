import numpy as np


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    np.random.seed(2025)
    # print(train_labels)
    n_classes = train_labels.max() + 1
    # 确保 alpha 大于 0
    if alpha <= 0:
        raise ValueError(f"❌ alpha 必须大于 0，当前值: {alpha}")
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # print(class_idcs)
    # print("class_idcs", class_idcs)
    # print("n_classes", n_classes)
    client_idcs = [[] for _ in range(n_clients)]
    # 首先，为每个客户端分配一个样本
    for i in range(n_clients):
        random_class = np.random.choice(n_classes)
        random_sample = np.random.choice(class_idcs[random_class])
        client_idcs[i].append(random_sample)
    # 分配余下的样本给每个客户端
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += idcs.tolist()

    client_idcs = [np.array(idcs) for idcs in client_idcs]
    # 假设 client_idcs 是一个包含多个子列表的列表
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
    """
    根据 num_clients 与 shards_per_client 自动计算总切片数并按类均匀切片。
    适用于每类样本数大致相等（如 CIFAR-10：每类 5000）。

    返回: client_id -> 样本索引数组
    """

    rng = np.random.default_rng(seed)

    # 0) 由需求反推每类切片数
    total_shards = num_clients * shards_per_client
    assert total_shards % num_classes == 0, \
        f"总切片数 {total_shards} 不能被类别数 {num_classes} 整除，请调整 shards_per_client 或 num_clients。"
    shards_per_class = total_shards // num_classes  # 例如 50*2/10=10

    # 1) 按类别收集索引并打乱；检查每类能否被均匀切分
    class_indices: List[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            raise ValueError(f"类别 {c} 没有样本，无法切片。")
        rng.shuffle(idx)
        # 要求能整分（CIFAR-10 满足：5000 % 10 == 0）
        assert len(idx) % shards_per_class == 0, \
            f"类别 {c} 的样本数 {len(idx)} 不能被 shards_per_class={shards_per_class} 整除。"
        class_indices.append(idx)

    # 2) 生成切片：每类切成 shards_per_class 份（等大小）
    shards: List[Tuple[int, np.ndarray]] = []  # (class_id, indices_of_shard)
    for c in range(num_classes):
        n_per_shard = len(class_indices[c]) // shards_per_class
        for s in range(shards_per_class):
            start = s * n_per_shard
            end   = (s + 1) * n_per_shard
            shards.append((c, class_indices[c][start:end]))

    # 3) 分配：每客户端拿 shards_per_client 片，尽量不同类别
    rng.shuffle(shards)
    clients: Dict[int, List[np.ndarray]] = {cid: [] for cid in range(num_clients)}

    # 建类到队列的池子，方便按类取片
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
                # 优先挑“尚未拿过”的类别，且库存尽量多的
                candidate_classes = [c for c in range(num_classes)
                                     if c not in taken_classes and len(pool_by_class[c]) > 0]
                if not candidate_classes:
                    # 兜底：若实在没有不同类，就随便拿一个有货的
                    candidate_classes = [c for c in range(num_classes) if len(pool_by_class[c]) > 0]
                if candidate_classes:
                    # 按库存多少排序，优先取库存多的
                    candidate_classes.sort(key=lambda x: -len(pool_by_class[x]))
                    chosen = candidate_classes[0]
            else:
                # 不要求不同类：直接挑库存最多的类别
                avail = [c for c in range(num_classes) if len(pool_by_class[c]) > 0]
                if avail:
                    avail.sort(key=lambda x: -len(pool_by_class[x]))
                    chosen = avail[0]

            if chosen is None:
                raise RuntimeError("切片已耗尽，无法继续分配（参数可能不匹配）。")

            sl = pop_shard_from_class(chosen)
            if sl is None:
                raise RuntimeError("内部错误：选择了空类别。")
            clients[cid].append(sl)
            taken_classes.add(chosen)

    # 4) 合并为每客户端的索引数组
    client_indices = {cid: np.concatenate(shards_list, axis=0) for cid, shards_list in clients.items()}
    return client_indices
