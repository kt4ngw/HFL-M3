import numpy as np
import torch
import time

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from src.utils.metrics import Metrics
criterion = F.cross_entropy
from src.utils.torch_utils import *

import random

from src.fed_cloud.base_cloud import BaseCloud
from src.optimizers.gd import GD
from src.models.model import choose_model
import logging  # 引入 logging 模块
import numpy as np
from collections import deque
from virtualset.set_maker import Set_Maker

class Proposed(BaseCloud):

    def __init__(self, options):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) 
        super(Proposed, self).__init__(options, model, self.optimizer, )
        self.best_clients_to_edge_map = {} 
        self.clients_label_distribution = self.get_clients_label_distribution()
        self.Set = Set_Maker(self.clients_label_distribution, self.options)
        self.best_G = self.Set.G
        # print(self.clients_label_distribution)
        print("self.best_G", self.best_G)
        # === 新增：按组汇总分布（计数与比例） ===
        num_classes = self.options.get('num_classes', 10)  # CIFAR-10 默认 10
        self.group_label_counts = self._aggregate_group_label_counts(
            best_G=self.best_G,
            clients_label_distribution=self.clients_label_distribution,
            num_classes=num_classes
        )
        self.group_label_props = self._normalize_group_label_counts(
            group_label_counts=self.group_label_counts
        )

        # 可选：打印核查
        # print("self.best_G", self.best_G)
        for gid, cnt in self.group_label_counts.items():
            tot = cnt.sum()
            print(f"[Group {gid}] total={int(tot)} counts={cnt.tolist()} props={self.group_label_props[gid].round(4).tolist()}")
        self.total_hops = 0 
    # ===== 新增：组内计数汇总 =====
    def _aggregate_group_label_counts(self, best_G, clients_label_distribution, num_classes: int):
        """
        返回 {group_id: np.ndarray(num_classes,)}，为组内客户端标签计数逐元素相加。
        """
        # 转为 np 数组便于 vectorize
        # shape: (num_clients, num_classes)
        client_mat = np.asarray(clients_label_distribution, dtype=np.int64)
        if client_mat.ndim != 2 or client_mat.shape[1] != num_classes:
            raise ValueError(f"clients_label_distribution 形状异常：{client_mat.shape}, 期望 (*, {num_classes})")

        group_counts = {}
        print(best_G)
        for gid, cid_list in best_G.items():
            if not cid_list:
                group_counts[gid] = np.zeros((num_classes,), dtype=np.int64)
                continue
            # 过滤越界 id
            valid = [cid for cid in cid_list if 0 <= cid < client_mat.shape[0]]
            if len(valid) == 0:
                group_counts[gid] = np.zeros((num_classes,), dtype=np.int64)
                continue
            # 汇总（行相加）
            cnt = client_mat[valid].sum(axis=0)
            group_counts[gid] = cnt
        return group_counts

    # ===== 新增：计数 -> 比例 =====
    def _normalize_group_label_counts(self, group_label_counts: dict):
        """
        对每个组的计数向量按总和归一化，返回 {group_id: np.ndarray(num_classes,)}（float64）。
        若总和为 0，则返回全 0 向量。
        """
        group_props = {}
        for gid, cnt in group_label_counts.items():
            total = cnt.sum()
            if total > 0:
                group_props[gid] = cnt.astype(np.float64) / float(total)
            else:
                group_props[gid] = np.zeros_like(cnt, dtype=np.float64)
        return group_props

    def get_clients_label_distribution(self, ):
        clients_data_distribution = []
        
        for client_id in range(self.options['num_of_clients']):
            train_dir = f"{self.options['data_path']}/client_{client_id + 1}"
            train_data = np.load(f"{train_dir}/train_data.npy")
            labels = train_data[:, -1].astype(int)
            unique_labels, counts = np.unique(labels, return_counts=True)  # 统计类别
            client_distribution = np.zeros(self.options['num_classes'], dtype=int)  # 生成一个全零数组，长度为类别数
            client_distribution[unique_labels] = counts  # 仅填充出现的类别
            
            clients_data_distribution.append(client_distribution.tolist())
        return clients_data_distribution
    
     

    def train(self):
        # self.logger.info("=== 开始训练 ===")
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_e_fraction * self.edges_num)))
        
        for round_i in range(self.num_round):
            print('-' * 51 + "\n")
            self.global_test_latest_model_on_testdata(round_i)
            edge_test = {}
            for round_edge in range(self.options['edge_epoch']):
                self.assign_clients_to_edges(round_i * self.options['edge_epoch'] + round_edge)
                # print("self.client_to_edge_map", self.client_to_edge_map)
                selected_edges = self.select_edges()
                self.edge_round_last(round_i, selected_edges)
                mapping, extra, missing, relocation_plan, logical_owner = self.DMM(round_i * self.options['edge_epoch'] + round_edge, selected_edges)
                # 计算每个边的 组合分布
                
                round_hops = sum(max(0, len(item.get('path', [])) - 1) for item in relocation_plan)
                self.total_hops = getattr(self, 'total_hops', 0) + round_hops
                # print("self.total_hops", self.total_hops)
                latency_cost = self.cost.get_latency_sum(self.clients, round_i * self.options['edge_epoch'] + round_edge, self.system_params)
                # print("latency_cost", latency_cost)
                self.metrics.update_costs(round_i, latency_cost, self.options['num_of_clients'] * 2 + self.options['num_of_edges'] * 2 /  self.options['edge_epoch'] + round_hops)         
                # if self.options['dataset_name'] ==  'cifar10':
                #     self.optimizer.adjust_learning_rate(round_i * self.options['edge_epoch'] + round_edge)
                # else:
                # self.optimizer.soft_decay_learning_rate()
                if self.options['pathe'] == True:
                    self.optimizer.soft_decay_learning_rate2()
                else:
                    self.optimizer.soft_decay_learning_rate(self.options['dataset_name'], self.options['dirichlet'])    

            # for _ in range(10):
            #     edge_test[_] = self.edge_test_latest_model_on_testdata(round_i, _)
            # print(f"Edge Test Results: {edge_test}, average: {np.mean(list(edge_test.values()))}")
            
            self.cloud_latest_global_model = self.aggregate_parameters(self.edge_latest_model_set)
            for e in range(self.edges_num):
                self.edge_latest_model_set[e] = (0, copy.deepcopy(self.cloud_latest_global_model))
            # print("after cloud's cloud_latest_global_model_{}".format(self.cloud_latest_global_model))
        self.global_test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    def edge_round(self, round_i, selected_edges):

        edge_model_paras_set = {}
        for e in selected_edges:
            edge_data_num = 0
            clients_in_edge = self.client_to_edge_map[e]

            client_model_paras_set = []
            for client_id in clients_in_edge:
                client = self.clients[client_id]
                client.set_flat_model_params(self.edge_latest_model_set[e][1])  
                local_model_paras, _ = client.local_train()
                client_model_paras_set.append(local_model_paras)
                edge_data_num += local_model_paras[0]
            edge_model = self.aggregate_parameters(client_model_paras_set)
            self.edge_latest_model_set[e] = (edge_data_num, copy.deepcopy(edge_model))


    def DMM(self, round_i, selected_edges, alpha=0.0, lambda_switch=0.0, prev_mapping=None):
        # 建立虚拟映射
        self.set_edge_graph_from_adj_matrix(self.adj_for_edges)

        # print("self.best_G", self.best_G) 
        # self.best_G 这个是虚拟映射组 {0: [client_id, ...], 1: [client_id, ...], ...}

        # 现在的真实组是怎么样的。
        # self.client_to_edge_map  这个是真实映射 {edge_id: [client_id, ...], ...}
        
        # 怎么给上面两个建立起联系呢？ 虚拟组：真实组 需要一个算法
        
        # 邻接矩阵
        # self.adj_for_edges
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        edges = list(selected_edges)
        virt_ids = list(self.best_G.keys())
        E = len(edges)
        assert E == len(virt_ids), "ES 数与虚拟组数需一致（或先做子集/合并）。"
        # --- 1) 集合准备 ---
        real_sets = {e: set(self.client_to_edge_map.get(e, [])) for e in edges}
        virt_sets = {v: set(self.best_G.get(v, [])) for v in virt_ids}
        # 客户端 -> 物理接入 ES（本轮）
        client_phys = {}
        for e, clist in real_sets.items():
            for c in clist:
                client_phys[c] = e

        if not hasattr(self, "es_dist"):
            raise RuntimeError("Call set_edge_graph_from_adj_matrix(...) before DAR/DMM")

        # --- 2) 构造路径代价矩阵 C_path (E x E) ---
        # C_path[i,j] = sum_{c in virt_j} d( phys(c), edges[i] )
        C_path = np.zeros((E, E), dtype=float)
        for i, e in enumerate(edges):              # 候选负责的 ES
            for j, v in enumerate(virt_ids):       # 虚拟集合
                # 仅统计本轮实际接入的客户端；未接入者不产生路由代价
                cost = 0.0
                for c in virt_sets[v]:
                    src = client_phys.get(c)       # c 本轮物理接入的 ES
                    if src is None:
                        continue
                    d = self.es_dist[src][e]       # O(1) 查询最短代价（跳数/时延）
                    # 如需对不可达设置惩罚：
                    if d == float("inf"):
                        d = 1e6
                    cost += d
                C_path[i, j] = cost

            # --- 4) 匈牙利算法（最小代价指派） ---
            row_ind, col_ind = linear_sum_assignment(C_path)
            mapping = {edges[i]: virt_ids[j] for i, j in zip(row_ind, col_ind)}

            # --- 5) 计算 extra/missing ---
            extra, missing = {}, {}
            for e in edges:
                v = mapping[e]
                extra[e]   = real_sets[e] - virt_sets[v]
                missing[e] = virt_sets[v] - real_sets[e]

            # --- 6) 生成局部修正：仅对 extra 中的 MC 做逻辑路由 ---
            virt_to_edge = {v: e for e, v in mapping.items()}
            relocation_plan = []

            def _shortest_path(u, v):
                if hasattr(self, "shortest_es_path"):
                    return self.shortest_es_path(u, v)
                # fallback（无拓扑时仅返回端点）
                return [u, v] if u != v else [u]

            # MC 的虚拟归属：c 属于哪个虚拟组 v
            home_v = {}
            for v in virt_ids:
                for c in virt_sets[v]:
                    home_v[c] = v

            for e in edges:
                for c in extra[e]:
                    v_tgt = home_v.get(c, None)
                    if v_tgt is None:
                        continue
                    to_edge = virt_to_edge[v_tgt]
                    path = _shortest_path(e, to_edge)
                    relocation_plan.append({
                        "client": c, "from": e, "to": to_edge,
                        "path": path, "round": round_i
                    })

            # --- 7) 本轮逻辑归属表：client -> owner ES（用于聚合时查找） ---
            logical_owner = {}
            for v in virt_ids:
                owner = virt_to_edge[v]
                for c in virt_sets[v]:
                    logical_owner[c] = owner

        return mapping, extra, missing, relocation_plan, logical_owner
        
        # pass

    def edge_round_last(self, round_i, selected_edges):
        # ② DAR 算法设计
        # self.logger.info("=== 边缘训练轮次 {} ===".format(round_i))
        all_clients_model_paras_set = []
        # print("self.client_to_edge_map", self.client_to_edge_map)
        edge_model_paras_set = {}
        for e in selected_edges:
            edge_data_num = 0
            clients_in_edge = self.client_to_edge_map[e]
            # self.logger.info("边 {} 负责的客户端: {}".format(e, clients_in_edge))
            # self.best_G 
            # client_model_paras_set = []
            for client_id in clients_in_edge:
                client = self.clients[client_id]
                client.set_flat_model_params(self.edge_latest_model_set[e][1])  
                local_model_paras, _ = client.local_train()
                all_clients_model_paras_set.append((client_id, local_model_paras))
                edge_data_num += local_model_paras[0]
            # 
        self.aggregate_by_best_group(self.best_G, all_clients_model_paras_set)

            # edge_model = self.aggregate_parameters(client_model_paras_set)
            # self.edge_latest_model_set[e] = (edge_data_num, copy.deepcopy(edge_model))

    def aggregate_by_best_group(self, best_G, client_updates):

        cid2upd = {cid: (n, w) for cid, (n, w) in client_updates}
        # print("cid2upd", cid2upd)
        for e, cid_list in best_G.items():
            client_model_paras_set = []
            edge_data_num = 0   
            for cid in cid_list:
                client_model_paras_set.append(cid2upd[cid])
                edge_data_num += cid2upd[cid][0]    
            edge_model = self.aggregate_parameters(client_model_paras_set)
            self.edge_latest_model_set[e] = (edge_data_num, copy.deepcopy(edge_model))

# ── 在 Proposed 类里添加：ES 拓扑设置、最短路预计算与查询 ──



    def set_edge_graph_from_adj_matrix(self, A):
        """
        A: np.ndarray 或 list[list]，0/1（或 >0 表示连边），默认无权、无向。
        结果：
        self.edge_graph: {u: [v,...]}
        self.es_dist[u][v]: 最短跳数
        self._es_next[u][v]: u->v 的下一跳
        """
        A = np.asarray(A)
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "adjacency matrix must be square"
        n = A.shape[0]

        # 若不是严格对称，可在此对称化（按需要开启）
        # A = np.maximum(A, A.T)

        # 邻接表（无权）
        edge_graph = {i: [j for j in range(n) if A[i, j] != 0 and i != j] for i in range(n)}
        self.edge_graph = edge_graph

        # 预计算：全点对 BFS
        self.es_dist = {u: {v: float('inf') for v in range(n)} for u in range(n)}
        self._es_next = {u: {} for u in range(n)}

        for s in range(n):
            prev = {s: None}
            dist = {s: 0}
            q = deque([s])
            while q:
                x = q.popleft()
                for y in edge_graph.get(x, []):
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        prev[y] = x
                        q.append(y)
            # 写回距离与下一跳
            for t, d in dist.items():
                self.es_dist[s][t] = float(d)
                if s == t:
                    self._es_next[s][t] = s
                else:
                    cur = t
                    while prev[cur] is not None and prev[cur] != s:
                        cur = prev[cur]
                    self._es_next[s][t] = cur

    def shortest_es_path(self, u: int, v: int):
        """按预计算的下一跳表返回最短路径 [u,...,v]；若不可达抛错。"""
        u = int(u); v = int(v)
        if u == v:
            return [u]
        if not hasattr(self, "_es_next") or u not in self._es_next or v not in self._es_next[u]:
            raise RuntimeError("Paths not precomputed. Call set_edge_graph_from_adj_matrix(...) first.")
        if self.es_dist[u][v] == float('inf'):
            raise ValueError(f"ES graph disconnected: no path {u}->{v}")
        path = [u]
        cur = u
        visited = {u}
        # 最多 n 步，防御意外环
        for _ in range(len(self._es_next) + 5):
            cur = self._es_next[cur][v]
            if cur in visited:
                break
            path.append(cur)
            visited.add(cur)
            if cur == v:
                return path
        # 理论上不会走到这；若走到这，可退回 BFS 重建
        raise RuntimeError("Failed to reconstruct path; check adjacency matrix or next-table.")
