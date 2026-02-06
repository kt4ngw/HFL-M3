import numpy as np
import torch
import time
from src.fed_client.base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from src.utils.metrics import Metrics
criterion = F.cross_entropy
from src.utils.torch_utils import *
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import random
from colorama import Fore, Style
import logging
from src.cost import Cost

class BaseCloud(object):

    def __init__(self, options, model=None, optimizer=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer

        # self.logger = logging.getLogger(self.__class__.__name__)  # 获取类名作为日志名
        self.cloud_latest_global_model = self.get_flat_model_params()
        self.options = options
        self.gpu = options['gpu']
        self.num_round = options['round_num']
        self.per_round_e_fraction = options['e_fraction']
        self.clients = self.setup_clients(self.options['num_of_clients'])
        self.clients_num = len(self.clients)
        self.edges = [_ for _ in range(10)]
        self.edges_num = 10
        self.edge_latest_model_set = [(0, self.get_flat_model_params()) for _ in range(int(self.clients_num / 10))]
        self.client_to_edge_map = {}  # {client_id: edge_id}
        self.name = '_'.join([name, f'cn{int(self.clients_num)}', f'en{self.edges_num}'])
        self.cost = Cost()
        self.metrics = Metrics(options, self.clients, self.name)

        self.system_params = np.load(options['sys_para_path'], allow_pickle=True).item() # [客户端1: [round1, round2, ...] , [], []] of ndarray(round_num,)
            # 初始化时调用
        # self.markov_matrix = self.generate_markov_matrix(len(self.edges))
        self.adj_for_edges = self.build_random_adj(num_edges=len(self.edges))
        print("self.adj_for_edges", self.adj_for_edges)

        for client in self.clients:
            client.P = self.generate_markov_matrix(self.adj_for_edges, stay_prob=options['stay_prob'])

    def build_random_adj(self, num_edges, min_links=3, max_links=5):

        assert 0 <= min_links <= max_links < num_edges, "参数不合法"
        rng = np.random.default_rng(2025)
        adj = np.zeros((num_edges, num_edges), dtype=int)

        # 先随机目标度
        target_deg = rng.integers(min_links, max_links, size=num_edges)

        # 按度需求贪心连边
        nodes = list(range(num_edges))
        for i in nodes:
            while adj[i].sum() < target_deg[i]:
                # 可选邻居：非自己、尚未连
                candidates = [j for j in nodes if j != i and adj[i, j] == 0]
                if not candidates:
                    break
                j = rng.choice(candidates)
                adj[i, j] = adj[j, i] = 1

        # 兜底：避免孤点
        for i in nodes:
            if adj[i].sum() == 0:
                j = rng.choice([x for x in nodes if x != i])
                adj[i, j] = adj[j, i] = 1

        # 对角清零
        np.fill_diagonal(adj, 0)
        return adj

    def assign_clients_to_edges(self, round_i):
        # 将客户端分配到不同的边
        # 第一轮是随机分配
        # 后面的轮数 客户端根据马尔可夫链转移到不同的边
        np.random.seed(self.options['seed'] + round_i)
        num_edges = len(self.edges)
        self.client_to_edge_map = {e: [] for e in range(num_edges)}

        if round_i == 0:
            # 首轮：均衡或随机进场（这里均衡）
            for j, client in enumerate(self.clients):
                e = j % num_edges
                client.current_edge = e
                self.client_to_edge_map[e].append(client.idx)
        else:
            for client in self.clients:
                cur = int(client.current_edge)
                P = client.P
                # 保险起见做一遍行和校验（调试期可保留，稳定后可关掉以提速）
                # assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
                nxt = np.random.choice(num_edges, p=P[cur])
                client.current_edge = int(nxt)
                self.client_to_edge_map[nxt].append(client.idx)
                
    def generate_markov_matrix(self, adj, stay_prob):
        """
        生成统一的马尔可夫转移矩阵，stay_prob 是客户端停留在原边的概率。
        """
        n = adj.shape[0]
        P = np.zeros((n, n), dtype=float)

        stay = (np.full(n, float(stay_prob)) if np.isscalar(stay_prob)
                else np.asarray(stay_prob, dtype=float))
        assert stay.shape == (n,)
        stay = np.clip(stay, 0.0, 1.0)

        for i in range(n):
            nbrs = np.where(adj[i] == 1)[0]
            if len(nbrs) == 0:
                P[i, i] = 1.0
                continue
            P[i, i] = stay[i]
            residual = 1.0 - stay[i]
            P[i, nbrs] = residual / len(nbrs)
        # print("P", P)
        return P

    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] >= 0:
            device = options['gpu']
            torch.cuda.set_device(device)
            # torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def get_flat_model_params(self):
        flat_feature_extractor_params = get_flat_params_from(self.model.feature_extractor)
        flat_classifier_params = get_flat_params_from(self.model.classifier)
        return torch.cat((flat_feature_extractor_params, flat_classifier_params)).detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)
    
    def train(self):
        """The whole training procedure
        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, num_of_cleint):
        all_client = []
        for i in range(num_of_cleint):
            local_client = Client(self.options, i, self.model, self.optimizer)
            all_client.append(local_client)
        return all_client
    
    def aggregate_parameters(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.cloud_latest_global_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        self.simple_average = False 
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

    def edge_test_latest_model_on_testdata(self, round_i, edge_id):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.edge_test(edge_id, use_test_data=True)
        end_time = time.time()

        # if True:
        #     # print('= Test = round: {} / acc: {:.3%} / '
        #     #       'loss: {:.4f} / Time: {:.2f}s'.format(
        #     #        round_i, stats_from_test_data['acc'],
        #     #        stats_from_test_data['loss'], end_time-begin_time))
        #     # print('=' * 102 + "\n")
        #     print(
        #     Fore.YELLOW + '= Test =' + 
        #     Fore.CYAN + f' round: {round_i} / ' + 
        #     Fore.CYAN + f' edge: {edge_id} / ' + 
        #     Fore.GREEN + f'acc: {stats_from_test_data["acc"]:.3%} / ' + 
        #     Fore.RED + f'loss: {stats_from_test_data["loss"]:.4f} / ' + 
        #     Fore.MAGENTA + f'Time: {end_time - begin_time:.2f}s' + 
        #     Style.RESET_ALL  # 颜色重置，避免影响后续输出
        #     )

        #     print(Fore.BLUE + '=' * 102 + "\n" + Style.RESET_ALL)
        return stats_from_test_data["acc"]
    
    def edge_test(self, edge_id, use_test_data=True):
        assert self.cloud_latest_global_model is not None
        self.set_flat_model_params(self.edge_latest_model_set[edge_id][1])
            # 读取 numpy 数据
        test_dir = f"{self.options['data_path']}/test_data"
        test_data = np.load(f"{test_dir}/test_data.npy")
        X_test = test_data[:, :-1]  # 特征
        # print(X_test.shape)
        y_test = test_data[:, -1]   # 标签
        # print(y_test)
        # X_test = X_test.reshape(-1, 3, 32, 32)  
        # X_test = X_test.reshape(-1, 1, 28, 28)      
        testDataLoader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=100, shuffle=False)
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                feature, pred = self.model(X)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)
        
        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        # a = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
        return stats


    def global_test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            # print('= Test = round: {} / acc: {:.3%} / '
            #       'loss: {:.4f} / Time: {:.2f}s'.format(
            #        round_i, stats_from_test_data['acc'],
            #        stats_from_test_data['loss'], end_time-begin_time))
            # print('=' * 102 + "\n")
            print(
            Fore.YELLOW + '= Test =' + 
            Fore.CYAN + f' round: {round_i} / ' + 
            Fore.GREEN + f'acc: {stats_from_test_data["acc"]:.3%} / ' + 
            Fore.RED + f'loss: {stats_from_test_data["loss"]:.4f} / ' + 
            Fore.MAGENTA + f'Time: {end_time - begin_time:.2f}s' + 
            Style.RESET_ALL  # 颜色重置，避免影响后续输出
            )

            print(Fore.BLUE + '=' * 102 + "\n" + Style.RESET_ALL)

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def global_test(self, use_test_data=True):
        assert self.cloud_latest_global_model is not None
        self.set_flat_model_params(self.cloud_latest_global_model)
            # 读取 numpy 数据
        test_dir = f"{self.options['data_path']}/test_data"
        print(test_dir)
        test_data = np.load(f"{test_dir}/test_data.npy")
        print(test_data.shape)
        X_test = test_data[:, :-1]  # 特征
        print(X_test.shape)
        y_test = test_data[:, -1]   # 标签
        # print(y_test)
        # X_test = X_test.reshape(-1, 3, 32, 32)  
        # X_test = X_test.reshape(-1, 1, 28, 28)      
        testDataLoader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=100, shuffle=False)
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                feature, pred = self.model(X)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)
        
        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        # a = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
        return stats

    def select_edges(self):
        nums_edges = int(self.per_round_e_fraction * self.edges_num)
        select_edges = []
        index = np.random.choice(len(self.edges), nums_edges, replace=False)

        for i in index:
            select_edges.append(self.edges[i])
        return select_edges
    

    
    # def assign_clients_to_edges(self, npy_folder_path):
    #     """
    #     将客户端分配给边类（每轮训练时可以重新分配）
    #     假设边类通过 ID 进行索引，客户端 ID 被均匀分配到不同的边类
    #     """

    #     num_edges = len(self.edges)
    #     # 重新初始化映射表，避免旧数据干扰
    #     self.client_to_edge_map = {i: [] for i in range(num_edges)}  
    #     # edge_clients_map = {i: [] for i in range(num_edges)}

    #     # 1. 加载所有客户端特征
    #     client_ids = []
    #     client_features = []
    #     for client in self.clients:
    #         client_id = client.idx
    #         npy_path = os.path.join(npy_folder_path, f"client_{client_id + 1}/train_data.npy")
    #         if not os.path.exists(npy_path):
    #             raise FileNotFoundError(f"找不到 {npy_path}")
    #         data = np.load(npy_path)  # shape: (num_samples, feature_dim + 1)
    #         labels = data[:, -1].astype(int)  # 提取最后一列作为标签
    #         # 标签分布统计（例如标签为 0~9，共10类）
    #         label_count = Counter(labels)
    #         label_vector = np.zeros(10)  # 如果有10个类别
    #         for lbl, cnt in label_count.items():
    #             label_vector[lbl] = cnt  

    #         client_ids.append(client_id)
    #         client_features.append(label_vector)  
    #     client_features = np.array(client_features)
    #     # 2. 聚类
    #     kmeans = KMeans(n_clusters=num_edges, random_state=42)
    #     cluster_labels = kmeans.fit_predict(client_features)
    #     # 构建映射表
    #     for idx, cluster_id in zip(client_ids, cluster_labels):
    #         self.client_to_edge_map[cluster_id].append(idx)
    #     print(self.client_to_edge_map)

    #     # shuffled_clients = random.sample(self.clients, len(self.clients)) 
    #     # count = 0
    #     # for client in shuffled_clients:
    #     #     self.client_to_edge_map[count // 20].append(client.idx)  
    #     #     # edge_clients_map[count // 20].append(client)
    #     #     count += 1
    #     #     if count % 10 == 0:
    #     #         continue
    #     # # 将分配的客户端信息传递给各个边类
    #     # for edge_id, edge in enumerate(self.edges):
    #     #     edge.clients = edge_clients_map[edge_id]
  