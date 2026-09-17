# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import numpy as np
import torch
import time
from src.fed_client.base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.utils.metrics import Metrics
criterion = F.cross_entropy
from src.utils.torch_utils import get_flat_params_from, set_flat_params_to
from colorama import Fore, Style
from src.cost import Cost
from src.mobility import (
    PrecomputedMobility,
    default_slaw_path,
    random_es_adjacency,
)

class BaseCloud(object):

    def __init__(self, options, model=None, optimizer=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer

        self.cloud_latest_global_model = self.get_flat_model_params()
        self.options = options
        self.gpu = options['gpu']
        self.num_round = options['round_num']
        self.per_round_e_fraction = options['e_fraction']
        self.clients = self.setup_clients(self.options['num_of_clients'])
        self.clients_num = len(self.clients)
        self.edges_num = int(options['num_of_edges'])
        self.edges = list(range(self.edges_num))
        self.edge_latest_model_set = [
            (0, self.get_flat_model_params()) for _ in range(self.edges_num)
        ]
        self.mobility_model = str(options.get('mobility_model', 'slaw')).lower()
        self.mobility = None
        self.client_to_edge_map = {}  # {edge_id: [client_id, ...]}
        self._test_data_loader = None
        self.name = '_'.join([name, f'cn{int(self.clients_num)}', f'en{self.edges_num}'])
        self.cost = Cost()
        self.metrics = Metrics(options, self.clients, self.name)

        self.system_params = np.load(
            options['sys_para_path'], allow_pickle=True
        ).item()
        expected_rounds = self.num_round * int(options['edge_epoch'])
        self._validate_system_params(expected_rounds)
        self.adj_for_edges = self.build_random_adj(num_edges=len(self.edges))
        self._validate_connected_graph(self.adj_for_edges)

        if self.mobility_model != 'slaw':
            raise ValueError("Proposed requires mobility_model='slaw'")
        mobility_path = options.get('mobility_file') or default_slaw_path(options)
        self.mobility = PrecomputedMobility(
            path=mobility_path,
            num_clients=self.clients_num,
            num_edges=self.edges_num,
            num_steps=expected_rounds,
        )
        print("Mobility: prepared SLAW associations from {}".format(self.mobility.path))

    def _validate_system_params(self, expected_rounds):
        client_keys = ('cpu_frequency', 'U', 'D')
        required = client_keys + ('V',)
        missing = [key for key in required if key not in self.system_params]
        if missing:
            raise KeyError(
                "system parameter file is missing {}".format(', '.join(missing))
            )

        for key in required:
            if len(self.system_params[key]) < expected_rounds:
                raise ValueError(
                    "{} has {} rounds, expected at least {}".format(
                        key, len(self.system_params[key]), expected_rounds
                    )
                )

        for round_i in range(expected_rounds):
            for key in client_keys:
                values = np.asarray(
                    self.system_params[key][round_i], dtype=float
                )
                if values.shape != (self.clients_num,):
                    raise ValueError(
                        "{}[{}] has shape {}, expected ({},)".format(
                            key, round_i, values.shape, self.clients_num
                        )
                    )
                if not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError(
                        "{}[{}] must contain positive finite values".format(
                            key, round_i
                        )
                    )

            rates = np.asarray(self.system_params['V'][round_i], dtype=float)
            expected_shape = (self.edges_num, self.edges_num)
            if rates.shape != expected_shape:
                raise ValueError(
                    "V[{}] has shape {}, expected {}".format(
                        round_i, rates.shape, expected_shape
                    )
                )
            if not np.all(np.isfinite(rates)) or np.any(rates < 0):
                raise ValueError(
                    "V[{}] must contain finite non-negative values".format(
                        round_i
                    )
                )

    @staticmethod
    def _validate_connected_graph(adj):
        matrix = np.asarray(adj)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("ES adjacency matrix must be square")
        if matrix.shape[0] == 0:
            raise ValueError("ES graph must contain at least one ES")

        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(matrix[node]):
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(visited) != matrix.shape[0]:
            raise ValueError("ES graph must be connected")

    def build_random_adj(self, num_edges, min_links=3, max_links=5):
        return random_es_adjacency(
            num_edges=num_edges,
            min_links=min_links,
            max_links=max_links,
            seed=2025,
        )

    def assign_clients_to_edges(self, round_i):
        """Assign every client to its physical ES for one edge round."""
        num_edges = len(self.edges)
        self.client_to_edge_map = {e: [] for e in range(num_edges)}

        edge_ids = self.mobility.edges_at(round_i)
        for client, edge_id in zip(self.clients, edge_ids):
            edge_id = int(edge_id)
            client.current_edge = edge_id
            self.client_to_edge_map[edge_id].append(client.idx)

    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] >= 0:
            device = options['gpu']
            torch.cuda.set_device(device)
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

    def evaluate_latency(self):
        """Replay client communication only and replace latency_metrics.json.

        Baseline frameworks do not route model updates between ESs in the
        common latency model.  Their latency can therefore be recomputed from
        the saved mobility/system traces without executing local SGD.
        """
        edge_epoch = int(self.options['edge_epoch'])
        traffic_per_edge_round = (
            self.options['num_of_clients'] * 2
            + self.options['num_of_edges'] * 2 / edge_epoch
        )
        total_edge_rounds = self.num_round * edge_epoch
        for system_round in range(total_edge_rounds):
            cloud_round = system_round // edge_epoch
            self.assign_clients_to_edges(system_round)
            latency_cost = float(
                self.cost.get_latency_sum(
                    self.clients,
                    system_round,
                    self.system_params,
                )
            )
            self.metrics.update_costs(
                cloud_round,
                latency_cost,
                traffic_per_edge_round,
            )
            if (
                (system_round + 1) % 100 == 0
                or system_round + 1 == total_edge_rounds
            ):
                print(
                    "Latency replay: {}/{} edge rounds".format(
                        system_round + 1,
                        total_edge_rounds,
                    )
                )
        output_path = self.metrics.write_latency()
        print("Latency metrics: {}".format(output_path))
        return output_path

    def setup_clients(self, num_clients):
        return [
            Client(self.options, client_id, self.model, self.optimizer)
            for client_id in range(num_clients)
        ]

    def aggregate_parameters(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        weighted_solutions = list(solns)
        total_samples = sum(
            int(num_sample) for num_sample, _ in weighted_solutions
        )
        if total_samples <= 0:
            raise ValueError("cannot aggregate solutions with no samples")

        averaged_solution = torch.zeros_like(self.cloud_latest_global_model)
        for num_sample, local_solution in weighted_solutions:
            averaged_solution += int(num_sample) * local_solution
        averaged_solution /= total_samples
        return averaged_solution.detach()

    def global_test_latest_model_on_testdata(self, round_i):
        begin_time = time.time()
        stats_from_test_data = self.global_test()
        end_time = time.time()

        print(
            Fore.YELLOW + '= Test =' +
            Fore.CYAN + f' round: {round_i} / ' +
            Fore.GREEN + f'acc: {stats_from_test_data["acc"]:.3%} / ' +
            Fore.RED + f'loss: {stats_from_test_data["loss"]:.4f} / ' +
            Fore.MAGENTA + f'Time: {end_time - begin_time:.2f}s' +
            Style.RESET_ALL
        )
        print(Fore.BLUE + '=' * 102 + "\n" + Style.RESET_ALL)

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def _get_test_data_loader(self):
        if self._test_data_loader is None:
            test_path = "{}/test_data/test_data.npy".format(
                self.options['data_path']
            )
            test_data = np.load(test_path)
            features = torch.tensor(
                test_data[:, :-1], dtype=torch.float32
            )
            labels = torch.tensor(test_data[:, -1], dtype=torch.long)
            self._test_data_loader = DataLoader(
                TensorDataset(features, labels),
                batch_size=100,
                shuffle=False,
            )
        return self._test_data_loader

    def global_test(self):
        assert self.cloud_latest_global_model is not None
        self.set_flat_model_params(self.cloud_latest_global_model)
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for features, labels in self._get_test_data_loader():
                if self.gpu >= 0:
                    features = features.cuda()
                    labels = labels.cuda()
                _, prediction = self.model(features)
                loss = criterion(prediction, labels)

                predicted = prediction.argmax(dim=1)
                correct = predicted.eq(labels).sum()
                test_acc += correct.item()
                test_loss += loss.item() * labels.size(0)
                test_total += labels.size(0)

        return {
            'acc': test_acc / test_total,
            'loss': test_loss / test_total,
            'num_samples': test_total,
        }

    def select_edges(self):
        num_edges = int(self.per_round_e_fraction * self.edges_num)
        indices = np.random.choice(len(self.edges), num_edges, replace=False)
        return [self.edges[index] for index in indices]

