# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import numpy as np
import time
import copy
import heapq
import csv
import json
import os

from src.fed_cloud.base_cloud import BaseCloud
from src.optimizers.gd import GD
from src.models.model import choose_model
from src.virtualset.set_maker import load_prepared_groups

class Proposed(BaseCloud):

    # Subclasses only replace the logical-group mapping policy.  They retain
    # the same model continuity, communication, and FCFS latency accounting.
    enable_fcfs_mapping_search = True
    static_virtual_mapping = False
    random_virtual_mapping = False

    def __init__(self, options):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'])
        super(Proposed, self).__init__(options, model, self.optimizer, )
        self.best_clients_to_edge_map = {}
        self.group_distribution_source = self.options.get(
            'group_distribution', 'private_gram'
        )
        if self.group_distribution_source != 'private_gram':
            raise ValueError(
                f"unsupported group distribution: {self.group_distribution_source}"
            )
        self.best_G, private_cache = load_prepared_groups(self.options)
        print(f"Private-Gram virtual groups loaded from {private_cache}")
        private_runtime = self._load_private_preprocessing_metadata(
            private_cache
        )
        self.bfv_vsf_computation_time = private_runtime[
            'computation_seconds'
        ]
        self.bfv_client_context_bytes = private_runtime[
            'client_context_bytes'
        ]
        self.bfv_encrypted_histogram_max_bytes = private_runtime[
            'encrypted_histogram_max_bytes'
        ]
        self.client_group = {
            client_id: group_id
            for group_id, client_ids in self.best_G.items()
            for client_id in client_ids
        }
        if len(self.client_group) != self.options['num_of_clients']:
            raise ValueError("Every client must belong to exactly one virtual group")
        self.group_latest_model_set = {
            group_id: (0, copy.deepcopy(self.cloud_latest_global_model))
            for group_id in self.best_G
        }
        # virtual group -> physical ES currently hosting that group's model
        self.group_owner = {}
        self.random_mapping_seed = int(
            options.get('random_mapping_seed', options.get('seed', 2025))
        )
        self.random_mapping_rng = np.random.default_rng(
            self.random_mapping_seed
        )
        print(f"Virtual-group distribution source: {self.group_distribution_source}")
        print("self.best_G", self.best_G)

    @staticmethod
    def _load_private_preprocessing_metadata(cache_path):
        metadata_path = cache_path + '.privacy.json'
        try:
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)
        except FileNotFoundError:
            print(
                'WARNING: privacy runtime metadata is missing; BFV-VSF '
                'preprocessing time will not be added: {}'.format(metadata_path)
            )
            return {
                'computation_seconds': 0.0,
                'client_context_bytes': 0,
                'encrypted_histogram_max_bytes': 0,
            }

        if 'preprocessing_computation_seconds' in metadata:
            runtime = float(metadata['preprocessing_computation_seconds'])
        else:
            runtime = sum(float(metadata.get(field, 0.0)) for field in (
                'keygen_seconds',
                'client_encryption_seconds',
                'gram_evaluation_and_decryption_seconds',
                'grouping_seconds',
            ))
        if runtime < 0:
            raise ValueError('BFV-VSF preprocessing time cannot be negative')
        num_clients = int(metadata.get('num_clients', 0))
        encrypted_total = int(metadata.get('encrypted_histogram_bytes', 0))
        encrypted_max = metadata.get('encrypted_histogram_max_bytes')
        if encrypted_max is None:
            encrypted_max = (
                int(np.ceil(encrypted_total / num_clients))
                if num_clients > 0 else 0
            )
        print('BFV-VSF preprocessing computation time: {:.6f}s'.format(runtime))
        return {
            'computation_seconds': runtime,
            'client_context_bytes': int(metadata.get('client_context_bytes', 0)),
            'encrypted_histogram_max_bytes': int(encrypted_max),
        }

    def _bfv_vsf_mc_communication_time(self, system_round=0):
        """Slowest MC's one-time BFV context download and ciphertext upload."""
        if self.group_distribution_source != 'private_gram':
            return 0.0
        down_rates = np.asarray(self.system_params['D'][system_round], dtype=float)
        up_rates = np.asarray(self.system_params['U'][system_round], dtype=float)
        if np.any(down_rates <= 0) or np.any(up_rates <= 0):
            raise ValueError('MC access rates must be positive for BFV communication')
        context_mb = self.bfv_client_context_bytes / 1_000_000.0
        ciphertext_mb = self.bfv_encrypted_histogram_max_bytes / 1_000_000.0
        per_client_seconds = (
            8.0 * context_mb / down_rates
            + 8.0 * ciphertext_mb / up_rates
        )
        runtime = (
            float(per_client_seconds.max())
            if per_client_seconds.size else 0.0
        )
        print('BFV-VSF MC-side communication time: {:.6f}s'.format(runtime))
        return runtime
    def get_clients_label_distribution(self):
        clients_data_distribution = []

        for client_id in range(self.options['num_of_clients']):
            train_dir = f"{self.options['data_path']}/client_{client_id + 1}"
            train_data = np.load(f"{train_dir}/train_data.npy")
            labels = train_data[:, -1].astype(int)
            unique_labels, counts = np.unique(labels, return_counts=True)
            client_distribution = np.zeros(
                self.options['num_classes'], dtype=int
            )
            client_distribution[unique_labels] = counts

            clients_data_distribution.append(client_distribution.tolist())
        return clients_data_distribution

    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_e_fraction * self.edges_num)))
        self.last_completed_mapping = None
        measure_own_runtime = self.options.get('server') == 'proposed'
        preprocessing_pending = measure_own_runtime
        bfv_mc_communication_time = (
            self._bfv_vsf_mc_communication_time(system_round=0)
            if measure_own_runtime else 0.0
        )

        for round_i in range(self.num_round):
            print('-' * 51 + "\n")
            self.global_test_latest_model_on_testdata(round_i)
            for round_edge in range(self.options['edge_epoch']):
                system_round = round_i * self.options['edge_epoch'] + round_edge
                cloud_sync = round_edge == 0
                source_owner = None if cloud_sync else dict(self.group_owner)
                self.assign_clients_to_edges(system_round)
                selected_edges = self.select_edges()
                dmm_started = time.perf_counter()
                mapping, logical_owner = self.DMM(
                    selected_edges,
                    link_rates=self.system_params['V'][system_round],
                )

                physical_owner = {
                    client_id: edge_id
                    for edge_id, client_ids in self.client_to_edge_map.items()
                    for client_id in client_ids
                }
                if self.enable_fcfs_mapping_search:
                    mapping, logical_owner, latency_result = self.optimize_mapping_by_latency(
                        mapping=mapping,
                        system_round=system_round,
                        physical_owner=physical_owner,
                        source_owner=source_owner,
                        cloud_sync=cloud_sync,
                    )
                else:
                    latency_result = self.cost.get_routed_latency(
                        selected_clients=self.clients,
                        round_i=system_round,
                        system_params=self.system_params,
                        physical_owner=physical_owner,
                        logical_owner=logical_owner,
                        path_finder=self.shortest_es_path,
                        model_size=self.options['model_size'],
                        cloud_sync=cloud_sync,
                        client_group=self.client_group,
                        source_owner=source_owner,
                    )
                dmm_time = time.perf_counter() - dmm_started
                preprocessing_computation_time = 0.0
                preprocessing_communication_time = 0.0
                if preprocessing_pending:
                    preprocessing_computation_time = (
                        self.bfv_vsf_computation_time
                    )
                    preprocessing_communication_time = (
                        bfv_mc_communication_time
                    )
                    preprocessing_pending = False
                latency_cost = latency_result['latency']
                # Decision runtimes are reported separately and are not added
                # to the common system latency model used by all frameworks.
                if measure_own_runtime:
                    self.metrics.update_hflm3_runtime(
                        round_i,
                        dmm_time=dmm_time,
                        preprocessing_computation_time=(
                            preprocessing_computation_time
                        ),
                        preprocessing_communication_time=(
                            preprocessing_communication_time
                        ),
                    )
                self.virtual_group_round(selected_edges, mapping)
                self.group_owner = {
                    virtual_group: owner
                    for owner, virtual_group in mapping.items()
                }
                self.metrics.update_costs(
                    round_i,
                    latency_cost,
                    self.options['num_of_clients'] * 2
                    + self.options['num_of_edges'] * 2 / self.options['edge_epoch'],
                )
            if self.options['pathe'] == True:
                self.optimizer.soft_decay_learning_rate2(self.options['dataset_name'])
            else:
                self.optimizer.soft_decay_learning_rate(self.options['dataset_name'], self.options['dirichlet'])

            self.cloud_latest_global_model = self.aggregate_parameters(
                list(self.group_latest_model_set.values())
            )
            for e in range(self.edges_num):
                self.edge_latest_model_set[e] = (0, copy.deepcopy(self.cloud_latest_global_model))
            for group_id in self.group_latest_model_set:
                self.group_latest_model_set[group_id] = (
                    0,
                    copy.deepcopy(self.cloud_latest_global_model),
                )
            # The next edge round starts from the cloud-broadcast global model,
            # which is already available at every ES.
            self.group_owner = {}
        self.global_test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    def DMM(
        self,
        selected_edges,
        link_rates=None,
    ):
        self.set_edge_graph_from_adj_matrix(
            self.adj_for_edges,
            link_rates=link_rates,
        )
        edges = sorted(selected_edges)
        virt_ids = sorted(self.best_G)
        E = len(edges)
        assert E == len(virt_ids), "number of edge servers must equal the number of virtual groups"
        if not hasattr(self, "es_dist"):
            raise RuntimeError("Call set_edge_graph_from_adj_matrix(...) before DMM")

        if self.random_virtual_mapping:
            permutation = self.random_mapping_rng.permutation(virt_ids)
            mapping = {
                edge: int(group)
                for edge, group in zip(edges, permutation)
            }
            self.last_mapping_seed_name = 'random'
        elif self.static_virtual_mapping:
            # The virtual-set identity is permanently bound to the matching
            # physical ES; mobility never triggers a remapping.
            mapping = {edge: group for edge, group in zip(edges, virt_ids)}
            self.last_mapping_seed_name = 'static'
        else:
            # Warm-start the FCFS search from the preceding round's final
            # permutation; only the first edge round uses the identity.
            previous = getattr(self, 'last_completed_mapping', None)
            if (
                previous is not None
                and set(previous) == set(edges)
                and set(previous.values()) == set(virt_ids)
            ):
                mapping = dict(previous)
                seed_name = 'previous_round'
            else:
                mapping = {
                    edge: group for edge, group in zip(edges, virt_ids)
                }
                seed_name = 'identity'
            self.last_mapping_seed_name = seed_name

        virt_to_edge = {v: e for e, v in mapping.items()}
        logical_owner = {}
        for v in virt_ids:
            owner = virt_to_edge[v]
            for c in self.best_G[v]:
                logical_owner[c] = owner

        return mapping, logical_owner

    def _logical_owner_from_mapping(self, mapping):
        group_owner = {group_id: edge_id for edge_id, group_id in mapping.items()}
        return {
            client_id: group_owner[group_id]
            for group_id, client_ids in self.best_G.items()
            for client_id in client_ids
        }

    @staticmethod
    def _solve_bottleneck_assignment(cost_matrix, edges, group_ids):
        """Solve ``min_mapping max(cost[edge, group])`` exactly.

        Feasibility at a threshold is a bipartite perfect-matching problem.
        Searching only the finite entries of the matrix gives the exact
        bottleneck value without enumerating all ``E!`` permutations.
        """
        from scipy.optimize import linear_sum_assignment

        costs = np.asarray(cost_matrix, dtype=float)
        edges = list(edges)
        group_ids = list(group_ids)
        if costs.shape != (len(edges), len(group_ids)):
            raise ValueError(
                "bottleneck cost matrix shape does not match edges/groups"
            )
        if len(edges) != len(group_ids):
            raise ValueError("bottleneck assignment requires equal partitions")
        finite = np.unique(costs[np.isfinite(costs)])
        if finite.size == 0:
            raise ValueError("bottleneck assignment has no finite candidate")

        low, high = 0, int(finite.size) - 1
        while low < high:
            mid = (low + high) // 2
            forbidden = (costs > finite[mid]).astype(np.int8)
            rows, cols = linear_sum_assignment(forbidden)
            if int(forbidden[rows, cols].sum()) == 0:
                high = mid
            else:
                low = mid + 1

        threshold = float(finite[low])
        forbidden = (costs > threshold).astype(np.int8)
        rows, cols = linear_sum_assignment(forbidden)
        if int(forbidden[rows, cols].sum()) != 0:
            raise ValueError("no perfect matching exists for finite costs")
        mapping = {
            int(edges[row]): int(group_ids[col])
            for row, col in zip(rows, cols)
        }
        return threshold, mapping

    def no_queue_bottleneck_lower_bound(
        self, system_round, physical_owner, source_owner, cloud_sync
    ):
        """Return a state-conditioned lower bound on the FCFS makespan.

        The relaxation retains the unavoidable shortest-path service time of
        group-model dissemination and client-update forwarding, but gives
        every ES flow a private link (i.e., removes all FCFS waiting).  The
        remaining one-to-one group/ES decision is solved as an exact
        bottleneck assignment.
        """
        edges = sorted(int(edge) for edge in self.edges)
        group_ids = sorted(int(group) for group in self.best_G)
        if len(edges) != len(group_ids):
            raise ValueError("lower bound requires one virtual group per ES")
        if not cloud_sync and source_owner is None:
            raise ValueError("source_owner is required outside cloud-sync rounds")

        rates = np.asarray(self.system_params['V'][system_round], dtype=float)
        adjacency = np.asarray(self.adj_for_edges)
        model_size = float(self.options['model_size'])
        edge_count = len(edges)

        # All-pairs fastest no-queue model-transfer times, in seconds.
        transfer = np.full((edge_count, edge_count), np.inf, dtype=float)
        for source in edges:
            transfer[source, source] = 0.0
            queue = [(0.0, source)]
            while queue:
                distance, current = heapq.heappop(queue)
                if distance != transfer[source, current]:
                    continue
                for target in edges:
                    if current == target or adjacency[current, target] == 0:
                        continue
                    rate = float(rates[current, target])
                    if not np.isfinite(rate) or rate <= 0:
                        raise ValueError(
                            "invalid ES rate V[{},{}]={}".format(
                                current, target, rate
                            )
                        )
                    candidate = distance + 8.0 * model_size / rate
                    if candidate < transfer[source, target]:
                        transfer[source, target] = candidate
                        heapq.heappush(queue, (candidate, target))

        clients_by_id = {int(client.idx): client for client in self.clients}
        base_ready = {
            client_id: (
                clients_by_id[client_id].get_downmodel_latency(
                    system_round, self.system_params
                )
                + clients_by_id[client_id].getLocalDelay(
                    system_round, self.system_params
                )
                + clients_by_id[client_id].getUploadDelay(
                    system_round, self.system_params
                )
            )
            for client_id in sorted(physical_owner)
        }

        costs = np.full((edge_count, edge_count), np.inf, dtype=float)
        for row, aggregation_edge in enumerate(edges):
            for column, group_id in enumerate(group_ids):
                if not cloud_sync and group_id not in source_owner:
                    raise KeyError("missing previous owner for group {}".format(group_id))
                previous_owner = None if cloud_sync else int(source_owner[group_id])
                completion_times = []
                for client_id in self.best_G[group_id]:
                    client_id = int(client_id)
                    if client_id not in physical_owner:
                        raise KeyError(
                            "missing physical ES for client {}".format(client_id)
                        )
                    physical_edge = int(physical_owner[client_id])
                    dissemination = (
                        0.0
                        if cloud_sync
                        else transfer[previous_owner, physical_edge]
                    )
                    completion_times.append(
                        dissemination
                        + base_ready[client_id]
                        + transfer[physical_edge, aggregation_edge]
                    )
                costs[row, column] = max(completion_times, default=0.0)

        latency, mapping = self._solve_bottleneck_assignment(
            costs, edges, group_ids
        )
        return {
            'latency': float(latency),
            'mapping': mapping,
            'cost_matrix': costs,
            'transfer_time': transfer,
        }

    def optimize_mapping_by_latency(
        self, mapping, system_round, physical_owner, source_owner, cloud_sync
    ):
        """Optimize a complete mapping using the exact FCFS round makespan."""
        def evaluate(candidate):
            logical = self._logical_owner_from_mapping(candidate)
            result = self.cost.get_routed_latency(
                selected_clients=self.clients,
                round_i=system_round,
                system_params=self.system_params,
                physical_owner=physical_owner,
                logical_owner=logical,
                path_finder=self.shortest_es_path,
                model_size=self.options['model_size'],
                cloud_sync=cloud_sync,
                client_group=self.client_group,
                source_owner=source_owner,
            )
            return logical, result

        best_mapping = dict(mapping)
        best_logical, best_result = evaluate(best_mapping)
        seed_name = getattr(self, 'last_mapping_seed_name', 'identity')

        initial_mapping = dict(best_mapping)
        initial_latency = float(best_result['latency'])
        edges = sorted(best_mapping)
        max_passes = max(0, int(self.options.get('mapping_search_passes', 10)))
        candidate_evaluations = 0
        passes_evaluated = 0
        accepted_swaps = []

        for pass_index in range(max_passes):
            passes_evaluated += 1
            improved = False
            pass_mapping = best_mapping
            pass_logical = best_logical
            pass_result = best_result
            pass_pair = None
            for i, edge_a in enumerate(edges):
                for edge_b in edges[i + 1:]:
                    candidate_evaluations += 1
                    candidate = dict(best_mapping)
                    candidate[edge_a], candidate[edge_b] = (
                        candidate[edge_b], candidate[edge_a]
                    )
                    logical, result = evaluate(candidate)
                    if result['latency'] + 1e-12 < pass_result['latency']:
                        pass_mapping = candidate
                        pass_logical = logical
                        pass_result = result
                        pass_pair = (edge_a, edge_b)
                        improved = True
            if not improved:
                break
            latency_before = float(best_result['latency'])
            group_pair_before = [
                int(best_mapping[pass_pair[0]]),
                int(best_mapping[pass_pair[1]]),
            ]
            best_mapping = pass_mapping
            best_logical = pass_logical
            best_result = pass_result
            accepted_swaps.append(
                {
                    'pass': int(pass_index + 1),
                    'edge_pair': [int(pass_pair[0]), int(pass_pair[1])],
                    'group_pair_before': group_pair_before,
                    'latency_before': latency_before,
                    'latency_after': float(best_result['latency']),
                }
            )

        final_latency = float(best_result['latency'])
        absolute_improvement = initial_latency - final_latency
        relative_improvement = (
            absolute_improvement / initial_latency if initial_latency > 0 else 0.0
        )
        self.last_mapping_search_stats = {
            'search_method': 'previous_pair',
            'system_round': int(system_round),
            'cloud_sync': bool(cloud_sync),
            'initial_latency': initial_latency,
            'optimized_latency': final_latency,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'accepted_swaps': accepted_swaps,
            'accepted_swap_count': len(accepted_swaps),
            'passes_evaluated': passes_evaluated,
            'candidate_evaluations': candidate_evaluations,
            'initial_mapping': {
                int(edge): int(group) for edge, group in initial_mapping.items()
            },
            'optimized_mapping': {
                int(edge): int(group) for edge, group in best_mapping.items()
            },
            'selected_seed': seed_name,
            'seed_latencies': {seed_name: initial_latency},
            'seed_evaluations': 1,
        }

        self.last_completed_mapping = dict(best_mapping)

        return best_mapping, best_logical, best_result

    def evaluate_mapping_search(self, write_latency=False):
        """Replay mobility and mapping only, without any local model training.

        The optimized owner of each virtual group is carried into the next
        edge round.  It is reset only at a cloud-synchronization boundary, as
        in :meth:`train`, so download forwarding and FCFS queues are evaluated
        with the same model-continuity semantics as formal training.
        """
        if len(self.best_G) != self.edges_num:
            raise ValueError(
                "mapping replay requires one virtual group per physical ES"
            )
        if float(self.per_round_e_fraction) != 1.0:
            raise ValueError("mapping replay currently requires --e_fraction 1")

        total_edge_rounds = self.num_round * int(self.options['edge_epoch'])
        edge_epoch = int(self.options['edge_epoch'])
        rows = []
        self.group_owner = {}
        self.last_completed_mapping = None
        measure_own_runtime = self.options.get('server') == 'proposed'
        preprocessing_pending = bool(write_latency and measure_own_runtime)
        bfv_mc_communication_time = (
            self._bfv_vsf_mc_communication_time(system_round=0)
            if preprocessing_pending else 0.0
        )
        traffic_per_edge_round = (
            self.options['num_of_clients'] * 2
            + self.options['num_of_edges'] * 2 / edge_epoch
        )

        for system_round in range(total_edge_rounds):
            edge_round = system_round % edge_epoch
            cloud_round = system_round // edge_epoch
            cloud_sync = edge_round == 0
            if cloud_sync:
                self.group_owner = {}
            source_owner = None if cloud_sync else dict(self.group_owner)

            self.assign_clients_to_edges(system_round)
            selected_edges = list(self.edges)
            mapping_started = time.perf_counter()
            mapping, logical_owner = self.DMM(
                selected_edges,
                link_rates=self.system_params['V'][system_round],
            )
            physical_owner = {
                client_id: edge_id
                for edge_id, client_ids in self.client_to_edge_map.items()
                for client_id in client_ids
            }
            client_only_latency = float(
                self.cost.get_latency_sum(
                    self.clients,
                    system_round,
                    self.system_params,
                )
            )

            if self.enable_fcfs_mapping_search:
                mapping, logical_owner, latency_result = self.optimize_mapping_by_latency(
                    mapping=mapping,
                    system_round=system_round,
                    physical_owner=physical_owner,
                    source_owner=source_owner,
                    cloud_sync=cloud_sync,
                )
                stats = dict(self.last_mapping_search_stats)
            else:
                latency_result = self.cost.get_routed_latency(
                    selected_clients=self.clients,
                    round_i=system_round,
                    system_params=self.system_params,
                    physical_owner=physical_owner,
                    logical_owner=logical_owner,
                    path_finder=self.shortest_es_path,
                    model_size=self.options['model_size'],
                    cloud_sync=cloud_sync,
                    client_group=self.client_group,
                    source_owner=source_owner,
                )
                latency = float(latency_result['latency'])
                stats = {
                    'system_round': system_round,
                    'cloud_sync': cloud_sync,
                    'initial_latency': latency,
                    'optimized_latency': latency,
                    'absolute_improvement': 0.0,
                    'relative_improvement': 0.0,
                    'accepted_swaps': [],
                    'accepted_swap_count': 0,
                    'passes_evaluated': 0,
                    'candidate_evaluations': 0,
                    'initial_mapping': {
                        int(edge): int(group) for edge, group in mapping.items()
                    },
                    'optimized_mapping': {
                        int(edge): int(group) for edge, group in mapping.items()
                    },
                }
            mapping_runtime = time.perf_counter() - mapping_started

            lower_bound_result = self.no_queue_bottleneck_lower_bound(
                system_round=system_round,
                physical_owner=physical_owner,
                source_owner=source_owner,
                cloud_sync=cloud_sync,
            )
            no_queue_lower_bound = float(lower_bound_result['latency'])
            optimized_latency = float(latency_result['latency'])
            if optimized_latency + 1e-9 < no_queue_lower_bound:
                raise RuntimeError(
                    "FCFS latency is below its no-queue lower bound"
                )
            lower_bound_gap = (
                (optimized_latency - no_queue_lower_bound) / no_queue_lower_bound
                if no_queue_lower_bound > 0
                else 0.0
            )

            self.group_owner = {
                virtual_group: owner for owner, virtual_group in mapping.items()
            }
            stats.update(
                {
                    'cloud_round': int(cloud_round),
                    'edge_round': int(edge_round),
                    'client_only_latency': client_only_latency,
                    'forwarding_overhead': (
                        optimized_latency - client_only_latency
                    ),
                    'no_queue_lower_bound': no_queue_lower_bound,
                    'gap_vs_no_queue_lower_bound': lower_bound_gap,
                    'lower_bound_mapping': {
                        str(edge): int(group)
                        for edge, group in sorted(
                            lower_bound_result['mapping'].items()
                        )
                    },
                    'initial_mapping': {
                        str(edge): int(group)
                        for edge, group in sorted(stats['initial_mapping'].items())
                    },
                    'optimized_mapping': {
                        str(edge): int(group)
                        for edge, group in sorted(mapping.items())
                    },
                }
            )
            rows.append(stats)

            if write_latency:
                self.metrics.update_costs(
                    cloud_round,
                    optimized_latency,
                    traffic_per_edge_round,
                )
                if measure_own_runtime:
                    self.metrics.update_hflm3_runtime(
                        cloud_round,
                        dmm_time=mapping_runtime,
                        preprocessing_computation_time=(
                            self.bfv_vsf_computation_time
                            if preprocessing_pending else 0.0
                        ),
                        preprocessing_communication_time=(
                            bfv_mc_communication_time
                            if preprocessing_pending else 0.0
                        ),
                    )
                    preprocessing_pending = False

            if (system_round + 1) % 50 == 0 or system_round + 1 == total_edge_rounds:
                print(
                    "Mapping replay: {}/{} edge rounds".format(
                        system_round + 1, total_edge_rounds
                    )
                )

        report_path = self._write_mapping_search_report(rows)
        if write_latency:
            latency_path = self.metrics.write_latency()
            print("Latency metrics: {}".format(latency_path))
        return report_path, self._mapping_search_summary(rows)

    def evaluate_latency(self):
        """Replay routed communication and replace latency_metrics.json."""
        return self.evaluate_mapping_search(write_latency=True)

    @staticmethod
    def _mapping_search_summary(rows):
        count = len(rows)
        successful = [row for row in rows if row['accepted_swap_count'] > 0]
        relative = np.asarray(
            [row['relative_improvement'] for row in rows], dtype=float
        )
        summary = {
            'edge_rounds': count,
            'successful_edge_rounds': len(successful),
            'success_rate': len(successful) / count if count else 0.0,
            'total_accepted_swaps': int(
                sum(row['accepted_swap_count'] for row in rows)
            ),
            'mean_swaps_per_edge_round': float(
                np.mean([row['accepted_swap_count'] for row in rows])
            ) if rows else 0.0,
            'mean_relative_improvement': float(relative.mean()) if count else 0.0,
            'median_relative_improvement': float(np.median(relative)) if count else 0.0,
            'p95_relative_improvement': float(np.percentile(relative, 95)) if count else 0.0,
            'max_relative_improvement': float(relative.max()) if count else 0.0,
            'candidate_evaluations': int(
                sum(row['candidate_evaluations'] for row in rows)
            ),
        }
        if rows and all('no_queue_lower_bound' in row for row in rows):
            optimized_total = float(
                sum(row['optimized_latency'] for row in rows)
            )
            lower_bound_total = float(
                sum(row['no_queue_lower_bound'] for row in rows)
            )
            summary.update({
                'mean_no_queue_lower_bound': lower_bound_total / count,
                'total_no_queue_lower_bound': lower_bound_total,
                'aggregate_gap_vs_no_queue_lower_bound': (
                    (optimized_total - lower_bound_total) / lower_bound_total
                    if lower_bound_total > 0
                    else 0.0
                ),
            })
        if rows and all('client_only_latency' in row for row in rows):
            routed_total = float(
                sum(row['optimized_latency'] for row in rows)
            )
            client_only_total = float(
                sum(row['client_only_latency'] for row in rows)
            )
            forwarding_total = routed_total - client_only_total
            summary.update({
                'mean_routed_latency': routed_total / count,
                'total_routed_latency': routed_total,
                'mean_client_only_latency': client_only_total / count,
                'total_client_only_latency': client_only_total,
                'mean_forwarding_overhead': forwarding_total / count,
                'total_forwarding_overhead': forwarding_total,
            })
        return summary

    def _write_mapping_search_report(self, rows):
        requested = self.options.get('mapping_report')
        if requested:
            csv_path = os.path.abspath(requested)
            if not csv_path.lower().endswith('.csv'):
                csv_path += '.csv'
        else:
            filename = (
                'mapping_search_{}_{}_m{}_e{}_r{}_ee{}_seed{}_passes{}.csv'.format(
                    self.options['server'],
                    self.options.get('mobility_model', 'slaw'),
                    self.options['num_of_clients'],
                    self.options['num_of_edges'],
                    self.num_round,
                    self.options['edge_epoch'],
                    self.options.get('mobility_seed', self.options['seed']),
                    self.options.get('mapping_search_passes', 0),
                )
            )
            csv_path = os.path.abspath(
                os.path.join('analysis', 'mapping_search', filename)
            )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        csv_fields = [
            'system_round', 'cloud_round', 'edge_round', 'cloud_sync',
            'initial_latency', 'optimized_latency', 'absolute_improvement',
            'relative_improvement', 'accepted_swap_count', 'passes_evaluated',
            'candidate_evaluations', 'no_queue_lower_bound',
            'gap_vs_no_queue_lower_bound', 'client_only_latency',
            'forwarding_overhead',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in csv_fields})

        json_path = os.path.splitext(csv_path)[0] + '.json'
        payload = {
            'summary': self._mapping_search_summary(rows),
            'rounds': rows,
        }
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

        print("Mapping-search CSV: {}".format(csv_path))
        print("Mapping-search JSON: {}".format(json_path))
        print(json.dumps(payload['summary'], indent=2))
        return csv_path

    def virtual_group_round(self, selected_edges, mapping):
        all_clients_model_paras_set = []
        for e in selected_edges:
            clients_in_edge = self.client_to_edge_map[e]
            for client_id in clients_in_edge:
                client = self.clients[client_id]
                virtual_group = self.client_group[client_id]
                client.set_flat_model_params(
                    self.group_latest_model_set[virtual_group][1]
                )
                local_model_paras, _ = client.local_train()
                all_clients_model_paras_set.append((client_id, local_model_paras))
        self.aggregate_by_mapping(mapping, all_clients_model_paras_set)

    def aggregate_by_mapping(self, mapping, client_updates):

        cid2upd = {cid: (n, w) for cid, (n, w) in client_updates}
        for owner, virtual_group in mapping.items():
            cid_list = self.best_G[virtual_group]
            client_model_paras_set = []
            edge_data_num = 0
            for cid in cid_list:
                client_model_paras_set.append(cid2upd[cid])
                edge_data_num += cid2upd[cid][0]
            edge_model = self.aggregate_parameters(client_model_paras_set)
            self.group_latest_model_set[virtual_group] = (
                edge_data_num,
                copy.deepcopy(edge_model),
            )
            self.edge_latest_model_set[owner] = (edge_data_num, copy.deepcopy(edge_model))

    def set_edge_graph_from_adj_matrix(self, A, link_rates):
        """Build the inter-ES graph and its shortest transfer paths.

        A: adjacency matrix (ndarray or nested list); entries > 0 denote a link.
        link_rates: this round's V[e][f] link rates in Mbps.
        Sets self.edge_graph ({u: [v, ...]}), self.es_dist[u][v] (shortest
        transfer-time weight) and self._es_next[u][v] (next hop from u to v).
        """
        A = np.asarray(A)
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "adjacency matrix must be square"
        n = A.shape[0]
        rates = np.asarray(link_rates, dtype=float)
        if rates.shape != A.shape:
            raise ValueError(f"link rate shape {rates.shape} does not match adjacency {A.shape}")

        edge_graph = {i: [j for j in range(n) if A[i, j] != 0 and i != j] for i in range(n)}
        self.edge_graph = edge_graph

        self.es_dist = {u: {v: float('inf') for v in range(n)} for u in range(n)}
        self._es_next = {u: {} for u in range(n)}

        for s in range(n):
            prev = {s: None}
            dist = {s: 0.0}
            q = [(0.0, s)]
            while q:
                distance_x, x = heapq.heappop(q)
                if distance_x != dist.get(x):
                    continue
                for y in sorted(edge_graph.get(x, [])):
                    rate = float(rates[x, y])
                    if not np.isfinite(rate) or rate <= 0:
                        raise ValueError(f"invalid rate V[{x},{y}]={rate} on an ES link")
                    weight = 8.0 / rate
                    candidate = distance_x + weight
                    if candidate < dist.get(y, float('inf')):
                        dist[y] = candidate
                        prev[y] = x
                        heapq.heappush(q, (candidate, y))
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
        """Return the shortest path [u, ..., v] from the next-hop table; raise if unreachable."""
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
        for _ in range(len(self._es_next) + 5):
            cur = self._es_next[cur][v]
            if cur in visited:
                break
            path.append(cur)
            visited.add(cur)
            if cur == v:
                return path
        raise RuntimeError("Failed to reconstruct path; check adjacency matrix or next-table.")
