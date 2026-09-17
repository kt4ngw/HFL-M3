# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
# Portions adapted from lx10077/fedavgpy (MIT License).
import pickle
import json
import hashlib
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image
import random
import torch

from src.data_paths import partition_tag



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        self.bytes_written = {c.idx: [0] * num_rounds for c in clients}
        self.client_computations = {c.idx: [0] * num_rounds for c in clients}
        self.bytes_read = {c.idx: [0] * num_rounds for c in clients}

        # global_test_data
        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds
        # local and upload e and d
        # self.local_latency = [0] * num_rounds
        # self.local_energy = [0] * num_rounds

        # self.upload_latency = [0] * num_rounds
        # self.upload_energy = [0] * num_rounds
        self.D = []

        # cost time and delay
        self.accumulation_delay = [0] * num_rounds
        self.dmm_runtime = [0.0] * num_rounds
        self.bfv_vsf_computation_time = 0.0
        self.bfv_vsf_mc_communication_time = 0.0
        # self.accumulation_energy = [0] * num_rounds
        self.Network_traffic = [0] * num_rounds
        self._active_cost_round = None



        data_partition = str(options.get('data_partition', 'dirichlet'))
        if data_partition == 'mobcorr_strict':
            self.result_path = mkdir(
                os.path.join(
                    './result/mobcorr_strict',
                    '{}_{}'.format(
                        str(self.options['dataset_name']).lower(),
                        partition_tag(options),
                    ),
                )
            )
        else:
            self.result_path = mkdir(os.path.join('./result/dirichlet', str(self.options['dataset_name']).lower() + str(self.options['dirichlet'])))
        if self.options['pathe'] == True:
            self.result_path = mkdir(os.path.join('./result/pathe', str(self.options['dataset_name']).lower() ))

        mobility_model = str(options.get('mobility_model', 'slaw')).lower()
        if mobility_model != 'slaw':
            raise ValueError("Proposed requires mobility_model='slaw'")
        mobility_suffix = 'mob_slaw_ms{}'.format(options.get('mobility_seed', 2025))

        if self.options['batch_size'] == 0:
            suffix = '{}_sd{}_lr{}_ne{}_le{}_ee{}_{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                            options['local_epoch'],
                                            options['edge_epoch'],
                                            mobility_suffix
                                               )
        else:
            suffix = '{}_sd{}_lr{}_ne{}_bs{}_le{}_ee{}_{}'.format(name,
                                            options['seed'],
                                            options['lr'],
                                            options['round_num'],
                                            options['batch_size'],
                                            options['local_epoch'],
                                            options['edge_epoch'],
                                            mobility_suffix
                                                )
        if self.options['pathe'] == True:
            suffix = suffix + '_slice{}'.format(options['slice'])
        if options['server'] in {'proposed', 'static_vg', 'random_vg'}:
            source = options.get('group_distribution', 'private_gram')
            suffix = suffix + '_gdist_{}'.format(source)
        # self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algorithm'],
        #                                      options['model_name'], suffix)

        self.exp_name = '{}_{}_gd_{}'.format(
            options['server'], options['model_name'], suffix
        )
        # train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        if options.get('mapping_only', False) or options.get(
            'latency_only', False
        ):
            # Replay modes write their reports explicitly and must not add
            # TensorBoard files to a formal training directory.
            self.eval_writer = None
        else:
            test_event_folder = mkdir(
                os.path.join(self.result_path, self.exp_name, 'eval.event')
            )
            # self.train_writer = SummaryWriter(train_event_folder)
            self.eval_writer = SummaryWriter(test_event_folder)

    def update_communication_stats(self, round_i, stats):
        idx, bytes_w, comp, bytes_r = \
            stats['idx'], stats['bytes_w'], stats['comp'], stats['bytes_r']
        self.bytes_written[idx][round_i] += bytes_w
        self.client_computations[idx][round_i] += comp
        self.bytes_read[idx][round_i] += bytes_r

    def extend_communication_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_communication_stats(round_i, stats)

    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def update_costs(self, round_i, latency_cost, traffic_cost):
        # Multiple edge rounds belong to the same cloud round.  Initialise the
        # cumulative value once, then add every edge-round cost instead of
        # overwriting all but the last one.
        if self._active_cost_round != round_i:
            previous_delay = self.accumulation_delay[round_i - 1] if round_i > 0 else 0
            previous_traffic = self.Network_traffic[round_i - 1] if round_i > 0 else 0
            self.accumulation_delay[round_i] = previous_delay
            self.Network_traffic[round_i] = previous_traffic
            self._active_cost_round = round_i

        self.accumulation_delay[round_i] += latency_cost
        self.Network_traffic[round_i] += traffic_cost

    def update_hflm3_runtime(
        self,
        round_i,
        dmm_time,
        preprocessing_computation_time=0.0,
        preprocessing_communication_time=0.0,
    ):
        """Record method-specific overhead outside the common latency model."""
        self.dmm_runtime[round_i] += float(dmm_time)
        self.bfv_vsf_computation_time += float(
            preprocessing_computation_time
        )
        self.bfv_vsf_mc_communication_time += float(
            preprocessing_communication_time
        )

    # def update_pre_compustion(self, e_comsumption):
    #     self.accumulation_energy[0] = e_comsumption


    def update_class_vloume_round(self, D):
        self.D.append(D)

    @staticmethod
    def _write_json_atomic(path, payload):
        temp_path = '{}.tmp-{}'.format(path, os.getpid())
        try:
            with open(temp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=8)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _run_metadata(self):
        keys = (
            'dataset_name',
            'model_name',
            'server',
            'seed',
            'round_num',
            'local_epoch',
            'edge_epoch',
            'batch_size',
            'lr',
            'num_of_clients',
            'num_of_edges',
            'data_partition',
            'dirichlet',
            'pathe',
            'slice',
            'mobility_model',
            'mobility_seed',
        )
        return {
            key: self.options[key]
            for key in keys
            if key in self.options
        }

    def _system_trace_metadata(self):
        configured_path = self.options.get('sys_para_path')
        if not configured_path:
            return {'path': None, 'available': False}

        path = os.path.abspath(os.path.expanduser(str(configured_path)))
        metadata = {
            'path': path,
            'file': os.path.basename(path),
            'available': os.path.isfile(path),
            'system_seed': int(self.options.get('system_seed', 2025)),
            'system_trace_slots': int(
                self.options.get('system_trace_slots', 0)
            ),
            'es_link_rate_min_mbps': float(
                self.options.get('es_link_rate_min', 0.0)
            ),
            'es_link_rate_max_mbps': float(
                self.options.get('es_link_rate_max', 0.0)
            ),
        }
        if not metadata['available']:
            return metadata

        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        metadata['sha256'] = digest.hexdigest()

        params = np.load(path, allow_pickle=True).item()
        for key, label in (
            ('U', 'client_uplink_mbps'),
            ('D', 'client_downlink_mbps'),
        ):
            values = np.asarray(params[key], dtype=float)
            metadata[label] = {
                'min': float(values.min()),
                'max': float(values.max()),
                'shape': list(values.shape),
            }
        rates = np.asarray(params['V'], dtype=float)
        positive_rates = rates[rates > 0]
        metadata['inter_es_mbps'] = {
            'min': float(positive_rates.min()),
            'max': float(positive_rates.max()),
            'shape': list(rates.shape),
        }
        return metadata

    def _output_dir(self):
        output_dir = os.path.join(self.result_path, self.exp_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def write_accuracy(self):
        output_dir = self._output_dir()
        accuracy_metrics = {
            'schema_version': 2,
            'metric_group': 'accuracy',
            'dataset': self.options['dataset_name'],
            'run': self._run_metadata(),
            'loss_on_g_test_data': self.loss_on_g_test_data,
            'acc_on_g_test_data': self.acc_on_g_test_data,
            'class_vloume': self.D,
        }
        accuracy_path = os.path.join(output_dir, 'accuracy_metrics.json')
        self._write_json_atomic(accuracy_path, accuracy_metrics)
        return accuracy_path

    def write_latency(self):
        """Atomically replace only the latency artifact for this run."""
        output_dir = self._output_dir()
        system_trace = self._system_trace_metadata()
        latency_metrics = {
            'schema_version': 2,
            'metric_group': 'latency',
            'dataset': self.options['dataset_name'],
            'run': self._run_metadata(),
            'system_trace': system_trace,
            'accumulation_delay': self.accumulation_delay,
            'network_traffic': self.Network_traffic,
            'dmm_runtime_per_cloud_round': self.dmm_runtime,
            'dmm_runtime_total': float(sum(self.dmm_runtime)),
            'bfv_vsf_computation_time': float(
                self.bfv_vsf_computation_time
            ),
            'bfv_vsf_mc_communication_time': float(
                self.bfv_vsf_mc_communication_time
            ),
            'bfv_vsf_preprocessing_time': float(
                self.bfv_vsf_computation_time
                + self.bfv_vsf_mc_communication_time
            ),
            'hflm3_runtime_overhead_total': float(
                self.bfv_vsf_computation_time
                + self.bfv_vsf_mc_communication_time
                + sum(self.dmm_runtime)
            ),
        }
        latency_path = os.path.join(output_dir, 'latency_metrics.json')
        self._write_json_atomic(latency_path, latency_metrics)
        return latency_path

    def write(self):
        """Write the two independent artifacts used by a training run."""
        self.write_accuracy()
        self.write_latency()
