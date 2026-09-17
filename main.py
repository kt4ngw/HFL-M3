# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
from src.getdata import GetDataSet
import argparse
import torch
from src.utils.dirichlet import dirichlet_split_noniid
import importlib
from src.utils.tool_utils import setup_seed
from src.utils.tool_utils import defaultSystemParameterPath
from src.utils.tool_utils import ensureSystemParameters
from src.models.model import choose_model
from src.data_paths import federated_data_path, validate_federated_data
import logging
import datetime
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# GLOBAL PARAMETERS
DATASETS = ['mnist', 'fashionmnist', 'cifar10']
TRAINERS = {'proposed': 'Proposed'}
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

OPTIMIZERS = TRAINERS.keys()
def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument('-is_iid', type=bool, default=True, help='data distribution is iid.')
    parser.add_argument('--dirichlet', default=0.2, type=float, help='Dirichlet;')
    parser.add_argument(
        '--data_partition',
        choices=('dirichlet', 'mobcorr_strict'),
        default='dirichlet',
        help='prepared client-data partition; mobcorr_strict does not use Dirichlet',
    )
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='name of dataset.')
    parser.add_argument('--model_name', type=str, default='cifar10_alexnet', help='the model to train')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=500, help='number of round in comm')
    parser.add_argument('--num_of_edges', type=int, default=10, help='number of the edges')
    parser.add_argument('--num_of_clients', type=int, default=200, help='number of the clients')
    parser.add_argument('--e_fraction', type=float, default=1, help='E fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--c_fraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=3, help='local train epoch')
    parser.add_argument('--edge_epoch', type=int, default=2, help='edge train epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='local train batch size')
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate, use value from origin paper as default")
    parser.add_argument('--seed', type=int, default=2025, help='seed for randomness;')
    parser.add_argument(
        '--group_seed',
        type=int,
        default=2025,
        help='seed identifying the fixed offline virtual-group artifact',
    )
    parser.add_argument('--server', type=str, default='proposed', choices=('proposed',), help='server')
    parser.add_argument(
        '--group_distribution',
        choices=('private_gram',),
        default='private_gram',
        help=(
            'source of the virtual groups; private_gram loads groups prepared '
            'with scripts/prepare_private_virtual_groups.py'
        ),
    )
    parser.add_argument('--C', type=int, default=200000, help='comptu. one sample.',)
    parser.add_argument('--num_classes', type=int, default=10, help='labels',)
    parser.add_argument('--pathe', type=bool, default=False, help='use pathe or not;')
    parser.add_argument('--slice', type=int, default=1, help='1')
    parser.add_argument(
        '--sys_para_path',
        type=str,
        default=None,
        help='shared system-parameter trace; derived from the physical setup if omitted',
    )
    parser.add_argument(
        '--system_trace_slots',
        type=int,
        default=1000 * 4,
        help='number of reusable edge-round system states to generate',
    )
    parser.add_argument(
        '--system_seed',
        type=int,
        default=2025,
        help='random seed for the reusable system-parameter trace',
    )
    parser.add_argument(
        '--mobility_model',
        choices=('slaw',),
        default='slaw',
        help=(
            'prepared client mobility source: SLAW associations'
        ),
    )
    parser.add_argument(
        '--mobility_file',
        type=str,
        default=None,
        help=(
            'prepared .npz association matrix; a conventional path for the '
            'selected mobility model is used if omitted'
        ),
    )
    parser.add_argument(
        '--mobility_seed',
        type=int,
        default=2025,
        help='mobility realization seed for SLAW; independent of --seed',
    )
    parser.add_argument('--es_link_rate_min', type=float, default=500.0,
                        help='minimum ES-to-ES link rate in Mbps')
    parser.add_argument('--es_link_rate_max', type=float, default=500.0,
                        help='maximum ES-to-ES link rate in Mbps')
    parser.add_argument('--mapping_search_passes', type=int, default=10,
                        help='maximum FCFS-aware pairwise mapping-search passes')
    parser.add_argument(
        '--random_mapping_seed',
        type=int,
        default=2025,
        help='seed for the random virtual-group-to-ES mapping baseline',
    )
    parser.add_argument(
        '--mapping_only',
        action='store_true',
        help='replay mobility and FCFS mapping without local model training',
    )
    parser.add_argument(
        '--latency_only',
        action='store_true',
        help=(
            'replay communication without local training and overwrite only '
            'latency_metrics.json'
        ),
    )
    parser.add_argument(
        '--mapping_report',
        type=str,
        default=None,
        help='optional output CSV path for --mapping_only',
    )
    args = parser.parse_args()
    options = args.__dict__
    args.data_path = federated_data_path(options)
    print("args.data_path", args.data_path)
    options['model_size'] = choose_model(options).get_model_size()
    print(options['model_size'])
    return options


import logging
import importlib


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    options = input_options()

    if options.get('data_partition', 'dirichlet') == 'dirichlet':
        GetDataSet(options)
    else:
        if options['dataset_name'].lower() not in ('fashionmnist', 'cifar10'):
            raise ValueError(
                'mobcorr_strict is implemented for Fashion-MNIST and CIFAR-10'
            )
        validate_federated_data(
            options['data_path'], options['num_of_clients']
        )
    system_param_path = options.get('sys_para_path')
    if not system_param_path:
        system_param_path = defaultSystemParameterPath(options)
        options['sys_para_path'] = system_param_path
    ensureSystemParameters(
        options,
        save_dir=os.path.dirname(system_param_path) or '.',
        save_filename=os.path.basename(system_param_path),
    )
    # params = np.load(options['sys_para_path'], allow_pickle=True).item()
    # print(params['bandwidth_ul'])
    trainer_path = 'src.fed_cloud.%s' % options['server']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['server']])
    Fed = trainer_class(options)
    setup_seed(options['seed'])
    if options['mapping_only'] and options['latency_only']:
        raise ValueError("choose either --mapping_only or --latency_only")
    if options['latency_only']:
        Fed.evaluate_latency()
    elif options['mapping_only']:
        if not hasattr(Fed, 'evaluate_mapping_search'):
            raise ValueError(
                "--mapping_only is supported only by mapping-based trainers"
            )
        Fed.evaluate_mapping_search()
    else:
        Fed.train()

if __name__ == '__main__':
    main()
