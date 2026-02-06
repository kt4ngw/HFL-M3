
from src.getdata import GetDataSet
import argparse
import torch
from src.utils.dirichlet import dirichlet_split_noniid
import importlib
from src.utils.tool_utils import setup_seed
from src.utils.tool_utils import paraGeneration
from src.models.model import choose_model
import logging
import datetime
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# GLOBAL PARAMETERS
DATASETS = ['mnist', 'fashionmnist', 'cifar10']
TRAINERS = {'base_cloud': 'BaseCloud',
            'hfl': 'HFL',
            'proposed': 'Proposed',
            'macfl': 'MACFL',
            # 'proposed2': 'Proposed2',
            'middle': 'MIDDLE',
            # 'csahfl': 'CSAHFL',
            }
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

OPTIMIZERS = TRAINERS.keys()
def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument('-is_iid', type=bool, default=True, help='data distribution is iid.')
    parser.add_argument('--dirichlet', default=0.2, type=float, help='Dirichlet;')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='name of dataset.')
    parser.add_argument('--model_name', type=str, default='cifar10_resnet18', help='the model to train')
    parser.add_argument('--gpu', type=int, default=5, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=500, help='number of round in comm')
    parser.add_argument('--num_of_edges', type=int, default=10, help='number of the edges')
    parser.add_argument('--num_of_clients', type=int, default=200, help='number of the clients')
    parser.add_argument('--e_fraction', type=float, default=1, help='E fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--c_fraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=3, help='local train epoch')
    parser.add_argument('--edge_epoch', type=int, default=2, help='edge train epoch')    
    parser.add_argument('--batch_size', type=int, default=128, help='local train batch size')
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate, use value from origin paper as default")
    parser.add_argument('--seed', type=int, default=2028, help='seed for randomness;')
    parser.add_argument('--server', type=str, default='hfl', help='server')
    parser.add_argument('--opti', type=str, default='gd', help='optimize_;')
    parser.add_argument('--is_real_class', type=bool, default=True, help='is or is not evaluate class;')
    parser.add_argument('--C', type=int, default=200000, help='comptu. one sample.',)
    parser.add_argument('--num_classes', type=int, default=10, help='labels',)
    parser.add_argument('--pathe', type=bool, default=False, help='use pathe or not;')
    parser.add_argument('--slice', type=int, default=1, help='1')
    parser.add_argument('--sys_para_path', type=str, default='./data/system_heter/parameters.npy', help='system para;')
    parser.add_argument('--stay_prob', type=float, default=0.2, help='the prob that client stay in the edge;')
    args = parser.parse_args()
    args.data_path = os.path.join('./data/federated_data', args.dataset_name, f"nc{args.num_of_clients}", 'dir' + str(args.dirichlet))

    if args.pathe == True:
        args.data_path = os.path.join('./data/federated_data', args.dataset_name, f"nc{args.num_of_clients}", f"slice{args.slice}", 'pathe' )
    print("args.data_path", args.data_path)
    options = args.__dict__
    options['model_size'] = choose_model(options).get_model_size()
    print(options['model_size'])
    return options


import logging
import importlib


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    options = input_options()      
    # 创建 logs 目录（如果不存在）
    # log_dir = "logs"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # 生成日志文件路径
    # log_filename = os.path.join(log_dir, f"{options['server']}_{timestamp}.log")    
    # 配置全局 logger
    # logging.basicConfig(
    #     filename=log_filename,  # 存放到 logs 文件夹
    #     filemode="a",  # 追加模式
    #     level=logging.INFO,  # 记录 INFO 及以上级别的日志
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )

    # logger = logging.getLogger("Main")  # 主日志 
    # logger.info("日志系统初始化完成，日志存放在 %s", log_filename)
    
    dataset = GetDataSet(options)
    paraGeneration(options, )
    # params = np.load(options['sys_para_path'], allow_pickle=True).item()
    # print(params['bandwidth_ul']) 
    trainer_path = 'src.fed_cloud.%s' % options['server']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['server']])
    # logger.info("加载训练器：{}".format(options['server']))
    Fed = trainer_class(options)
    # logger.info("设置随机种子：{}".format(options['seed']))
    setup_seed(options['seed'])
    Fed.train()

if __name__ == '__main__':
    main()