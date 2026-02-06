from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
import math

from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to

from torch.utils.data import TensorDataset

criterion = F.cross_entropy
mse_loss = nn.MSELoss()
from src.utils.torch_utils import *
import logging
class Client():
    def __init__(self, options, idx, model, optimizer):
        self.logger = logging.getLogger(__name__)  # 继承全局 logger
        self.options = options
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.gpu = options['gpu']
        self.train_dir = f"{self.options['data_path']}/client_{self.idx + 1}"  # 一位小数
        train_data = np.load(f"{self.train_dir}/train_data.npy")
        y = train_data[:, -1]   # 最后一列是标签
        self.data_count = len(y) 
    
    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)


    def local_train(self, ):
        begin_time = time.time()
        local_model_paras, dict = self.local_update(self.options)
        end_time = time.time()
        stats = {'id': self.idx, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (dict["size"], local_model_paras), stats

    def local_update(self, options):
        train_dir = f"{self.options['data_path']}/client_{self.idx + 1}"  # 一位小数
        train_data = np.load(f"{train_dir}/train_data.npy")
        # 数据拆分为输入特征（X）和标签（y）
        # X = train_data[:, :-1].reshape(-1, 3, 32, 32)    # 假设最后一列是标签
        X = train_data[:, :-1]    # 假设最后一列是标签
        y = train_data[:, -1]   # 最后一列是标签
        # 转换为PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        local_dataset = TensorDataset(X, y)
        # batch_size=options['batch_size']
        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
                used_indices = list(range(len(local_dataset)))  # 全部样本
            else:
                sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
                used_indices = list(sampler)
                # print("used_", used_indices)
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
                # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        # print(f"本轮训练使用的样本索引: {used_indices}")
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                feature, pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
           # local_model_paras = self.get_model_parameters()   
        # print(self.get_flat_model_params())
        local_model_paras = self.get_flat_model_params()
        return_dict = {"size": len(train_data[:, :-1]),
                        "id": self.idx,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict
    

    # def getLocalEngery(self, round_i):
    #     if len(self.local_dataset) < self.options['batch_size']:
    #         dataset_len = len(self.local_dataset)
    #     else:
    #         dataset_len = self.options['batch_size']
    #     localEngery = (10 ** -26) * (self.attr_dict['cpu_frequency'][round_i][self.idx] * 10 ** 9) ** 2 * self.options['C'] * dataset_len * self.options['local_epoch']
    #     return localEngery

    # def getUploadEngery(self, round_i, bandwidth):
    #     uploadEngery = self.attr_dict['transmit_power'] * self.getUploadDelay(round_i, bandwidth)
    #     return uploadEngery

    def getLocalDelay(self, round_i, system_params):
        if self.data_count < self.options['batch_size']:
            dataset_len = self.data_count 
        else:
            dataset_len = self.options['batch_size']
        localDelay = (self.options['C'] * dataset_len * self.options['local_epoch']) / (system_params['cpu_frequency'][round_i][self.idx] * 10 ** 9)
        return localDelay

    def getUploadDelay(self, round_i, system_params):
        # R_K =  system_params['bandwidth_ul'][round_i][self.idx] * 1000000 * np.log2(1 + system_params['transmit_power'][self.idx] * 8) # 1M bit / s / self.B
        # # print("R_K", system_params['bandwidth_ul'][round_i][-1])
        # uploadDelay = self.options['model_size'] / (R_K / 8 / 1024 / 1024) # 100KB 0.1M  # 1S
        uploadDelay = self.options['model_size'] / (system_params['U'][round_i][self.idx])
        return uploadDelay

    def get_downmodel_latency(self, round_i, system_params):
        # Down_R_K = system_params['bandwidth_dl'][round_i][self.idx] * 1000000 * np.log2(1 + system_params['transmit_power'][self.idx] * 8)
        # down_model_latency = self.options['model_size'] / (Down_R_K / 8 / 1024 / 1024)
        down_model_latency = self.options['model_size'] / (system_params['D'][round_i][self.idx])
        return down_model_latency

    # def getSumEngery(self, round_i):
    #     return self.getUploadEngery(round_i) + self.getLocalEngery(round_i)

    # def getSumDelay(self, round_i):
    #     return self.getUploadDelay(round_i) + self.getLocalDelay(round_i)

