# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
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
        self.logger = logging.getLogger(__name__)
        self.options = options
        self.idx = idx
        self.model = model
        self.optimizer = optimizer
        self.gpu = options['gpu']
        self.train_dir = f"{self.options['data_path']}/client_{self.idx + 1}"
        train_data = np.load(f"{self.train_dir}/train_data.npy", mmap_mode='r')
        # Only the total local sample count is exposed to the server-side
        # estimator; per-class counts remain local to the client.
        self.data_count = int(train_data.shape[0])

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def pretrain_for_distribution(self, epochs):
        """Return a warm-up model used only to estimate this client's labels.

        Every client is reset to the same initial model by the caller.  A
        full-batch update follows the estimator used in ICC-J-P and avoids
        exposing the client's per-class sample counts.
        """
        if epochs <= 0:
            raise ValueError("pretrain epochs must be positive")

        train_data = np.load(f"{self.train_dir}/train_data.npy")
        X = torch.tensor(train_data[:, :-1], dtype=torch.float32)
        y = torch.tensor(train_data[:, -1], dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=len(y), shuffle=False)

        self.model.train()
        for _ in range(int(epochs)):
            for batch_x, batch_y in loader:
                if self.gpu >= 0:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                self.optimizer.zero_grad()
                _, prediction = self.model(batch_x)
                loss = criterion(prediction, batch_y)
                loss.backward()
                self.optimizer.step()

        return self.get_flat_model_params()


    def local_train(self, ):
        begin_time = time.time()
        local_model_paras, dict = self.local_update(self.options)
        end_time = time.time()
        stats = {'id': self.idx, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (dict["size"], local_model_paras), stats

    def local_update(self, options):
        train_dir = f"{self.options['data_path']}/client_{self.idx + 1}"
        train_data = np.load(f"{train_dir}/train_data.npy")
        X = train_data[:, :-1]
        y = train_data[:, -1]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        local_dataset = TensorDataset(X, y)
        # batch_size=options['batch_size']
        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
                used_indices = list(range(len(local_dataset)))
            else:
                sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
                used_indices = list(sampler)
                # print("used_", used_indices)
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
                # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
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
        # model_size is MB and U is Mbps.
        uploadDelay = 8.0 * self.options['model_size'] / (system_params['U'][round_i][self.idx])
        return uploadDelay

    def get_downmodel_latency(self, round_i, system_params):
        # model_size is MB and D is Mbps.
        down_model_latency = 8.0 * self.options['model_size'] / (system_params['D'][round_i][self.idx])
        return down_model_latency

    # def getSumEngery(self, round_i):
    #     return self.getUploadEngery(round_i) + self.getLocalEngery(round_i)

    # def getSumDelay(self, round_i):
    #     return self.getUploadDelay(round_i) + self.getLocalDelay(round_i)

