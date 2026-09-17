# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import torch

from src.models.cifar100_alexnet import CIFAR100_AlexNet
from src.models.cifar10_alexnet import CIFAR10_AlexNet
from src.models.fmnist_cnn import FMnist_CNN


MODELS = {
    'fmnist_cnn': FMnist_CNN,
    'cifar10_alexnet': CIFAR10_AlexNet,
    'cifar100_alexnet': CIFAR100_AlexNet,
}


def choose_model(options):
    model_name = str(options['model_name']).lower()
    torch.manual_seed(options['seed'] + 1)
    try:
        return MODELS[model_name]()
    except KeyError as error:
        raise ValueError(
            f"unsupported model_name {model_name!r}; "
            f"expected one of {sorted(MODELS)}"
        ) from error
