
import torch
import math
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import numpy as np
import os
def paraGeneration(options, save_dir="data/system_heter/", save_filename="parameters.npy"):
    round_num = options['round_num']
    num_clients = options['num_of_clients']
    edge_round = options['edge_epoch']
    np.random.seed(2025)

    # 打印当前工作目录，确保路径正确
    print(f"Current working directory: {os.getcwd()}")

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    # CPU clock speed f_ for each client
    # ================== CPU 频率：示例仍按客户端存 ==================
    # cpu_frequency[c][t]：第 c 个客户端在第 t 轮的 CPU 频率
    cpu_frequency = [
        np.round(np.random.uniform(0.1, 5, size=num_clients), 1)
        for _ in range(round_num * edge_round)
    ]

    # ================== 上行带宽：==================
    # bandwidth_ul = [np.round(np.random.uniform(1, 5, size=num_clients), 1) for _ in range(round_num * edge_round)]
    # # print(bandwidth_ul)
    # # print("上行带宽示例（客户端 0）：", len(bandwidth_ul))
    # # ================== 下行带宽：为上行的固定倍数 ==================
    # ratio_dl_ul = 10.0
    # bandwidth_dl = [ratio_dl_ul * arr for arr in bandwidth_ul]  # 列表里仍是 ndarray
    # # 如果也想固定一位小数：
    # bandwidth_dl = [np.round(ratio_dl_ul * arr, 1) for arr in bandwidth_ul]

    U = [
        np.round(np.random.uniform(1, 10, size=num_clients), 1)
        for _ in range(round_num * edge_round)
    ]
    D = [u_array * 10 for u_array in U]
    # ================== 发射功率：按客户端 ==================
    transmit_power = [
        np.round(np.random.uniform(1, 10), 1)
        for _ in range(num_clients)
    ]

    # ================== 打包参数并保存 ==================
    params = {
        # cpu_frequency：仍是 [client][round]，如果你也想改成 [round][client]，可以再调
        'cpu_frequency':  cpu_frequency,
        # 这两个已经是你要的结构：
        # [[客户端1, 客户端2, ...], [客户端1, 客户端2, ...], ...]
        # 'bandwidth_ul':   bandwidth_ul,
        # 'bandwidth_dl':   bandwidth_dl,
        # 'ratio_dl_ul':    ratio_dl_ul,
        'U': U,
        'D': D,
        'transmit_power': transmit_power
    }

    save_path = os.path.join(save_dir, save_filename)
    np.save(save_path, params)
    print(f"Parameters saved to {save_path}")
   

if __name__ == "__main__":
    # Example usage
    options = {
        'num_of_clients': 5,
        'round_num': 10
    }
    paraGeneration(options)