import random
import pickle
import os
import numpy as np
import copy
np.random.seed(2025)
def banlance(D_distribution, clients_data_distribution):
    D = copy.deepcopy(D_distribution)
    for i in range(len(D)):
        D[i] += clients_data_distribution[i]
    sum_sample = sum(D)
    mathcal_Q = 0
    for i in range(len(D)):
        mathcal_Q += (D[i] / sum_sample - 1 / abs(len(D))) ** 2
    return mathcal_Q

class Set_Maker():
    def __init__(self, clients_data_distribution, options):
        self.clients_label_distribution = clients_data_distribution
        self.options = options
        if self.options['pathe'] == True:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setfile', 'pathology')
            os.makedirs(self.script_dir, exist_ok=True)
            self.suffix = 'dn_{}_noc_{}_noe_{}_slice_{}'.format(
                                self.options['dataset_name'],
                                self.options['num_of_clients'],
                                self.options['num_of_edges'],
                                self.options['slice']
                                )
        else:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setfile', 'dirichlet')
            os.makedirs(self.script_dir, exist_ok=True)
            self.suffix = 'dn_{}_noc_{}_dir_{}_noe{}'.format(
                                            self.options['dataset_name'],
                                            self.options['num_of_clients'],
                                            self.options['num_of_edges'],
                                            self.options['dirichlet'],)
        self.load_or_generate_set()

    def get_best_set(self):
        a = []
        G = {}
        k = 0
        random.seed(2025)
        clients_index = [_ for _ in range(self.options['num_of_clients'])]
        while len(clients_index) != 0:
            set = []
            D_distribution = [0 for _ in range(len(self.clients_label_distribution[0]))]
            g1 = random.choice(clients_index)
            for i in range(len(D_distribution)):
                D_distribution[i] += self.clients_label_distribution[g1][i]
            set.append(g1)
            clients_index.remove(g1)
            # while (len(set) < (self.options['num_of_clients'] / 10)) and (sum(D_distribution) < 800):
            while (len(set) < (self.options['num_of_clients'] / 10)):
                more_banlance_client = clients_index[0]
                for j in clients_index:
                    # 将j和set组合, 然后判断最小
                    # If set + 第一个客户端的距离 比 set + 第二个客户端的距离 更平衡，那么就选择第一个客户端;
                    if banlance(D_distribution, self.clients_label_distribution[j]) \
                            < banlance(D_distribution, self.clients_label_distribution[more_banlance_client]):
                        more_banlance_client = j
                set.append(more_banlance_client)
                for i in range(len(D_distribution)):
                    D_distribution[i] += self.clients_label_distribution[more_banlance_client][i]
                clients_index.remove(more_banlance_client)
            a.append(D_distribution)
            G[k] = set
            k += 1
        # print(a)
        return G   

    def load_or_generate_set(self, filename='set.pkl'):
        filename = self.suffix + filename
        print("self.script_dir", self.script_dir)
        try:
            # 尝试加载已保存的文件
            self.load_set(os.path.join(self.script_dir, filename))
            print("Group loaded from file.")
        except FileNotFoundError:
            # 如果文件不存在，生成新的set并保存
            self.G = self.get_best_set()
            self.save_set(os.path.join(self.script_dir, filename))
            print("New set generated and saved to file.")


    def save_set(self, filename='set.pkl'):
 
        with open(os.path.join(self.script_dir, filename), 'wb') as file:
            pickle.dump(self.G, file)

    def load_set(self, filename='set.pkl'):

        with open(os.path.join(self.script_dir, filename), 'rb') as file:
            self.G = pickle.load(file)