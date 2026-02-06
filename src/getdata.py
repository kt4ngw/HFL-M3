import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))          # 把 src/ 加入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根加入
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import gzip
# import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform
from torchvision import transforms
from PIL import Image
import scipy.io as sio  # 在文件开头加上
from skimage import img_as_float
from src.utils.dirichlet import dirichlet_split_noniid, shard_split_noniid
import random
from data_processing.imagenet20_dataset import ImageNet20Dataset
class GetDataSet():
    def __init__(self, options):
        self.options = options
        self.dataSetName = options['dataset_name']
        self.train_data = None
        self.train_label = None
        self.train_datasize = None

        self.test_data = None
        self.test_label = None
        self.test_datasize = None

        if self.dataSetName == 'MNIST' or self.dataSetName == 'mnist':
            self.mnistDataDistribution()
            print("mnist!!")
        elif self.dataSetName == 'EMNIST' or self.dataSetName == 'emnist':
            self.emnistDataDistribution()
            print("Emnist!!")
        elif self.dataSetName == 'CIFAR10' or self.dataSetName == 'cifar10':
            self.cifar10DataDistribution()
            print("cifar10!!")
        elif self.dataSetName == 'FASHIONMNIST' or self.dataSetName == 'fashionmnist':
            self.fashionmnistDataDistribution()
            print("fashion!!")
        elif self.dataSetName == 'CIFAR100' or self.dataSetName == 'cifar100':
            self.cifar100DataDistribution()
            print("cifar100!!")
        elif self.dataSetName == 'imagenette2' or self.dataSetName == 'IMAGENETTE2':
            self.imagenette2DataDistribution()
        elif self.dataSetName in ['tinyimagenet', 'tiny-imagenet-200', 'TINYIMAGENET']:
            self.tiny_imagenet_data_distribution()
            print("tiny-imagenet-200!!")
        elif self.dataSetName == 'SVHN' or self.dataSetName == 'svhn':
            self.svhnDataDistribution()
            print("SVHN!!")
        elif self.dataSetName == 'imagenet' or self.dataSetName == 'IMAGENET':
            self.imagenetDataDistribution()
            print("SVHN!!")
        elif self.dataSetName == 'stl':
            self.stlDataDistribution()
            print("STL~~~")

    def stlDataDistribution(self):
        data_dir = 'data/STL10' 
        # train_images_path = os.path.join(data_dir, 'train.bin')
        # test_images_path = os.path.join(data_dir, 'test.bin') 
        self.train_data = read_stl10_images(os.path.join(data_dir, 'train_X.bin'))
        self.train_label = read_stl10_labels(os.path.join(data_dir, 'train_y.bin'))    
        self.test_data = read_stl10_images(os.path.join(data_dir, 'test_X.bin'))
        self.test_label = read_stl10_labels(os.path.join(data_dir, 'test_y.bin'))
        print(self.train_label)
        print(self.train_data.shape)
        # 创建均衡的测试集
        balance_testData = []
        balance_testLabel = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(10)]
        min_number = min([len(class_) for class_ in class_index])
        min_number = 800
        for number in range(10):
            balance_testData.append(self.test_data[class_index[number][:min_number]])
            balance_testLabel += [number] * min_number
        self.test_data = np.concatenate(balance_testData, axis=0)
        self.test_label = np.array(balance_testLabel)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)    

        print("label 分布:", np.bincount(self.train_label))
        print("label 分布:", np.bincount(self.test_label))
        print(self.train_data.shape)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)
        pass

    def imagenetDataDistribution(self):
        data_dir = 'data/imagenet10'
        train_images_path = os.path.join(data_dir, 'train')
        test_images_path = os.path.join(data_dir, 'val')
        # ✅ 构造 Dataset
        self.train_dataset = ImageNet20Dataset(train_images_path)
        self.test_dataset  = ImageNet20Dataset(test_images_path)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,      # ✅ 多进程并行读图
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        train_data, train_label = [], []
        for x, y in train_loader:
            train_data.append(x)       # Tensor, 不转 numpy
            train_label.append(y)

        self.train_data  = torch.cat(train_data, dim=0)
        self.train_label = torch.cat(train_label, dim=0)

        test_data, test_label = [], []
        for x, y in test_loader:
            test_data.append(x)
            test_label.append(y)

        self.test_data  = torch.cat(test_data, dim=0)
        self.test_label = torch.cat(test_label, dim=0)

        self.save_splits(
            self.train_data, self.train_label,
            self.test_data,  self.test_label
        )
    
    def svhnDataDistribution(self):
        data_dir = './data/SVHN'
        train_data_path = os.path.join(data_dir, 'train_32x32.mat')
        test_data_path = os.path.join(data_dir, 'test_32x32.mat')
        
        # 加载数据
        train_data = sio.loadmat(train_data_path)
        test_data = sio.loadmat(test_data_path)
        
        # 提取图像和标签
        X_train = train_data['X']  # shape: (32, 32, 3, N)
        y_train = train_data['y'].flatten()  # shape: (N,)
        
        X_test = test_data['X']
        y_test = test_data['y'].flatten()
        
        # SVHN 的标签是 1-10，其中 10 表示数字 0，需要调整
        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0
        # {X:1 } 
        # 转换形状为 (N, 3, 32, 32)
        X_train = X_train.transpose(3, 2, 0, 1).astype(np.float32) / 255.0
        X_test = X_test.transpose(3, 2, 0, 1).astype(np.float32) / 255.0
        

        self.train_data = X_train
        self.train_label = y_train.astype(np.int64)
        
        self.test_data = X_test
        self.test_label = y_test.astype(np.int64)
        balance_trainData = []
        balance_trainLabel = []
        class_index = [np.argwhere(self.train_label == y).flatten() for y in range(10)]
        min_number = min([len(class_) for class_ in class_index])
        min_number = 4000
        for number in range(10):
            balance_trainData.append(self.train_data[class_index[number][:min_number]])
            balance_trainLabel += [number] * min_number
        self.train_data = np.concatenate(balance_trainData, axis=0)
        self.train_label = np.array(balance_trainLabel)
        self.train_label = torch.tensor(self.train_label).to(torch.int64)


        # 创建均衡的测试集
        balance_testData = []
        balance_testLabel = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(10)]
        min_number = min([len(class_) for class_ in class_index])
        min_number = 1000
        for number in range(10):
            balance_testData.append(self.test_data[class_index[number][:min_number]])
            balance_testLabel += [number] * min_number
        self.test_data = np.concatenate(balance_testData, axis=0)
        self.test_label = np.array(balance_testLabel)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)    
        print(self.train_data.shape)
        print(self.test_label)
        print(self.test_label.shape)
        print("label 分布:", np.bincount(self.train_label))
        print('获取测试数据 ...')
        print("label 分布:", np.bincount(self.test_label))
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)

                
    def tiny_imagenet_data_distribution(self):
        """
        下载/准备 tiny-imagenet-200 并加载到:
          self.train_data:  (N, 3, 32, 32) float32, [0,1]
          self.train_label: torch.int64
          self.test_data:   (M, 3, 32, 32) float32, [0,1]  (来自官方 val)
          self.test_label:  torch.int64
        """
        data_root = 'data'
        zip_url  = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        md5_ref  = '90528d7ca1a48142e341f4ef8d21d0de'
        out_dir  = os.path.join(data_root, 'tiny-imagenet-200')
        zip_path = os.path.join(data_root, 'tiny-imagenet-200.zip')

        # 1) 下载+校验+解压+验证集重排
        self._prepare_tiny_imagenet(zip_url, zip_path, md5_ref, out_dir)

        # 2) 加载（与 imagenette2 的读取逻辑一致，但这里源分辨率是 64x64；
        #    为了与你现有管线统一，这里会 resize 到 32x32）
        train_dir = os.path.join(out_dir, 'train')
        val_dir   = os.path.join(out_dir, 'val')   # 已被重排为 val/<cls>/*

        print('Preparing Tiny-ImageNet-200 dataset ...')
        print('获取训练数据 ...')

        # 构建 label 映射（保证类别顺序固定）
        labels = sorted(os.listdir(train_dir))
        label_to_idx = {lb:i for i, lb in enumerate(labels)}

        # 读取 train
        train_img_files, train_labels = [], []
        for lb in labels:
            lb_dir = os.path.join(train_dir, lb, 'images')
            if not os.path.isdir(lb_dir):
                # 有些版本是 train/<cls>/images/，也有版本是 train/<cls> 直接放图
                lb_dir = os.path.join(train_dir, lb)
            for fname in os.listdir(lb_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    train_img_files.append(os.path.join(lb_dir, fname))
                    train_labels.append(label_to_idx[lb])

        print('训练集:共 %d 张' % len(train_img_files))
        train_images = []
        for i, fpath in enumerate(train_img_files):
            img_3_32_32 = self._read_as_3x32x32(fpath)  # 统一到 (3,32,32)
            train_images.append(img_3_32_32)
            if (i + 1) % 2000 == 0:
                print('train 已完成 %d 张' % (i + 1))

        self.train_data = np.array(train_images, dtype=np.float32)
        self.train_label = torch.tensor(np.array(train_labels), dtype=torch.int64)
        self.train_datasize = len(self.train_data)
        print('训练数据准备完毕')

        # 读取 val 作为 test（官方无 label 的 test）
        print('获取验证(作为测试)数据 ...')
        val_labels = sorted(os.listdir(val_dir))
        # 与 train 的 label_to_idx 对齐（val 中类别应该一致）
        test_img_files, test_labels = [], []
        for lb in val_labels:
            lb_dir = os.path.join(val_dir, lb)
            for fname in os.listdir(lb_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_img_files.append(os.path.join(lb_dir, fname))
                    test_labels.append(label_to_idx[lb])  # 映射到同一索引空间

        print('测试集:共 %d 张' % len(test_img_files))
        test_images = []
        for i, fpath in enumerate(test_img_files):
            img_3_32_32 = self._read_as_3x32x32(fpath)
            test_images.append(img_3_32_32)
            if (i + 1) % 2000 == 0:
                print('test 已完成 %d 张' % (i + 1))

        self.test_data = np.array(test_images, dtype=np.float32)
        self.test_label = torch.tensor(np.array(test_labels), dtype=torch.int64)
        self.test_datasize = len(self.test_data)
        print('测试数据准备完毕')
        print(self.train_label.numpy(),)
        # 如果你希望像 CIFAR/Fashion 那样直接做划分保存，解除下一行注释即可
        self.save_splits(self.train_data, self.train_label.numpy(), self.test_data, self.test_label.numpy())


    def _prepare_tiny_imagenet(self, url, zip_path, md5_ref, out_dir):
        """
        如已存在解压后的 tiny-imagenet-200/ 且 val 已重排，则直接返回；
        否则下载→校验→解压→把 val 重排成 val/<cls>/xxx.jpeg
        """
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        # 如果目录已经就绪且 val 已按类别重排，则跳过
        val_dir = os.path.join(out_dir, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        annotations = os.path.join(val_dir, 'val_annotations.txt')
        ready = os.path.isdir(out_dir) and os.path.isdir(val_dir) and (not os.path.isdir(val_images_dir)) and (not os.path.isfile(annotations))
        if ready:
            print("Tiny-ImageNet-200 已准备好，跳过下载与重排。")
            return

        # 若未解压或未就绪，先检查 zip
        if not os.path.isfile(zip_path):
            print(f"下载: {url}")
            self._download(url, zip_path)

        # 校验 md5
        ok = self._check_md5(zip_path, md5_ref)
        if not ok:
            raise RuntimeError("MD5 校验失败，压缩包可能损坏。")

        # 解压
        if not os.path.isdir(out_dir):
            print("解压中 ...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(os.path.dirname(out_dir))
            print("解压完成。")

        # 重排 val 目录：val/images/* 依据 val_annotations.txt 移动到 val/<cls>/
        if os.path.isdir(val_images_dir) and os.path.isfile(annotations):
            print("重排验证集目录 ...")
            os.makedirs(val_dir, exist_ok=True)

            # 读取映射
            mapping = {}  # filename -> class
            with open(annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]

            # 创建类别目录
            for cls in set(mapping.values()):
                os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

            # 移动图片到相应目录
            for fname, cls in mapping.items():
                src = os.path.join(val_images_dir, fname)
                dst = os.path.join(val_dir, cls, fname)
                if os.path.isfile(src):
                    os.replace(src, dst)

            # 清理
            try:
                import shutil
                shutil.rmtree(val_images_dir)
            except Exception:
                pass
            # annotations 保留/删除均可，这里删除避免二次重排
            try:
                os.remove(annotations)
            except Exception:
                pass

            print("验证集目录重排完成。")

    def _download(self, url, dst):
        from urllib.request import urlretrieve
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        def _progress(block_count, block_size, total_size):
            downloaded = block_count * block_size
            percent = downloaded / total_size * 100 if total_size > 0 else 0
            print(f"\r  已下载 {percent:6.2f}% ({downloaded/1e6:5.1f}MB/{total_size/1e6:5.1f}MB)", end='')
        urlretrieve(url, dst, _progress)
        print("\n下载完成。")

    def _check_md5(self, file_path, md5_ref):
        import hashlib
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5.update(chunk)
        got = md5.hexdigest()
        print(f"MD5: {got} (期望 {md5_ref})")
        return got == md5_ref
    
    def _read_as_3x32x32(self, image_file):
        """读取任意图片 -> RGB -> resize 32x32 -> [0,1] float32 -> (3,32,32)"""
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((32, 32))
        arr = img_as_float(np.array(img))   # [0,1], HWC, float64
        arr = arr.astype(np.float32)
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return arr

    def imagenette2DataDistribution(self):
        data_dir = 'data/imagenette2'
        train_images_path = os.path.join(data_dir, 'train')
        test_images_path = os.path.join(data_dir, 'val')
        print('Preparing dataset ...')
        print('获取训练数据 ...')
        train_img = []
        train_label= []
        labels = os.listdir(train_images_path)
        print(labels) # ['n02979186', 'n03888257', 'n03394916', 'n03000684', 'n03417042', 'n03445777', 'n03425413', 'n02102040', 'n01440764', 'n03028079']
        for i, label in enumerate(labels): 
            img_files = os.listdir(train_images_path + '/' + label)
            train_img += img_files[:858]
            train_label += [i] * len(img_files[:858])
        print('训练集:共 %d 张' % len(train_img))
        train_images, train_labels = self.get_train_images(train_img, train_label, train_images_path)

        self.train_data = train_images.astype(np.float32)
        self.train_label= torch.tensor(train_labels).to(torch.int64)
        self.train_datasize = len(self.train_data)
        print("DEBUG train_images shape:", self.train_data.shape)
        print('训练数据准备完毕')
        # self.train_label = torch.tensor(train_labels).to(torch.int64)

        print("label 分布:", torch.bincount(self.train_label))
        print('获取测试数据 ...')
        test_img = []
        test_label = []
        labels = os.listdir(test_images_path) # ['n02979186', 'n03888257', 'n03394916', 'n03000684', 'n03417042', 'n03445777', 'n03425413', 'n02102040', 'n01440764', 'n03028079']
        print(labels)
        for i, label in enumerate(labels): 
            img_files = os.listdir(test_images_path + '/' + label + '/')
            test_img += img_files[:300]
            test_label += [i] * len(img_files[:300])
        print('测试集:共 %d 张' % len(test_img))
        test_images, test_labels = self.get_test_images(test_img, test_label, test_images_path)
        # test_images = np.multiply(test_images, 1.0 / 255.0)
        self.test_data = test_images.astype(np.float32)
        self.test_label = torch.tensor(test_labels).to(torch.int64)
        print("DEBUG self.test_data shape:", self.test_data.shape)
        self.test_datasize = len(self.test_data)
        print('测试数据准备完毕')
        print("self.train_label", self.train_label)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)
   
    
    def get_train_images(self, train_img, train_label, train_images_path):
        labels = os.listdir(train_images_path)
        train_images = []
        for i, img_file in enumerate(train_img):
            img_dir = train_images_path + '/' + labels[train_label[i]] + '/'
            img_3_32_32 = self.extract(img_dir + img_file)

            train_images.append(img_3_32_32)
            if (i + 1) % 1000== 0:
                print('已完成%d张图片' % (i+1))
        return np.array(train_images), np.array(train_label)
   
    def get_test_images(self, test_img, test_label, test_images_path):
        labels = os.listdir(test_images_path)
        test_images = []
        for i, img_file in enumerate(test_img):
            img_dir = test_images_path + '/' + labels[test_label[i]] + '/'
            img_3_32_32 = self.extract(img_dir + img_file)

            test_images.append(img_3_32_32)
            # test_images.append(img_3_32_32)
            if (i + 1) % 100== 0:
                print('已完成%d张图片' % (i+1))
        return np.array(test_images), np.array(test_label)
        
    def extract(self, image_file):
        img = Image.open(image_file)
        # print("img", img)

        img = img.resize((32, 32))
        # img = img.resize((224, 224))

        # print("Image mode:", img.mode)
        if img.mode == 'L':
            img = img.convert('RGB')
        # img_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img_as_float(np.array(img))
        img = np.array(img)
        # print(img)
        img = np.transpose(img, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        img = (img - mean) / std
        return img
        return img
    
    def emnistDataDistribution(self, ):
        data_dir = r'./data/EMNIST'
        train_images_path = os.path.join(data_dir, 'emnist-balanced-train.csv')
        test_images_path = os.path.join(data_dir, 'emnist-balanced-test.csv.gz')
        import pandas as pd
        import numpy as np

        # 读取 CSV 文件
        train_images = pd.read_csv(train_images_path)
        test_images = pd.read_csv(train_images_path)
        # 提取标签
        train_labels = train_images.iloc[:, 0].values
        test_labels = test_images.iloc[:, 0].values
        # 提取图像数据并转换为 numpy 数组
        train_images = train_images.iloc[:, 1:].values
        train_images = train_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        train_images = np.reshape(train_images, (-1, 1, 28, 28))  # 将图像数据重新整形为 28x28 的数组

        test_images = test_images.iloc[:, 1:].values
        test_images = test_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        test_images = np.reshape(test_images , (-1,  1,  28, 28))  # 将图像数据重新整形为 28x28 的数组
        # 打印标签和图像的形状
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        self.train_data = train_images
        self.train_label = train_labels
        self.test_data = test_images
        self.test_label = test_labels
        print(self.train_data.shape)
        print(self.train_label.shape)

        balance_test_data = []
        balance_test_label = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(self.test_label.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.test_label.max() + 1):
            balance_test_data.append(self.test_data[class_index[number][:min_number]])
            balance_test_label += [number] * min_number
        print(min_number)
        self.test_data = np.concatenate(balance_test_data, axis=0)
        self.test_label = np.array(balance_test_label)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)
    
    def mnistDataDistribution(self, ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = 'data/MNIST/raw'
        data_dir = os.path.join(current_dir, data_dir)
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.train_data = train_images
        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)
        print(self.train_data.shape)
        balance_test_data = []
        balance_test_label = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(self.test_label.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.test_label.max() + 1):
            balance_test_data.append(self.test_data[class_index[number][:min_number]])
            balance_test_label += [number] * min_number

        self.test_data = np.concatenate(balance_test_data, axis=0)
        self.test_label = np.array(balance_test_label)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)

    def fashionmnistDataDistribution(self, ):
        print("执行了吗？")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = '../data/FashionMNIST/raw'
        data_dir = os.path.join(current_dir, data_dir)
        #data_dir = r'./data/FashionMNIST/raw'
        
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1

        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)


        self.train_data = train_images
        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)

        print("self.train_data.shape", self.train_data.shape)
        print("self.test_data.shape", self.test_data.shape)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)

    def save_splits(self, train_images, train_labels, test_images, test_labels, save_dir="data/federated_data/"):

        # **创建主目录**
        dir_path = os.path.join(save_dir, self.options['dataset_name'], f"nc{self.options['num_of_clients']}", f"dir{self.options['dirichlet']}")
        os.makedirs(dir_path, exist_ok=True)
        client_indices, _ = dirichlet_split_noniid(train_labels, self.options['dirichlet'], self.options['num_of_clients'])

        # 1️⃣ **划分训练数据**
        if self.options['pathe'] == True:
            dir_path = os.path.join(save_dir, self.options['dataset_name'], f"nc{self.options['num_of_clients']}", f"slice{self.options['slice']}", f"pathe")
            os.makedirs(dir_path, exist_ok=True)
            client_indices = shard_split_noniid(train_labels, self.options['num_of_clients'], self.options['slice'], self.options['num_classes'])
        # print("-------------", train_labels)
        # **存储训练数据到各个客户端文件夹**
        for client_id in range(self.options['num_of_clients']):
            client_dir = os.path.join(dir_path, f"client_{client_id+1}")
             # 如果客户端数据已存在，跳过该客户端
            if os.path.exists(os.path.join(client_dir, "train_data.npy")):
                print(f"Client {client_id+1}: Data already exists. Skipping.")
                continue
            os.makedirs(client_dir, exist_ok=True)

            indices = client_indices[client_id]
            train_data = np.column_stack((train_images[indices].reshape(len(indices), -1), train_labels[indices]))  # 合并数据和标签

            np.save(os.path.join(client_dir, "train_data.npy"), train_data)

            print(f"Client {client_id+1}: {len(indices)} training samples saved in {client_dir}")
        # 2️⃣ **存储测试数据到单独的文件夹**
        test_dir = os.path.join(dir_path, "test_data")
        os.makedirs(test_dir, exist_ok=True)

        test_file = os.path.join(test_dir, "test_data.npy")
        if os.path.exists(test_file):
            print(f"Test dataset already exists in {test_dir}, skip writing.")
        else:
            test_data = np.column_stack((test_images.reshape(len(test_images), -1), test_labels))  # 合并数据和标签
            np.save(os.path.join(test_dir, "test_data.npy"), test_data)
        print(f"Test dataset saved in {test_dir} with {len(test_images)} samples.")
   
    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def cifar10DataDistribution(self):
        cifar10_dir = 'data/cifar-10-batches-py'
        self.train_data, self.train_label, self.test_data, self.test_label = self.load_CIFAR10(cifar10_dir)
        
        print("self.train_data.shape", self.train_data.shape)
        print("self.test_data.shape", self.test_data.shape)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)
    
    def cifar100DataDistribution(self):
        cifar100_dir = 'data/cifar-100-python'
        # print(self.trainLabel)
        self.train_data, self.train_label, self.test_data, self.test_label = self.load_CIFAR100(cifar100_dir)
        self.save_splits(self.train_data, self.train_label, self.test_data, self.test_label)
      
    def load_CIFAR100(self, ROOT):
        # f = os.path.join(ROOT, )
        train_data = self.unpickle_cifar100(os.path.join(ROOT, 'train'))
        test_data = self.unpickle_cifar100(os.path.join(ROOT, 'test'))
        meta_data = self.unpickle_cifar100(os.path.join(ROOT, 'meta'))
                # 提取特征和标签
        X_train = train_data['data']
        y_train_fine = train_data['fine_labels']  # 细粒度类别
        y_train_coarse = train_data['coarse_labels']  # 粗糙类别
        X_test = test_data['data']
        y_test_fine = test_data['fine_labels']
        y_test_coarse = test_data['coarse_labels']

        print(X_train.shape) 
        # 将数据转换为合适的形状
        X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0, 1, 2, 3)
        X_train = X_train.astype(np.float32) 
        X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0, 1, 2, 3)
        X_test = X_test.astype(np.float32)
        print(X_train.shape) 
        # 类别名称
        fine_label_names = meta_data['fine_label_names']
        coarse_label_names = meta_data['coarse_label_names']  # 粗糙类别名称

        y_train_fine = np.array(y_train_fine, dtype=np.int64)
        y_train_coarse = np.array(y_train_coarse, dtype=np.int64)
        y_test_fine = np.array(y_test_fine, dtype=np.int64)
        y_test_coarse = np.array(y_test_coarse, dtype=np.int64)
        print(y_train_coarse)
        X_train = np.multiply(X_train, 1.0 / 255.0)
        X_test = np.multiply(X_test, 1.0 / 255.0)
        return X_train, y_train_coarse, X_test, y_test_coarse      

    def load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del xs, ys
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

        X_train = np.multiply(Xtr, 1.0 / 255.0)
        X_test = np.multiply(Xte, 1.0 / 255.0)
        # Resize images to 224x224

        # X_train = Xtr
        # X_test = Xte
        # X_train = torch.Tensor(Xtr).permute(0, 1, 2, 3) / 255.0
        # X_test = torch.Tensor(Xte).permute(0, 1, 2, 3) / 255.0
        return X_train, Ytr, X_test, Yte

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 1, 2, 3, ).astype("float32")

            Y = np.array(Y).astype("int64")
            return X, Y

    def load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))


    def unpickle_cifar100(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict      
# def load_client_data(client_id, base_dir="mnist_clients"):
#     client_dir = os.path.join(base_dir, f"client_{client_id}")
#     train_data = np.load(os.path.join(client_dir, "train_data.npy"))

#     images = train_data[:, :-1].reshape(-1, 1, 28, 28)  # 还原 28x28 形状
#     labels = train_data[:, -1]  # 获取标签
#     return images, labels

# # 读取 client_1 的数据
# train_images, train_labels = load_client_data(client_id=1)
# print(train_images.shape, train_labels.shape)

# def load_test_data(base_dir="data/fashionmnist/dir0.1"):
#     test_dir = os.path.join(base_dir, "test_data")
#     test_data = np.load(os.path.join(test_dir, "test_data.npy"))

#     images = test_data[:, :-1].reshape(-1, 1, 28, 28)
#     labels = test_data[:, -1]
#     return images, labels

# # 读取测试数据
# test_images, test_labels = load_test_data()
# print(test_images.shape, test_labels.shape)

def read_stl10_images(path):
    """
    返回: (N, 3, 96, 96), dtype=float32, range [0,1]
    """
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)

    # 每张图 3*96*96
    data = data.reshape(-1, 3, 96, 96)
    return data.astype(np.float32) / 255.0


def read_stl10_labels(path):
    """
    返回: (N,), label 范围 0-9
    """
    with open(path, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)

    # STL-10 标签是 1~10，10 表示类别 10
    labels[labels == 10] = 0
    return labels.astype(np.int64)


if __name__ == '__main__':
    options = {
        'dataset_name': 'tiny-imagenet-200',  # 或 'tinyimagenet' / 'TINYIMAGENET'
        'dirichlet': 0.3,
        'num_of_clients': 10,
    }
    ds = GetDataSet(options)
    print(ds.train_data.shape, ds.test_data.shape) 