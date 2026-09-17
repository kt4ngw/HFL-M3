# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
# IDX/CIFAR reading helpers adapted from the TensorFlow MNIST tutorial (Apache 2.0)
# and Stanford CS231n data_utils.py.
import gzip
import os
import pickle
import platform

import numpy as np

from src.utils.dirichlet import dirichlet_split_noniid, shard_split_noniid


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

        name = str(self.dataSetName).lower()
        if name == 'fashionmnist':
            self.fashionmnistDataDistribution()
        elif name == 'cifar10':
            self.cifar10DataDistribution()
        elif name == 'cifar100':
            self.cifar100DataDistribution()
        else:
            raise ValueError(
                f"unsupported dataset_name {self.dataSetName!r}; "
                "expected fashionmnist, cifar10 or cifar100"
            )

    def fashionmnistDataDistribution(self, ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = '../data/FashionMNIST/raw'
        data_dir = os.path.join(current_dir, data_dir)
        #data_dir = r'./data/FashionMNIST/raw'

        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

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

        dir_path = os.path.join(save_dir, self.options['dataset_name'], f"nc{self.options['num_of_clients']}", f"dir{self.options['dirichlet']}")
        os.makedirs(dir_path, exist_ok=True)
        client_indices, _ = dirichlet_split_noniid(train_labels, self.options['dirichlet'], self.options['num_of_clients'])

        if self.options['pathe'] == True:
            dir_path = os.path.join(save_dir, self.options['dataset_name'], f"nc{self.options['num_of_clients']}", f"slice{self.options['slice']}", f"pathe")
            os.makedirs(dir_path, exist_ok=True)
            client_indices = shard_split_noniid(train_labels, self.options['num_of_clients'], self.options['slice'], self.options['num_classes'])
        # print("-------------", train_labels)
        for client_id in range(self.options['num_of_clients']):
            client_dir = os.path.join(dir_path, f"client_{client_id+1}")
            if os.path.exists(os.path.join(client_dir, "train_data.npy")):
                print(f"Client {client_id+1}: Data already exists. Skipping.")
                continue
            os.makedirs(client_dir, exist_ok=True)

            indices = client_indices[client_id]
            train_data = np.column_stack((train_images[indices].reshape(len(indices), -1), train_labels[indices]))

            np.save(os.path.join(client_dir, "train_data.npy"), train_data)

            print(f"Client {client_id+1}: {len(indices)} training samples saved in {client_dir}")
        test_dir = os.path.join(dir_path, "test_data")
        os.makedirs(test_dir, exist_ok=True)

        test_file = os.path.join(test_dir, "test_data.npy")
        if os.path.exists(test_file):
            print(f"Test dataset already exists in {test_dir}, skip writing.")
        else:
            test_data = np.column_stack((test_images.reshape(len(test_images), -1), test_labels))
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
        X_train = train_data['data']
        y_train_fine = train_data['fine_labels']
        y_train_coarse = train_data['coarse_labels']
        X_test = test_data['data']
        y_test_fine = test_data['fine_labels']
        y_test_coarse = test_data['coarse_labels']

        print(X_train.shape)
        X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0, 1, 2, 3)
        X_train = X_train.astype(np.float32)
        X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0, 1, 2, 3)
        X_test = X_test.astype(np.float32)
        print(X_train.shape)
        fine_label_names = meta_data['fine_label_names']
        coarse_label_names = meta_data['coarse_label_names']

        y_train_fine = np.array(y_train_fine, dtype=np.int64)
        y_train_coarse = np.array(y_train_coarse, dtype=np.int64)
        y_test_fine = np.array(y_test_fine, dtype=np.int64)
        y_test_coarse = np.array(y_test_coarse, dtype=np.int64)
        print(y_train_coarse)
        X_train = np.multiply(X_train, 1.0 / 255.0)
        X_test = np.multiply(X_test, 1.0 / 255.0)

        # Standardize CIFAR-100 before federated client/test arrays are saved.
        # Both training and test data use statistics from the CIFAR-100
        # training set. The saved .npy files are therefore model-ready and
        # must not be normalized again in the client or evaluation pipeline.
        mean = np.array(
            [0.5071, 0.4867, 0.4408], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        std = np.array(
            [0.2675, 0.2565, 0.2761], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
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

        # Standardize CIFAR-10 before federated client/test arrays are saved.
        # Both splits use the fixed statistics of the CIFAR-10 training set.
        mean = np.array(
            [0.4914, 0.4822, 0.4465], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        std = np.array(
            [0.2470, 0.2435, 0.2616], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
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
