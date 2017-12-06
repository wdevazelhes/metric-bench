# Code taken from https://github.com/johny-c/pylmnn/blob/sklearn/util/dataset_fetcher.py

import os
import csv
import numpy as np
from scipy.io import loadmat
from sklearn.datasets import get_data_home, fetch_olivetti_faces, \
    fetch_mldata, load_iris
from collections import namedtuple

DataSet = namedtuple('Dataset', ['data', 'target'])


def fetch_letters(data_dir=None):
    path = os.path.join(get_data_home(data_dir), 'letter-recognition.data')

    if not os.path.exists(path):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        print('Downloading letter-recognition dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found letter-recognition in {}!'.format(path))

    y = np.loadtxt(path, dtype=str, usecols=(0), delimiter=',')
    X = np.loadtxt(path, usecols=range(1, 17), delimiter=',')

    return DataSet(X, y)


def decompress_z(fname_in, fname_out=None):
    from utils import unlzw
    fname_out = fname_in[:-2] if fname_out is None else fname_out
    print('Extracting {} to {}...'.format(fname_in, fname_out))
    with open(fname_in, 'rb') as fin, open(fname_out, 'wb') as fout:
        compressed_data = fin.read()
        uncompressed_data = unlzw.unlzw(compressed_data)
        fout.write(uncompressed_data)


def fetch_isolet(data_dir=None):

    if data_dir is None:
        data_dir = os.path.join(get_data_home(), 'isolet')

    train = 'isolet1+2+3+4.data.Z'
    test = 'isolet5.data.Z'
    path_train = os.path.join(get_data_home(data_dir), train)
    path_test = os.path.join(get_data_home(data_dir), test)

    if not os.path.exists(path_train[:-2]) or not os.path.exists(
            path_test[:-2]):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/'
        if not os.path.exists(path_train[:-2]):
            if not os.path.exists(path_train):
                print(
                    'Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                        url))
                request.urlretrieve(url=url + train, filename=path_train)
            # os.system('gzip -d ' + path_train)
            decompress_z(path_train)
        if not os.path.exists(path_test[:-2]):
            if not os.path.exists(path_test):
                print(
                    'Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                        url))
                request.urlretrieve(url=url + test, filename=path_test)
            # os.system('gzip -d ' + path_test)
            decompress_z(path_test)
    else:
        print('Found Isolated Letter Speech Recognition data set!')

    X_train, y_train = [], []
    with open(path_train[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row[:-1])
            y_train.append(int(float(row[-1])))

    labels, y_train = np.unique(y_train, return_inverse=True)

    X_test, y_test = [], []
    with open(path_test[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            X_test.append(row[:-1])
            y_test.append(int(float(row[-1])))

    labels, y_test = np.unique(y_test, return_inverse=True)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    return DataSet(X_train, y_train)


def fetch_usps(data_dir=None):

    if data_dir is None:
        data_dir = os.path.join(get_data_home(), 'usps')

    # base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
    base_url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/'

    train_file = 'zip.train.gz'
    test_file = 'zip.test.gz'

    if not os.path.isdir(data_dir):
        raise NotADirectoryError('{} is not a directory.'.format(data_dir))

    def download_file(source, destination):

        if not os.path.exists(destination):
            from urllib import request
            print('Downloading dataset from {}...'.format(source))
            f, msg = request.urlretrieve(url=source, filename=destination)
            print('HTTP response: {}'.format(msg))
        else:
            print('Found dataset in {}!'.format(destination))

    train_source = os.path.join(base_url, train_file)
    test_source = os.path.join(base_url, test_file)

    train_dest = os.path.join(data_dir, train_file)
    test_dest = os.path.join(data_dir, test_file)

    download_file(train_source, train_dest)
    download_file(test_source, test_dest)

    X_train = np.loadtxt(train_dest)
    y_train, X_train = X_train[:, 0].astype(np.int32), X_train[:, 1:]

    X_test = np.loadtxt(test_dest)
    y_test, X_test = X_test[:, 0].astype(np.int32), X_test[:, 1:]

    return DataSet(X_train, y_train)


def fetch_mnistPCA(data_dir=None):

    path = os.path.join(get_data_home(data_dir), 'mnistPCA.mat')
    if not os.path.exists(path):
        from urllib import request
        url = 'https://dl.dropboxusercontent.com/u/4284723/DATA/mnistPCA.mat'
        print('Downloading mnistPCA dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found mnistPCA.mat in {}!'.format(path))

    mnist_mat = loadmat(path)

    X_train = np.asarray(mnist_mat['xTr'], dtype=np.float64).T
    X_test = np.asarray(mnist_mat['xTe'], dtype=np.float64).T
    y_train = np.asarray(mnist_mat['yTr'], dtype=np.int).ravel()
    y_test = np.asarray(mnist_mat['yTe'], dtype=np.int).ravel()

    return DataSet(X_train, y_train)


def fetch_mnist_deskewed(data_dir=None):

    MNIST_DESKEWED_URL = 'https://www.dropbox.com/s/mhsnormwt5i2ba6/mnist-deskewed-pca164.mat?dl=1'
    MNIST_DESKEWED_PATH = os.path.join(get_data_home(data_dir),
                                       'mnist-deskewed-pca164.mat')

    if not os.path.exists(MNIST_DESKEWED_PATH):
        from urllib import request
        print('Downloading deskewed MNIST from {} . . .'.format(
            MNIST_DESKEWED_URL), end='')
        request.urlretrieve(MNIST_DESKEWED_URL, MNIST_DESKEWED_PATH)
        print('done.')

    mnist_mat = loadmat(MNIST_DESKEWED_PATH)

    X_train = np.asarray(mnist_mat['X_train'], dtype=np.float64)
    X_test = np.asarray(mnist_mat['X_test'], dtype=np.float64)
    y_train = np.asarray(mnist_mat['y_train'], dtype=np.int).ravel()
    y_test = np.asarray(mnist_mat['y_test'], dtype=np.int).ravel()

    print('Loaded deskewed MNIST from {}.'.format(MNIST_DESKEWED_PATH))

    return DataSet(X_train, y_train)


def fetch_balance(data_dir=None):
    path = os.path.join(get_data_home(data_dir), 'balance.data')

    if not os.path.exists(path):
        from urllib import request
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
        print('Downloading balance dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found balance in {}!'.format(path))
    DataSet = namedtuple('Dataset', ['data', 'target'])
    y = np.loadtxt(path, dtype=str, usecols=(0), delimiter=',')
    X = np.loadtxt(path, usecols=range(1, 5), delimiter=',')
    balance = DataSet(X, y)
    return balance
