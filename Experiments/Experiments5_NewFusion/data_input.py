 # -*- coding: utf-8 -*-
"""
@Time    : 2020/6/22 22:21
@Author  : QuYue
@File    : data_input.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torchvision
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
# %matplotlib auto
#%% Functions
def data_input(dataset_no=1, download=True):
    """Input the data.
    Input: 
        dataset_no: Data set number [1, 2, 3]
        download: If download the data set [True, False]
    Output:
        datasets
    """
    # Input Data
    if dataset_no == 1 or dataset_no.lower() == 'disjoint mnist':
        dataset_no = 1
        print('Loading dataset 1: Disjoint MNIST...')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], dataset_no)
    elif dataset_no == 2 or dataset_no.lower() == 'split mnist':
        dataset_no = 2
        print('Loading dataset 2: Split MNIST...')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], dataset_no)
    elif dataset_no == 3 or dataset_no.lower() == 'permuted mnist':
        dataset_no = 3
        print('Loading dataset 3: Permuted MNIST')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], dataset_no)
    else:
        print('Please input right dataset number.')
        return None
    print('Data is ready.')
    return datasets

def MNIST_input(download=True):
    """Input MNIST data.
    Input:
        download: If download the data set [True, False]
    Output:
        mnist_data
        mnist_target
    """
    def data_combine(data_train, data_test):
        data = []; target = []
        for d,t in data_train:
            data.append(d), target.append(t)
        for d,t in data_test:
            data.append(d), target.append(t)
        return data, target
    # Data MNIST
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (1.0,))])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                            train=True,
                                            transform=trans,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                           train=False,
                                           transform=trans,
                                           download=download)
    # Data Combine
    mnist_data, mnist_target = data_combine(data_train, data_test)
    return mnist_data, mnist_target

def task_split(dataset, dataset_no=1):
    # Task Split
    datasets = []
    if dataset_no == 1:
        dataset1, dataset2 = [], []
        for i in range(len(dataset[1])):
            if dataset[1][i] < 5:
                dataset1.append(i)
            else:
                dataset2.append(i)
        datasets.append([dataset[0][dataset1], dataset[1][dataset1]])
        datasets.append([dataset[0][dataset2], dataset[2][dataset1]])
    else:
        print('Please input right dataset number.')
        return None
    return datasets

def data_split(data, target, test_size=0.2, random_state=0):
    # Data split to train and test
    X_train,X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return [[X_train, y_train], [X_test, y_test]]

if __name__ == "__main__":
    datasets = data_input(dataset_no=1, download=True)