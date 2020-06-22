# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/13 20:18
@Author  : QuYue
@File    : data_input.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torchvision
from sklearn.model_selection import train_test_split

#%% Functions
def data_input(dataset_no=1, download=True):
    # Input Data
    if dataset_no == 1 or 'MNIST':
        dataset_no = 1
        print('Loading dataset 1: MNIST...')
        data_train, data_test = MNIST_input(download)
    else:
        print('Please input right dataset number.')
        return None, None
    print('Data is ready.')
    return data_train, data_test


def MNIST_input(download=True):
    # Data1 MNIST
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
    return data_train, data_test

def data_split(dataset, dataset_no=1):
    # Data Split
    if dataset_no == 1 or 'MNIST':
        dataset1 = []
        dataset2 = []
        for i in dataset:
            if i[1] < 5:
                dataset1.append(i)
            else:
                dataset2.append(i)
    else:
        print('Please input right dataset number.')
        return None, None
    return dataset1, dataset2






