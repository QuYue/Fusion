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

#%% Functions
def data_input(dataset_no=1, download=True):
    if dataset_no == 1:
        print('Loading dataset 1: MNIST...')
        data_train, data_test = MNIST_input(download)
    print('Data is ready.')
    return data_train, data_test


def MNIST_input(download=True):
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (1.0,))])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                            transform=trans,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                           transform=trans,
                                           download=download)
    return data_train, data_test
