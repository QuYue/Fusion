# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/13 19:22
@Author  : QuYue
@File    : main.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torch
import numpy
import torchvision

#%%

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.5,), (1.0,))])
data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                        transform=trans,
                                        download=True)
data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                       transform=trans,
                                       download=True)