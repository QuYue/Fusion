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
import matplotlib.pyplot as plt
from data_input import data_input

#%% Data Input
data_train, data_test = data_input('MNIST', download=False)

#%%
for data, label in  data_train:
    pass

