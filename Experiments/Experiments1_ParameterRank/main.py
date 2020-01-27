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
from data_input import data_input
#%%
data_train, data_test = data_input(1, download=False)
