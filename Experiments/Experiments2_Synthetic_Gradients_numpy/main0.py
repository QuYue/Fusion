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
import numpy as np
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from data_input import data_input, data_split
from model import *
from drawing import draw_result
import multiprocessing
import sys

#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.dataset = 'MNIST'
        self.epoch = 1000
        self.batch_size = 500
        self.lr =    0.0001
        self.lr_sg = 0.0001
        self.cuda = True
        self.showepoch = 1

Parm = PARM()
if Parm.cuda:
    print('Using GPU.')
else:
    print('Using CPU.')

#%%
class TASK:
    def __init__(self):
        self.train = []
        self.test = []
        self.accuracy = []
        self.accuracy2 = []
#%% Data Input
data_train, data_test = data_input(Parm.dataset, download=False)
task1, task2 = TASK(), TASK()
task1.train, task2.train = data_split(data_train, Parm.dataset)
task1.test, task2.test = data_split(data_test, Parm.dataset)

task1.train_loader = Data.DataLoader(dataset=task1.train,
                               batch_size=Parm.batch_size,
                               shuffle=True)
task1.test_loader = Data.DataLoader(dataset=task1.test,
                               batch_size=1000,
                               shuffle=False)
task2.train_loader = Data.DataLoader(dataset=task2.train,
                               batch_size=Parm.batch_size,
                               shuffle=True)
task2.test_loader = Data.DataLoader(dataset=task2.test,
                               batch_size=1000,
                               shuffle=False)

#%% Create Model

def torch_numpy_data(data):
    data_np = data.reshape(data.shape[0], -1)
    data_np = data_np.numpy()
    return data_np

def torch_numpy_label(label):
    def index2vector(label_np, dim=10):
        vector = np.zeros((label_np.shape[0], dim))
        for i in range(label_np.shape[0]):
            vector[i, label_np[i]] = 1
        return vector
    label_np = label.numpy()
    label_np = index2vector(label_np)
    return label_np
def accuracy(output, label):
    output = output.argmax(1)
    a = np.sum(label.numpy() == output)
    return a


np.random.seed(1)


alpha = Parm.lr

input_dim = 28 * 28
layer_1_dim = 1000
layer_2_dim = 100
output_dim = 10

# layer_1 = Layer(input_dim,layer_1_dim,relu,relu_out2deriv)
# layer_2 = Layer(layer_1_dim,layer_2_dim,relu,relu_out2deriv)
# layer_3 = Layer(layer_2_dim, output_dim,relu, relu_out2deriv)
layer_1 = DNI_Layer(input_dim,layer_1_dim,relu,relu_out2deriv,Parm.lr, Parm.lr_sg)
layer_2 = DNI_Layer(layer_1_dim,layer_2_dim,relu,relu_out2deriv,Parm.lr, Parm.lr_sg)
layer_3 = DNI_Layer(layer_2_dim, output_dim,relu,relu_out2deriv,Parm.lr, Parm.lr_sg)




for epoch in range(Parm.epoch):
#%% Initialization
    error = 0
    synthetic_error3 = 0
    synthetic_error2 = 0
    synthetic_error1 = 0
    right = 0
#%% Training
    for step, (data, label) in enumerate(task1.train_loader):
        batch_x = torch_numpy_data(data)
        batch_y = torch_numpy_label(label)

        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.backward(layer_3_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)

        if epoch<=10:
            layer_1.update()
            layer_2.update()
            layer_3.update()
        else:
            layer_1.update0()
            layer_2.update0()
            layer_3.update0()

        #
        right += accuracy(layer_3_out, label)
        error += np.linalg.norm(layer_3_out-batch_y, 'fro')
        synthetic_error3 += np.linalg.norm(layer_3_delta - layer_3.synthetic_gradient, 'fro')
        synthetic_error2 += np.linalg.norm(layer_2_delta - layer_2.synthetic_gradient, 'fro')
        synthetic_error1 += np.linalg.norm(layer_1_delta - layer_1.synthetic_gradient, 'fro')

    print(f"Iter: {epoch} | Loss: {error} | Accuracy: {right/len(task1.train)} | Synthetic_loss3: {synthetic_error3} | Synthetic_loss2: {synthetic_error2} | Synthetic_loss1: {synthetic_error1}")
#%%




