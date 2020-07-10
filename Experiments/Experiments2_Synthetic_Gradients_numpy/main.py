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

#%%
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.dataset = 'MNIST'
        self.epoch = 30
        self.batch_size = 500
        self.lr = 0.0001
        self.lr_sg = 0.0001
        self.cuda = False
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
task1.model = FNN2_DNI(Parm.lr, Parm.lr_sg)
task2.model = FNN2_DNI(Parm.lr, Parm.lr_sg)
loss_func = MSEloss
#%%
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

def get_accuracy(output, label):
    output = output.argmax(1)
    a = np.sum(label.numpy() == output)
    return a

def training_process(task, Parm):
    for step, (data, label) in enumerate(task.train_loader):
        batch_x = torch_numpy_data(data)
        batch_y = torch_numpy_label(label)

        output = task.model.forward(batch_x)
        loss, loss_deriv = loss_func(output, batch_y)
        task.model.backward(loss_deriv)
        task.model.update(Parm.nowepoch)

        true = get_accuracy(output, label)
        amount = label.shape[0]
        accuracy = true / amount
        print(f'Epoch: {epoch}| Step: {step}| Batch Accuracy: {(accuracy*100):.2f}%')

def training_process2(model, train_loader1, train_loader2):
    def training(train_loader, num):
        true = 0
        amount = 0
        for step, (data, label) in enumerate(train_loader):
            batch_x = torch_numpy_data(data)
            batch_y = torch_numpy_label(label)
            output = model.fast_forward(batch_x, num)
            true += get_accuracy(output, label)
            amount += label.shape[0]
            print(f'Epoch: {epoch}| Step: {step}| task{num+1} Batch Accuracy: {(true/amount * 100):.2f}%')
    training(train_loader1, 0)
    training(train_loader2, 1)

def testing_process(task, Parm):
    true = 0
    amount = 0
    for data, label in task.test_loader:
        batch_x = torch_numpy_data(data)
        batch_y = torch_numpy_label(label)
        output = task.model.forward(batch_x)
        true += get_accuracy(output, label)
        amount += label.shape[0]
    task.accuracy.append(true/amount)
    print(f'Epoch: {epoch}| Train Accuracy: {(task.accuracy[-1]*100):.2f}%')

def testing_process2(model, Parm, test_loader):
    true = 0
    amount = 0
    for data, label in test_loader:
        batch_x = torch_numpy_data(data)
        batch_y = torch_numpy_label(label)
        output = model.forward(batch_x)
        true += get_accuracy(output, label)
        amount += label.shape[0]
    return true/amount

fig = plt.figure(1)
plt.ion()

for epoch in range(Parm.epoch):
    Parm.nowepoch = epoch
    print("Task 1:############################")
    training_process(task1, Parm)
    if epoch % Parm.showepoch == 0:
        testing_process(task1, Parm)
        task1.accuracy2.append(testing_process2(task1.model, Parm, task2.test_loader))

    print("Task 2:############################")
    training_process(task2, Parm)
    if epoch % Parm.showepoch == 0:
        testing_process(task2, Parm)
        task2.accuracy2.append(testing_process2(task2.model, Parm, task1.test_loader))

    draw_result([task1.accuracy, task2.accuracy, task1.accuracy2, task2.accuracy2],
                fig, ['Model1 on task1', 'Model2 on task2', 'Model1 on task2', 'Model2 on task1'], True)
    plt.ioff()
    plt.show()
#%%
accuracy1 = []
accuracy2 = []
fusion_model = FNN2_DNI(Parm.lr, Parm.lr_sg)
fusion_model = par_fusion([task1.model, task2.model], fusion_model)
accuracy1.append(testing_process2(fusion_model, Parm, task1.test_loader))
accuracy2.append(testing_process2(fusion_model, Parm, task2.test_loader))
print("Fusion :############################")
for epoch in range(Parm.epoch):
    training_process2(fusion_model, task1.train_loader, task2.train_loader)
    if epoch % Parm.showepoch == 0:
        accuracy1.append(testing_process2(fusion_model, Parm, task1.test_loader))
        accuracy2.append(testing_process2(fusion_model, Parm, task2.test_loader))
    draw_result([task1.accuracy, task2.accuracy, task1.accuracy2, task2.accuracy2, accuracy1, accuracy2],fig,
                ['Model1 on task1', 'Model2 on task2', 'Model1 on task2', 'Model2 on task1', 'Fusion on task1', 'Fusion on task2'], True)

    plt.ioff()
    plt.show()