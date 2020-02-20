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

#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.dataset = 'MNIST'
        self.epoch = 10
        self.batch_size = 500
        self.lr = 0.1
        self.cuda = True
        self.showepoch = 1

Parm = PARM()
if Parm.cuda:
    print('Using GPU.')
else:
    print('Using CPU.')

torch.manual_seed(1)
torch.cuda.manual_seed(1)

#%%
class TASK:
    def __init__(self):
        self.train = []
        self.test = []
        self.accuracy = []
        self.accuracy2 = []
#%% Data Input
data_train, data_test = data_input(Parm.dataset, download=True)
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
#
# task1.model = CNN()
# task2.model = CNN()
# fusion_model = CNN()
#
task1.model = FNN2()
task2.model = FNN2()
fusion_model = FNN2()
#
# task1.model = FNN()
# task2.model = FNN()
# fusion_model = FNN()

if Parm.cuda:
    task1.model = task1.model.cuda()
    task2.model = task2.model.cuda()
    fusion_model = fusion_model.cuda()

task1.optimizer = torch.optim.SGD(task1.model.parameters(), lr=Parm.lr)
task2.optimizer = torch.optim.SGD(task2.model.parameters(), lr=Parm.lr)
loss_func = torch.nn.CrossEntropyLoss()

#%%
def training_process(task, Parm):
    for step, (data, label) in enumerate(task.train_loader):
        if Parm.cuda:
            data = data.cuda()
            label = label.cuda()
        output = task.model(data)
        loss = loss_func(output, label)
        task.optimizer.zero_grad()
        loss.backward()
        task.optimizer.step()
        true = int(torch.sum(output.argmax(1).data == label.data))
        amount = label.shape[0]
        accuracy = true / amount
        print(f'Epoch: {epoch}| Step: {step}| Batch Accuracy: {(accuracy*100):.2f}%')
def testing_process(task, Parm):
    true = 0
    amount = 0
    for data, label in task.test_loader:
        if Parm.cuda:
            data = data.cuda()
            label = label.cuda()
        output = task.model(data)
        true += int(torch.sum(output.argmax(1).data == label.data))
        amount += label.shape[0]
    task.accuracy.append(true/amount)
    print(f'Epoch: {epoch}| Train Accuracy: {(task.accuracy[-1]*100):.2f}%')

def testing_process2(model, Parm, test_loader):
    true = 0
    amount = 0
    for data, label in test_loader:
        if Parm.cuda:
            data = data.cuda()
            label = label.cuda()
        output = model(data)
        true += int(torch.sum(output.argmax(1).data == label.data))
        amount += label.shape[0]
    return true/amount

fig = plt.figure(1)
plt.ion()

accuracy1 = []
accuracy2 = []

for epoch in range(Parm.epoch):
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

    print("Fusion:############################")
    fusion_model = par_fusion([task1.model, task2.model], fusion_model)
    if epoch % Parm.showepoch == 0:
        accuracy1.append(testing_process2(fusion_model, Parm, task1.test_loader))
        accuracy2.append(testing_process2(fusion_model, Parm, task2.test_loader))

    draw_result([task1.accuracy, task2.accuracy, task1.accuracy2, task2.accuracy2, accuracy1, accuracy2],
                fig, ['Model1 on task1', 'Model2 on task2', 'Model1 on task2', 'Model2 on task1', 'Fusion on task1', 'Fusion on task2'], True)
    plt.ioff()
    plt.show()

#%%


