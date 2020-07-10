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
import copy

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

#%% Task
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
#
# task1.model = CNN()
# task2.model = CNN()
# fusion_model = CNN()
#
# task1.model = FNN()
# task2.model = FNN()
# fusion_model = FNN()
#
task1.model = FNN2()
task2.model = FNN2()
fusion_model = FNN2()

if Parm.cuda:
    task1.model = task1.model.cuda()
    task2.model = task2.model.cuda()
    fusion_model = fusion_model.cuda()

task1.optimizer = torch.optim.SGD(task1.model.parameters(), lr=Parm.lr)
task2.optimizer = torch.optim.SGD(task2.model.parameters(), lr=Parm.lr)
loss_func = torch.nn.CrossEntropyLoss()

#%%
def training_process(task, Parm):
    task.model.train()
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
    task.model.eval()
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
    model.eval()
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

    draw_result([task1.accuracy, task2.accuracy, task1.accuracy2, task2.accuracy2],
                fig, ['Model1 on task1', 'Model2 on task2', 'Model1 on task2', 'Model2 on task1'], True)
    plt.ioff()
    plt.show()

#%%
print("Fusion 1: Average ############################")
fusion_model1 = FNN2()
if Parm.cuda:
    fusion_model1 = fusion_model1.cuda()
fusion_model1 = par_fusion([task1.model, task2.model], fusion_model1)
fusion_model1.eval()
t1 = testing_process2(task1.model, Parm, task1.test_loader)
t2 = testing_process2(task2.model, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

t1 = testing_process2(fusion_model1, Parm, task1.test_loader)
t2 = testing_process2(fusion_model1, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

#%%
print("Fusion 2: Oneshot Rank Average ############################")
fusion_model2 = FNN2()
if Parm.cuda:
    fusion_model2 = fusion_model2.cuda()

model1, model2 = copy.deepcopy(task1.model), copy.deepcopy(task2.model)
model1 = oneshot_rank(model1, Parm)
model2 = oneshot_rank(model2, Parm)

fusion_model2 = par_fusion([model1, model2], fusion_model2)
fusion_model2.eval()
t1 = testing_process2(model1, Parm, task1.test_loader)
t2 = testing_process2(model2, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

t1 = testing_process2(fusion_model2, Parm, task1.test_loader)
t2 = testing_process2(fusion_model2, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

#%%
print("Fusion 3: Linear Regression Average ############################")
def fusion_parm(fusion_model, task1, task2, Parm):
    fusion_model3 = copy.deepcopy(fusion_model)
    for i in range(300):
        for step,((data1, label1), (data2, label2) )in enumerate(zip(task1.test_loader, task2.test_loader)):
            break
        if Parm.cuda:
            data1 = data1.cuda()
            data2 = data2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

        fusion_model3 = par_fusion3(task1.model, task2.model, fusion_model3, data1, data2, step=0.000000000001)
        output = fusion_model3(data1)
        true = int(torch.sum(output.argmax(1).data == label1.data))
        amount = label1.shape[0]
        t1 = true/amount
        output = fusion_model3(data2)
        true = int(torch.sum(output.argmax(1).data == label2.data))
        amount = label2.shape[0]
        t2 = true/amount
        t1 = testing_process2(fusion_model3, Parm, task1.test_loader)
        t2 = testing_process2(fusion_model3, Parm, task2.test_loader)
        print(f"step{step}, t1:{t1:.4f}, t2:{t2:.4f}")
    return fusion_model3
fusion_model3 = fusion_parm(fusion_model2, task1, task2, Parm)
t1 = testing_process2(model1, Parm, task1.test_loader)
t2 = testing_process2(model2, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

t1 = testing_process2(fusion_model3, Parm, task1.test_loader)
t2 = testing_process2(fusion_model3, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")
#%%
print("Fusion 4: Linear Regression Oneshot Rank Average ############################")
fusion_model4 = fusion_parm(fusion_model2, task1, task2, Parm)
#%%
print("Fusion 5: Pinverse ############################")
for step, ((data1, label1), (data2, label2)) in enumerate(zip(task1.test_loader, task2.test_loader)):
    break
if Parm.cuda:
    data1 = data1.cuda()
    data2 = data2.cuda()
    label1 = label1.cuda()
    label2 = label2.cuda()
fusion_model5 = par_fusion2(task1.model, task2.model, copy.deepcopy(fusion_model1), data1, data2)
fusion_model5.eval()
t1 = testing_process2(model1, Parm, task1.test_loader)
t2 = testing_process2(model2, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

t1 = testing_process2(fusion_model5, Parm, task1.test_loader)
t2 = testing_process2(fusion_model5, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")
#%%
print("Fusion 6: Pinverse Oneshot Rank ############################")
for step, ((data1, label1), (data2, label2)) in enumerate(zip(task1.test_loader, task2.test_loader)):
    break
if Parm.cuda:
    data1 = data1.cuda()
    data2 = data2.cuda()
    label1 = label1.cuda()
    label2 = label2.cuda()
#%%
model1 = copy.deepcopy(task1.model)
model2 = copy.deepcopy(task2.model)
model1 = oneshot_rank(model1, Parm)
model2 = oneshot_rank(model2, Parm)
fusion_model6 = par_fusion2(model1, model2, copy.deepcopy(fusion_model1), data1, data2)
fusion_model6.eval()
t1 = testing_process2(model1, Parm, task1.test_loader)
t2 = testing_process2(model2, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")

t1 = testing_process2(fusion_model6, Parm, task1.test_loader)
t2 = testing_process2(fusion_model6, Parm, task2.test_loader)
print(f"t1:{t1}, t2:{t2}")