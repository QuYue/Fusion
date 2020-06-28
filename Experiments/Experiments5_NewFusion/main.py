# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/26 17:29:41
@Author      :Qu Yue
@File        :main.py
@Software    :Visual Studio Code
Introduction: 
'''
#%% Import Packages
import torch
import torchvision
import torch.utils.data as Datawwwwww
import numpy as np
import matplotlib.pyplot as plt
import copy
import multiprocessing
from data_input import data_input, data_split, DATASET
import sys
sys.path.append('../..') # add the path which includes the packages
import FusionLearning.Plugin
from model import *
from drawing import draw_result
%matplotlib auto
#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.data = DATASET() 
        self.dataset_ID = 1
        self.test_size = 0.2
        self.epoch = 100
        self.batch_size = 500
        self.lr = 0.1
        self.draw = True
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1
    @property
    def dataset_name(self):
        return self.data.data_dict [self.dataset_ID]
    @property
    def task_number(self):
        return self.data.tasks[self.dataset_name] 

Parm = PARM()
if Parm.cuda and torch.cuda.is_available():
    print('Using GPU.')
else:
    Parm.cuda = False
    print('Using CPU.')

torch.manual_seed(Parm.random_seed)
torch.cuda.manual_seed(Parm.random_seed)

#%% Task
class TASK:
    def __init__(self, ID=0):
        self.ID = ID
        self.train = []
        self.test = []
        self.train_accuracy = {ID:[]}
        self.test_accuracy = {ID:[]}

#%% Tasks data
datasets = data_input(Parm)
Tasks = []
for i in range(Parm.task_number):
    task = TASK(i)
    data = data_split(datasets[i]['data'], datasets[i]['target'], 
                      Parm.task_number, random_state=Parm.random_seed)
    task.train = Data.TensorDataset(torch.stack(data[0][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[0][1]).type(torch.LongTensor))
    task.test = Data.TensorDataset(torch.stack(data[1][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[1][1]).type(torch.LongTensor))
    task.train_loader = Data.DataLoader(dataset=task.train,
                               batch_size=Parm.batch_size,
                               shuffle=True)
    task.test_loader = Data.DataLoader(dataset=task.test, 
                               batch_size=1000,
                               shuffle=False)
    Tasks.append(task)

#%% Create Models
for i in range(Parm.task_number):
    Tasks[i].model = FNN1() if Parm.cuda==False else FNN1().cuda()
    Tasks[i].optimizer = torch.optim.SGD(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_model = FNN1() if Parm.cuda==False else FNN1().cuda()
loss_func = torch.nn.CrossEntropyLoss()


#%% Train
def training_process(Task, Parm):
    true_amount = 0; total_amount = 0
    for step, [x, y] in enumerate(Task.train_loader):
        if Parm.cuda:
            x = x.cuda()
            y = y.cuda()
        predict_y = Task.model(x)
        loss = loss_func(predict_y, y)
        Task.optimizer.zero_grad()
        loss.backward()
        Task.optimizer.step()
        true_amount += int(torch.sum(predict_y.argmax(1).data == y.data))
        total_amount += y.shape[0]
    train_accuracy = true_amount / total_amount
    Task.train_accuracy[Task.ID].append(train_accuracy)

def testing_process(Task, Parm):
    true_amount = 0; total_amount = 0
    for step, [x, y] in enumerate(Task.test_loader):
        if Parm.cuda:
            x = x.cuda()
            y = y.cuda()
        predict_y = Task.model(x)
        true_amount += int(torch.sum(predict_y.argmax(1).data == y.data))
        total_amount += y.shape[0]
    test_accuracy = true_amount / total_amount
    Task.test_accuracy[Task.ID].append(test_accuracy)

if Parm.draw:
    fig = plt.figure(1)
    plt.ion()

for epoch in range(Parm.epoch):
    training_process(Tasks[0], Parm)
    testing_process(Tasks[0], Parm)
    if Parm.draw:
        draw_result([Tasks[0].train_accuracy[0], Tasks[0].test_accuracy[0]], fig, ['train', 'test'], True)

if Parm.draw:
    plt.ioff()
    plt.show()
#%% 
