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
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import copy
import multiprocessing
from data_input import data_input, data_split, DATASET
import sys
sys.path.append('../..') # add the path which includes the packages
import FusionLearning.Plugin
from model import *
# from drawing import draw_result

#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.data = DATASET() 
        self.dataset_ID = 1
        self.test_size = 0.2
        self.epoch = 10
        self.batch_size = 500
        self.lr = 0.1
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1
    @property
    def dataset_name(self):
        return self.data.data_dict[self.dataset_ID]
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
        self.accuracy = {ID:[]}

datasets = data_input(Parm)
#%% Tasks data
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
for epoch in range(100):
    for step, [x, y] in enumerate(Tasks[0].train_loader):
        if Parm.cuda:
            x = x.cuda()
            y = y.cuda()
        predict_y = Tasks[0].model(x)
        loss = loss_func(predict_y, y)
        Tasks[0].optimizer.zero_grad()
        loss.backward()
        Tasks[0].optimizer.step()
        true = int(torch.sum(predict_y.argmax(1).data == y.data))
        amount = y.shape[0]
        accuracy = true / amount
        print(f'Epoch: {epoch}| Step: {step}| Batch Accuracy: {(accuracy*100):.2f}%') 
#%% 
