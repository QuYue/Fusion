# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/26 17:29:41
@Author      :Qu Yue
@File        :main.py
@Software    :Visual Studio Code
Introduction: 
'''
#%% Import Packages
# %matplotlib qt5
import torch
import torchvision
import torch.utils.data as Data
import time 
import numpy as np
import matplotlib.pyplot as plt
import copy
import multiprocessing
from data_input import data_input, data_split, DATASET
import sys
sys.path.append('../..') # add the path which includes the packages
from FusionLearning.Plugin import Plugin
from FusionLearning import Fusion
from model import *
from process import *
from drawing import draw_result

#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.data = DATASET() 
        self.dataset_ID = 4
        self.test_size = 0.2
        self.epoch = 500
        self.batch_size = 50
        self.lr = 0.0001# 0.1
        self.draw = True
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1
        self.fusion_lr = 1e-12 # 0.000000000001
        self.Lambda = 0
        self.model =  FNN3
        self.time = dict()
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
        self.train_accuracy = {ID:[]}
        self.test_accuracy = {ID:[]}

#%% Tasks data
datasets = data_input(Parm)
Tasks = []
Train = [[],[]]
Test = [[],[]]
Origin = TASK('origin')
Fusion_task = TASK('fusion')
for i in range(Parm.task_number):
    task = TASK(i)
    data = data_split(datasets[i]['data'], datasets[i]['target'], 
                      Parm.test_size, random_state=Parm.random_seed)
    task.train = Data.TensorDataset(torch.stack(data[0][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[0][1]).type(torch.LongTensor))
    Train[0].extend(data[0][0])
    Train[1].extend(data[0][1])
    Test[0].extend(data[1][0])
    Test[1].extend(data[1][1])
    task.test = Data.TensorDataset(torch.stack(data[1][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[1][1]).type(torch.LongTensor))
    task.train_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=Parm.batch_size,
                                        shuffle=True)
    task.test_loader = Data.DataLoader(dataset=task.test, 
                                       batch_size=100,
                                       shuffle=False)
    Tasks.append(task)

#%% Create Models
Model = Parm.model
for i in range(Parm.task_number):
    Tasks[i].model = Model() if Parm.cuda==False else Model().cuda()
    Tasks[i].optimizer = torch.optim.Adam(Tasks[i].model.parameters(), lr=Parm.lr)

loss_func = torch.nn.CrossEntropyLoss()
#%%
num = 0
testing_process(Tasks[num], Parm)
if Parm.draw:
    fig = plt.figure(1)
    plt.ion()

for epoch in range(200):
    training_process(Tasks[num], loss_func, Parm)
    testing_process(Tasks[num], Parm)
    if Parm.draw:
        accuracy, name = [], []
        accuracy.append(Tasks[num].test_accuracy[Tasks[num].ID])
        name.append(f"Task{Tasks[num].ID}")
        draw_result(accuracy, fig, name, True)

#%%
if Parm.draw:
    plt.ioff()
    plt.show()