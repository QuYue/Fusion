# -*- encoding: utf-8 -*-
'''
@Time        :2020/07/19 16:16:39
@Author      :Qu Yue
@File        :main_cnn.py
@Software    :Visual Studio Code
Introduction:  
'''
#%% Import Packages
# %matplotlib qt5
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
from FusionLearning.Plugin import Plugin
from FusionLearning import Fusion
from model import *
from process import *
from drawing import draw_result

#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.data = DATASET() 
        self.dataset_ID = 1
        self.test_size = 0.2
        self.epoch = 10
        self.batch_size = 500
        self.lr = 0.1
        self.draw = True
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1
        self.fusion_lr = 1e-12 # 0.000000000001
        self.Lambda = 0
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
Test = [[],[]]
for i in range(Parm.task_number):
    task = TASK(i)
    data = data_split(datasets[i]['data'], datasets[i]['target'], 
                      Parm.test_size, random_state=Parm.random_seed)
    task.train = Data.TensorDataset(torch.stack(data[0][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[0][1]).type(torch.LongTensor))
    Test[0].extend(data[1][0])
    Test[1].extend(data[1][1])
    task.test = Data.TensorDataset(torch.stack(data[1][0]).type(torch.FloatTensor), 
                                    torch.tensor(data[1][1]).type(torch.LongTensor))
    task.train_loader = Data.DataLoader(dataset=task.train,
                                        batch_size=Parm.batch_size,
                                        shuffle=True)
    task.test_loader = Data.DataLoader(dataset=task.test, 
                                        batch_size=1000,
                                        shuffle=False)
    Tasks.append(task)

Test = Data.TensorDataset(torch.stack(Test[0]).type(torch.FloatTensor), 
                            torch.tensor(Test[1]).type(torch.LongTensor))
Test_loader = Data.DataLoader(dataset=Test, 
                              batch_size=1000,
                              shuffle=False)

#%% Create Models
for i in range(Parm.task_number):
    Tasks[i].model = CNN1() if Parm.cuda==False else CNN1().cuda()
    Tasks[i].optimizer = torch.optim.SGD(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_model = CNN1() if Parm.cuda==False else CNN1().cuda()
loss_func = torch.nn.CrossEntropyLoss()

#%% Train
if Parm.draw:
    fig = plt.figure(1)
    plt.ion()

for epoch in range(Parm.epoch):
    for task in Tasks:
        training_process(task, loss_func, Parm)
        testing_process(task, Parm)
    if Parm.draw:
        accuracy, name = [], []
        for task in Tasks:
            accuracy.append(task.test_accuracy[task.ID])
            name.append(f"Task{task.ID}")
        draw_result(accuracy, fig, name, True)

#%% Adding Plugin
print('Before')
for i in range(Parm.task_number):
    Tasks[i].model0 = Tasks[i].model
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model))
    Tasks[i].model.plugin_hook()
    print(f"Accuracy: {testing_free(Tasks[i], Tasks[i].test_loader, Parm)}")
    print(f"Total Accuracy: {testing_free(Tasks[i], Test_loader, Parm)}")
fusion_model = Plugin(fusion_model)
#%% Average
print('Average Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
models = [Task.model for Task in Tasks]
fusion_model = Fusion.average_fusion(models, fusion_model)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")

#%%
print('Pinv Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
    #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
    #Tasks[i].model.oneshot_rank(Parm)
fusion_model = Fusion.pinv_fusion(Tasks, fusion_model, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")


# %%
if Parm.draw:
    plt.ioff()
    plt.show()