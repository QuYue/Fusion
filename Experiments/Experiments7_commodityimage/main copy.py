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
        self.dataset_ID = 2
        self.test_size = 0.2
        self.epoch = 500
        self.batch_size = 5000
        self.lr = 0.1# 0.1
        self.draw = True
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1
        self.fusion_lr = 1e-12 # 0.000000000001
        self.Lambda = 0
        self.model =  FNN1
        self.time = dict()
    @property
    def dataset_name(self):
        return self.data.data_dict[self.dataset_ID]
    @property
    def task_number(self):
        return self.data.tasks[self.dataset_name] 

    def lambda_list(self, ifzero=False):
        if self.model == FNN1:
            if ifzero == False:
                lambda_list = [[0.48, 0.52],[0.33, 0.5, 0.5, 0.4, 0.57],[0.5, 0.48, 0.6]]
            else:
                lambda_list = [[0.5, 0.6],[0.35, 0.5, 0.5, 0.4, 0.6],[0.51, 0.5, 0.51]]

        elif self.model == CNN1:
            if ifzero == False:
                lambda_list = [[0.54, 0.5],[0.57, 0.50, 0.5, 0.5, 0.51],[0.5,0.5,0.5]]
            else:
                lambda_list = [[0.5, 0.58],[0.62,0.5,0.5,0.7,0.5],[0.5,0.5,0.5]]
        return lambda_list[self.dataset_ID-1]


Parm = PARM()
#%%
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

Origin.train = Data.TensorDataset(torch.stack(Train[0]).type(torch.FloatTensor),
                            torch.tensor(Train[1]).type(torch.LongTensor))
Origin.test = Data.TensorDataset(torch.stack(Test[0]).type(torch.FloatTensor), 
                            torch.tensor(Test[1]).type(torch.LongTensor))
Origin.train_loader = Data.DataLoader(dataset=Origin.train,
                                      batch_size=Parm.batch_size,
                                      shuffle=True)
Origin.test_loader = Data.DataLoader(dataset=Origin.test, 
                                     batch_size=100,
                                     shuffle=False)
Fusion_task.train, Fusion_task.test, Fusion_task.train_loader, Fusion_task.test_loader = Origin.train, Origin.test, Origin.train_loader, Origin.test_loader
#%% Create Models
Model = Parm.model
Origin.model =  Model() if Parm.cuda==False else Model().cuda()
Origin.optimizer = torch.optim.SGD(Origin.model.parameters(), lr=Parm.lr)
for i in range(Parm.task_number):
    Tasks[i].model = Model() if Parm.cuda==False else Model().cuda()
    Tasks[i].optimizer = torch.optim.SGD(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_model = Model() if Parm.cuda==False else Model().cuda()

loss_func = torch.nn.CrossEntropyLoss()

#%% Train
if Parm.draw:
    fig = plt.figure(1)
    plt.ion()
start = time.time()
for epoch in range(100):
    for task in Tasks:
        training_process(task, loss_func, Parm)
        testing_process(task, Parm)
    if Parm.draw:
        accuracy, name = [], []
        for task in Tasks:
            accuracy.append(task.test_accuracy[task.ID])
            name.append(f"Task{task.ID}")
        draw_result(accuracy, fig, name, True)
finish = time.time()
Parm.time['solo_train'] = finish - start

#%% Adding Plugin
print('Before')
for i in range(Parm.task_number):
    Tasks[i].model0 = Tasks[i].model
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model))
    Tasks[i].model.plugin_hook()
    print(f"Accuracy: {testing_free(Tasks[i], Tasks[i].test_loader, Parm)}")
    print(f"Total Accuracy: {testing_free(Tasks[i], Origin.test_loader, Parm)}")
fusion_model = Plugin(fusion_model)

#%% Average
print('Average Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
models = [Task.model for Task in Tasks]
start = time.time()
fusion_model = Fusion.average_fusion(models, fusion_model)
finish = time.time()
Parm.time['AverFusion'] = finish - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, False))
    Tasks[i].model.plugin_hook()
start = time.time()
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True)
end = time.time()
Parm.time['PinvFusion'] = finish - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion Weight')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
start = time.time()
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=True)
end = time.time()
Parm.time['PinvFusionw'] = finish - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion Zero')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
start = time.time()
Tasks = Fusion.zero_rank_batch(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True)
end = time.time()
Parm.time['PinvFusion_z'] = finish - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()


#%%
print('Pinv Fusion MAN')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
start = time.time()
Tasks = Fusion.MAN_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=False)
end = time.time()
Parm.time['PinvFusion_man'] = end - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion MAN Weight')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.plugin_hook()
start = time.time()
Tasks = Fusion.MAN_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=True)
end = time.time()
Parm.time['PinvFusionw_man'] = end - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion MAN lambda')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
# lambda_list = [0.52, 0.50]
start = time.time()
Tasks = Fusion.MAN_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=False, lambda_list=[0.5, 0.5, 0.5, 0.5, 0.5])
end = time.time()
Parm.time['PinvFusionl_man'] = end - start
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Origin.test_loader, Parm)}")
torch.cuda.empty_cache()


# %%
Parm.T = 5
if Parm.draw:
    fig = plt.figure(10)
    plt.ion()
print('Pinv Fusion')
for i in range(Parm.task_number):
    Tasks[i].train_loader = Data.DataLoader(dataset=Tasks[i].train,
                                        batch_size=int(Parm.batch_size/Parm.task_number),
                                        shuffle=True)
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, False))
    Tasks[i].model.plugin_hook()
fusion_model.plugin_hook(True)
#%%
print('Pinv Fusion supervise')
import time
if Parm.draw:
    fig = plt.figure(4)
    plt.ion()
start = time.time()
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=True)#, lambda_list=[0.48,0.4])
Fusion_task.model = fusion_model
pinv_fusion_supervise = [testing_free(Fusion_task.model, Fusion_task.test_loader, Parm)]
print(f"Total Accuracy: {pinv_fusion_supervise[-1]}")
Fusion_task.optimizer = torch.optim.SGD(Fusion_task.model.parameters(), lr=Parm.lr)
for j in range(300): 
    Fusion.fine_tune(Fusion_task, Tasks, Parm, choose_type='supervise', layer_wise=False)
    pinv_fusion_supervise.append(testing_free(Fusion_task.model, Fusion_task.test_loader, Parm))
    if Parm.draw:
        accuracy, name = [], []
        # accuracy.append(Fusion_task.test_accuracy[Fusion_task.ID])
        name.append(f"Task")
        draw_result([pinv_fusion_supervise], fig, name, True)
    # print(f"{j} Total Accuracy: {pinv_fusion_supervise[-1]}")
end = time.time()
print(end - start)
#%%
print('Pinv Fusion kd_layer')
import time
if Parm.draw:
    fig = plt.figure(4)
    plt.ion()
start = time.time()
# Tasks = Fusion.MAN_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, ifbatch=True, ifweight=True)
Fusion_task.model = fusion_model
pinv_fusion_kd = [testing_free(Fusion_task.model, Fusion_task.test_loader, Parm)]
print(f"Total Accuracy: {pinv_fusion_kd[-1]}")
Fusion_task.optimizer = torch.optim.SGD(Fusion_task.model.parameters(), lr=Parm.lr/10)
for j in range(1000):
    fine_tune(Fusion_task, Tasks, Parm, choose_type='kd_layer', Lambda=0.5)
    pinv_fusion_kd.append(testing_free(Fusion_task.model, Fusion_task.test_loader, Parm))
    if Parm.draw:
        accuracy, name = [], []
        # accuracy.append(Fusion_task.test_accuracy[Fusion_task.ID])
        name.append(f"Task")
        draw_result([pinv_fusion_kd], fig, name, True)
end = time.time()
print(end - start)