# -*- encoding: utf-8 -*-
'''
@Time        :2020/08/30 20:26:58
@Author      :Qu Yue
@File        :experiment1_MNIST_FNN.py
@Software    :Visual Studio Code
Introduction: MNIST_FNN
'''
#%% Import Packages
# %matplotlib qt5
import torch
import torchvision
import torch.utils.data as Data
import time 
import numpy as np
import pandas as pd
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
import record


#%% Hyper Parameters
class PARM:
    def __init__(self):
        self.data = DATASET() 
        self.dataset_ID = 2 # 2
        self.test_size = 0.2
        self.epoch = 50 # 50
        self.epoch2 = 500 # 500
        self.batch_size = 5000
        self.ifbatch = True
        self.lr = 0.001  # 0.0005
        self.lr2 = 0.001 # 0.005
        self.T = 5
        self.draw = True
        self.cuda = True
        self.showepoch = 1
        self.random_seed = 1 #1,2,3,6,7
        self.fusion_lr = 1e-12 # 0.000000000001
        self.Lambda = 0.5
        self.model =  CNN1 # CNN1
        self.time = dict()
        self.result = {'SoloNet':{}, 'FusionNet':{}, 'Origin':{}}
        self.optimizer = torch.optim.Adam
        self.optimizer2 = torch.optim.Adam
    @property
    def dataset_name(self):
        return self.data.data_dict[self.dataset_ID]
    @property
    def task_number(self):
        return self.data.tasks[self.dataset_name] 

    def lambda_list(self, sort=None):
        if self.model == FNN1:
            if sort == "AF":
                lambda_list = [[0.40, 0.55],[0.35, 0.5, 0.5, 0.4, 0.6],[0.51, 0.5, 0.51]]
            elif sort == "MAN":
                lambda_list = [[0.41, 0.6], [0.3,0.51,0.5,0.4,0.6], [0.5,0.5,0.5]]
            else:
                lambda_list = [[0.40, 0.56],[0.33, 0.5, 0.5, 0.4, 0.57],[0.5, 0.48, 0.6]]
        elif self.model == CNN1:
            if sort == "AF":
                lambda_list = [[0.5, 0.58],[0.62,0.5,0.5,0.7,0.5],[0.5,0.5,0.5]]
            elif sort == "MAN":
                lambda_list = [[0.5, 0.5],[0.5, 0.5, 0.5, 0.5, 0.5],[0.5,0.5,0.5]]
            else:
                lambda_list = [[0.54, 0.5],[0.57, 0.50, 0.5, 0.5, 0.51],[0.5,0.5,0.5]]
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
if Parm.draw:
    fig_id = 1

#%% Task
class TASK:
    def __init__(self, ID=0):
        self.ID = ID
        self.train = []
        self.test = []
        self.train_accuracy = {self.ID:[]}
        self.test_accuracy = {self.ID:[]}
        self.time = []
    def clear(self):
        self.train_accuracy = {self.ID:[]}
        self.test_accuracy = {self.ID:[]}
        self.time = []

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
                                       batch_size=1000,
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
                                     batch_size=1000,
                                     shuffle=False)
Fusion_task.train, Fusion_task.test, Fusion_task.train_loader, Fusion_task.test_loader =\
Origin.train, Origin.test, Origin.train_loader, Origin.test_loader

#%% Create Models
Model = Parm.model
Origin.model =  Model() if Parm.cuda==False else Model().cuda()
Origin.optimizer = Parm.optimizer(Origin.model.parameters(), lr=Parm.lr)
for i in range(Parm.task_number):
    Tasks[i].model = Model() if Parm.cuda==False else Model().cuda()
    Tasks[i].optimizer = Parm.optimizer(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_net = Model() if Parm.cuda==False else Model().cuda()
# loss function
loss_func = torch.nn.CrossEntropyLoss()

#%% Train
if Parm.draw:
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.ion()
start = time.time()
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
finish = time.time()
Parm.time['SoloNet'] = finish - start

#%% Adding Plugin
print('Before')
Parm.result['SoloNet'] = []
for i in range(Parm.task_number):
    Tasks[i].model0 = Tasks[i].model
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model))
    Tasks[i].model.plugin_hook()
    acc_i = testing_free(Tasks[i], Tasks[i].test_loader, Parm)
    acc_t = testing_free(Tasks[i], Origin.test_loader, Parm)
    print(f"Accuracy: {acc_i}")
    print(f"Total Accuracy: {acc_t}")
    Parm.result['SoloNet'].append({'Acc': acc_i, 'TotalAcc': acc_t})
fusion_net = Plugin(fusion_net)
#%% Prepare
def Tasks_initial(Tasks, Parm, hook=True):
    for i in range(Parm.task_number):
        Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
        Tasks[i].model.clear_hook()
        if hook == True:
            Tasks[i].model.plugin_hook()
    return Tasks

def get_result(fusion_net, Tasks, Parm, result):
    for i in range(Parm.task_number):
        acc_i = testing_free(fusion_net, Tasks[i].test_loader, Parm)
        result['Acc'].append(acc_i)
        print(f"Accuracy: {acc_i}")
    acc_t = testing_free(fusion_net, Origin.test_loader, Parm)
    result['TotalAcc'] = acc_t
    print(f"Total Accuracy: {acc_t}")
    torch.cuda.empty_cache()
    return result

#%% Average
print('Average Fusion')
result = {'Acc':[], 'TotalAcc':0}
name = 'AverFusion'
Tasks = Tasks_initial(Tasks, Parm)
solo_net = [Task.model for Task in Tasks]
start = time.time()
fusion_net = Fusion.average_fusion(solo_net, fusion_net)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion
print('Pinv Fusion')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion Weight
print('Pinv Fusion Weight')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion_W'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion Lambda
# print('Pinv Fusion Lambda')
# result = {'Acc':[], 'TotalAcc':0}
# name = 'PinvFusion_L'
# Tasks = Tasks_initial(Tasks, Parm)
# start = time.time()
# fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True, lambda_list=[0.40, 0.56])
# finish = time.time()
# Parm.time[name] = finish - start
# print(f"{name}: {Parm.time[name]}s")
# result = get_result(fusion_net, Tasks, Parm, result)
# Parm.result['FusionNet'][name] = result

#%% Pinv Fusion + AF
print('Pinv Fusion + AF')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion+AF'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
Tasks = Fusion.AF_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion Weight + AF
print('Pinv Fusion Weight + AF')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion_W+AF'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
Tasks = Fusion.AF_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion Lambda + AF
# print('Pinv Fusion Lambda + AF')
# result = {'Acc':[], 'TotalAcc':0}
# name = 'PinvFusion_L+AF'
# Tasks = Tasks_initial(Tasks, Parm)
# start = time.time()
# Tasks = Fusion.AF_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
# fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True, lambda_list=[0.41, 0.50])
# finish = time.time()
# Parm.time[name] = finish - start
# print(f"{name}: {Parm.time[name]}s")
# result = get_result(fusion_net, Tasks, Parm, result)
# Parm.result['FusionNet'][name] = result

#%% Pinv Fusion + MAN
print('Pinv Fusion + MAN')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion+MAN'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
Tasks = Fusion.MAN_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

#%% Pinv Fusion Weight + MAN
print('Pinv Fusion Weight + MAN')
result = {'Acc':[], 'TotalAcc':0}
name = 'PinvFusion_W+MAN'
Tasks = Tasks_initial(Tasks, Parm)
start = time.time()
Tasks = Fusion.MAN_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True)
finish = time.time()
Parm.time[name] = finish - start
print(f"{name}: {Parm.time[name]}s")
result = get_result(fusion_net, Tasks, Parm, result)
Parm.result['FusionNet'][name] = result

# #%% Pinv Fusion Lambda + MAN
# print('Pinv Fusion Lambda + MAN')
# result = {'Acc':[], 'TotalAcc':0}
# name = 'PinvFusion_L+MAN'
# Tasks = Tasks_initial(Tasks, Parm)
# start = time.time()
# Tasks = Fusion.MAN_rank(Tasks, Parm, ifbatch=Parm.ifbatch)
# fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True, lambda_list=[0.5, 0.5])
# finish = time.time()
# Parm.time[name] = finish - start
# print(f"{name}: {Parm.time[name]}s")
# result = get_result(fusion_net, Tasks, Parm, result)
# Parm.result['FusionNet'][name] = result

#%% Origin Training
print('Origin Training')
name_t = 'Origin'
Origin.model =  Model() if Parm.cuda==False else Model().cuda()
Origin.optimizer = Parm.optimizer2(Origin.model.parameters(), lr=0.0001)
Origin.clear()
if Parm.draw:
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.ion()
start = time.time()
for epoch in range(Parm.epoch2):
    training_process(Origin, loss_func, Parm)
    testing_process(Origin, Parm)
    Origin.time.append(time.time()-start)
    if Parm.draw:
        accuracy, name = [], []
        accuracy.append(Origin.test_accuracy[Origin.ID])
        name.append(f"Task{Origin.ID}")
        draw_result(accuracy, fig, name, True)
finish = time.time()
Parm.time[name_t] = Origin.time
Parm.result[name_t] = Origin.test_accuracy
print(f"{name_t}: {Parm.time[name_t][-1]}s")

#%% Fine tuning
Parm.T = 5
for i in range(Parm.task_number):
    Tasks[i].train_loader = Data.DataLoader(dataset=Tasks[i].train,
                                        batch_size=int(Parm.batch_size/Parm.task_number),
                                        shuffle=True)
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, False))
    Tasks[i].model.plugin_hook()

#%% Fusion Fine-tuning
print('Fusion Fine-tuning')
name_t = 'FusionFineTune'
Tasks = Tasks_initial(Tasks, Parm)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True)
fusion_net.clear_hook()
Fusion_task.model = fusion_net
Fusion_task.optimizer = Parm.optimizer2(Fusion_task.model.parameters(), lr=Parm.lr2)
Fusion_task.clear()
if Parm.draw:
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.ion()
start = time.time()
for epoch in range(Parm.epoch2):
    training_process(Fusion_task, loss_func, Parm)
    testing_process(Fusion_task, Parm)
    Fusion_task.time.append(time.time()-start)
    if Parm.draw:
        accuracy, name = [], []
        accuracy.append(Fusion_task.test_accuracy[Fusion_task.ID])
        name.append(f"Task{Fusion_task.ID}")
        draw_result(accuracy, fig, name, True)
finish = time.time()
Parm.time[name_t] = Fusion_task.time
Parm.result['FusionNet'][name_t] = Fusion_task.test_accuracy[Fusion_task.ID]
print(f"{name_t}: {Parm.time[name_t][-1]}s")

#%% Fusion MLKD
print('Fusion MLKD Layer')
name_t = 'FusionMLKD'
Tasks = Tasks_initial(Tasks, Parm)
fusion_net = Fusion.pinv_fusion(Tasks, fusion_net, Parm, ifbatch=Parm.ifbatch, ifweight=True)
fusion_net.clear_hook()
Fusion_task.model = fusion_net
Fusion_task.optimizer = Parm.optimizer2(Fusion_task.model.parameters(), lr=Parm.lr2)
Fusion_task.clear()
for Task in Tasks:
    Task.model.clear_hook()
    Task.model.plugin_hook(False, True, False)
start = time.time()
if Parm.draw:
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.ion()
result = []
time_r = []
for j in range(Parm.epoch2):
    Fusion.fine_tune(Fusion_task, Tasks, Parm, choose_type='kd_layer', Lambda=Parm.Lambda)
    Fusion_task.time.append(time.time()-start)
    Fusion_task.model.clear_hook()
    result.append(testing_free(Fusion_task.model, Fusion_task.test_loader, Parm))
    time_r.append(time.time()-start)
    if Parm.draw:
        accuracy, name = [], []
        name.append(f"Task")
        draw_result([result], fig, name, True)
finish = time.time()
print(finish - start)
Parm.time[name_t] = time_r
Parm.result['FusionNet'][name_t] = result
print(f"{name_t}: {Parm.time[name_t][-1]}s")




# %% Save 
if Parm.draw:
    plt.ioff()
    plt.show()

record.record('./result/e4_2', Parm, 'pkl')
# %%

plt.figure(5)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionFineTune'])
plt.legend(['Normal', 'Fusion+FineTune'])
# plt.plot(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionMLKD'])
# plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'])

plt.plot()
plt.xlabel('Time(s)')
plt.ylabel('Accuracy')
plt.xlim(-10,300)
plt.ylim(0.1, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
plt.show()

# %%
