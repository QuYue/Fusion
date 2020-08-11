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
        self.dataset_ID = 3
        self.test_size = 0.2
        self.epoch = 100
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
Model = CNN1
for i in range(Parm.task_number):
    Tasks[i].model = Model() if Parm.cuda==False else Model().cuda()
    Tasks[i].optimizer = torch.optim.SGD(Tasks[i].model.parameters(), lr=Parm.lr)
fusion_model = Model() if Parm.cuda==False else Model().cuda()
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
torch.cuda.empty_cache()

#%%
import time
a = time.time()
print('Pinv Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, False))
    Tasks[i].model.plugin_hook()
    #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
    #Tasks[i].model.oneshot_rank(Parm)
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, False)
#fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm, True, [0.512, 0.488])
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()
b = time.time()
print(f'Time: {b-a}')
#%%
print('Pinv Fusion Weight')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
    Tasks[i].model.plugin_hook()
    #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
    #Tasks[i].model.oneshot_rank(Parm)
fusion_model = Fusion.pinv_fusion_weight(Tasks, fusion_model, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()
#%%
print('Pinv Fusion lambda')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.plugin_hook()

# lambda_list = [0.33, 0.5, 0.5, 0.4, 0.57]
# lambda_list = [0.5, 0.48, 0.6]
# lambda_list = [0.54, 0.5]
# lambda_list =[0.57, 0.50, 0.5, 0.5, 0.51]
fusion_model = Fusion.pinv_fusion_lambda(Tasks, fusion_model, lambda_list, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()
#%%
# print('Pinv Fusion2')
# for i in range(Parm.task_number):
#     Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
#     Tasks[i].model.plugin_hook()
#     #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
#     #Tasks[i].model.oneshot_rank(Parm)
# fusion_model = Fusion.pinv_fusion2(Tasks, fusion_model, Parm)
# for i in range(Parm.task_number):
#     print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
# print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
# torch.cuda.empty_cache()

# #%%
# print('Pinv Fusion Weight')
# for i in range(Parm.task_number):
#     Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
#     Tasks[i].model.plugin_hook()
#     #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
#     #Tasks[i].model.oneshot_rank(Parm)
# fusion_model = Fusion.pinv_fusion_weight(Tasks, fusion_model, Parm)
# for i in range(Parm.task_number):
#     print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
# print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")

#%%
# print('Pinv Fusion Weight')
# for i in range(Parm.task_number):
#     Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
#     Tasks[i].model.plugin_hook()
#     #Tasks[i].model.Normalization(Tasks[i].train[:1000][0], Parm)
#     #Tasks[i].model.oneshot_rank(Parm)
# fusion_model = Fusion.pinv_fusion_weight(Tasks, fusion_model, Parm)
# for i in range(Parm.task_number):
#     print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
# print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")

# #%%
print('Pinv Fusion Zero')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.plugin_hook()
Tasks = Fusion.zero_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion(Tasks, fusion_model, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()
#%%
print('Pinv Fusion Zero Weight')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.plugin_hook()
Tasks = Fusion.zero_rank(Tasks, Parm)
fusion_model = Fusion.pinv_fusion_weight(Tasks, fusion_model, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()

#%%
print('Pinv Fusion Zero lambda')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.plugin_hook()
Tasks = Fusion.zero_rank(Tasks, Parm)
# lambda_list = [0.5, 0.6]
# lambda_list = [0.35, 0.5, 0.5, 0.4, 0.6]
# lambda_list = [0.51, 0.5, 0.51]
# lambda_list = [0.5, 0.58]
# lambda_list = [0.62,0.5,0.5,0.7,0.5]
fusion_model = Fusion.pinv_fusion_lambda(Tasks, fusion_model, lambda_list, Parm)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()


#%%
# print('MAS Fusion')
# for i in range(Parm.task_number):
#     Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0))
#     Tasks[i].model.plugin_hook()
# fusion_model = Fusion.pinv_fusion(Tasks, fusion_model, Parm)
# Parm.Lambda = 0.000001
# for i in range(Parm.task_number):
#     print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}") 
# print(f"Total:{testing_free(fusion_model, Test_loader, Parm) :.5f}") 
# # Parm.fusion_lr2 = [1e-14, 1e-11,1e-13]
# Parm.fusion_lr2 = [1e-3, 1e-3,1e-3,1e-3, 1e-3,1e-3]
# fusion_model, save1 = Fusion.MAS_fusion(Tasks, fusion_model, Parm, testing_free, True, Test_loader)


# %%
if Parm.draw:
    plt.ioff()
    plt.show()




#%%
import time
a = time.time()
print('Pinv Fusion')
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, False))
    Tasks[i].model.plugin_hook()
fusion_model = Fusion.pinv_fusion_batch(Tasks, fusion_model, Parm,True)
for i in range(Parm.task_number):
    print(f"Accuracy: {testing_free(fusion_model, Tasks[i].test_loader, Parm)}")
print(f"Total Accuracy: {testing_free(fusion_model, Test_loader, Parm)}")
torch.cuda.empty_cache()
b = time.time()
print(f'Time: {b-a}')
#%%
import seaborn as sns
num_nodes = 30
d = []
for i in range(Parm.task_number):
    Tasks[i].model = copy.deepcopy(Plugin(Tasks[i].model0, True))
    Tasks[i].model.oneshot_rank(Parm)
    d.append(Tasks[i].model.get_sparse_connection1(num_nodes, threhold = 0.48))

fusion_model.model = Plugin(fusion_model.model, True)
d.append(fusion_model.model.get_sparse_connection1(num_nodes, threhold=0.68))
#%%
#plt.subplot(3,1,1)
sns.heatmap(d[0],cmap='YlGnBu')
plt.show()
#%%
sns.heatmap(d[1],cmap='YlGnBu')
plt.show()
#%%
sns.heatmap(d[2],cmap='YlGnBu')
plt.show()
# plt.subplot(3,1,2)
# sns.heatmap(d[1],cmap='YlGnBu')
# plt.subplot(3,1,3)
# sns.heatmap(d[2],cmap='YlGnBu')


# %%
import networkx as nx
def get_avg_cluater_path(g: nx.Graph) -> dict:
    try:
        cc = nx.average_clustering(g)
    except nx.NetworkXError as e:
        cc = np.nan
    try:
        apl = nx.average_shortest_path_length(g)
    except nx.NetworkXError as e:
        apl = np.nan
    return dict(
        cluster_coefficient=cc,
        avg_path_length=apl,
    )

g0 = nx.from_numpy_matrix(d[0])
g1 = nx.from_numpy_matrix(d[1])
g2 = nx.from_numpy_matrix(d[2])
d0 = get_avg_cluater_path(g0)
d1 = get_avg_cluater_path(g1)
d2 = get_avg_cluater_path(g2)
print(d0,d1,d2)



# %%



print(fusion_model.W['Linear1'].max())
print(fusion_model.W['Linear2'].max())
print(fusion_model.W['Linear3'].max())

print(Tasks[0].model.W['Linear1'].max())
print(Tasks[0].model.W['Linear2'].max())
print(Tasks[0].model.W['Linear3'].max())

print(Tasks[1].model.W['Linear1'].max())
print(Tasks[1].model.W['Linear2'].max())
print(Tasks[1].model.W['Linear3'].max())

# %%
