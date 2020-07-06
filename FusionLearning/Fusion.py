# -*- encoding: utf-8 -*-
'''
@Time        :2020/07/01 20:07:56
@Author      :Qu Yue
@File        :Fusion.py
@Software    :Visual Studio Code
Introduction: Fusion
'''
#%% Import Packages
import torch

#%% Average Fusion
def average_fusion(Tasks, model_fusion):
    def average(nets):
        aver_net = nets[0].clone()
        for i in range(1, len(nets)):
            aver_net += nets[i].data
        aver_net /= len(nets)
        return aver_net

    layers0 = list(model_fusion.model.named_parameters())
    layers = [list(task.model.named_parameters()) for task in Tasks]

    for i in range(len(layers0)):
        layers0[i][1].data = average([model[i][1].data for model in layers])
    return model_fusion

#%%
def pinv_fusion(Tasks, model_fusion, Parm):
    X_s = []
    W_s = []
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)kh

    W = model_fusion.W
    layers = list(W.keys())
    for layer in layers:
        Z_s = []
        H_s = []
        for i in range(len(X_s)):
            X = X_s[i]
            W = W_s[i]
            Z = X[layer].transpose(1,0).mm(X[layer])
            Z_s.append(Z)
            H_s.append(Z.mm(W[layer]))
        Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        W[layer].data = Z.pinverse().mm(H)
    model_fusion.W_update(W)
    return model_fusion
