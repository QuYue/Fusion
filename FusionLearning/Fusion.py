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

#%% Pseudo Inverse Fusion 
def pinv_fusion(Tasks, model_fusion, Parm):
    X_s = []
    W_s = []
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)

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

def pinv_fusion2(Tasks, model_fusion, Parm):
    def new_X():
        if model_fusion.ifhook == False:
            model_fusion.plugin_hook()
        X_s = []
        for Task in Tasks:
            X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
            model_fusion.forward(X)
            X_s.append(model_fusion.X)
        return X_s

    X_s = []
    W_s = []
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)
    
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
        X_s = new_X()
    
    return model_fusion
        

# #%% Linear Fusion
# def linear_fusion(Tasks, model_fusion, Parm):
#     for Task in Tasks:
#         if Task.model.ifhook == False:
#             Task.model.plugin_hook()
#             model1.forward(data1)
#             model2.forward(data2)

#     W = model_fusion.W
#     W1 = model1.W
#     W2 = model2.W
#     X1 = model1.X
#     X2 = model2.X

#     layers = list(W1.keys())
#     loss = []
#     for layer in layers:
#         Z1 = X1[layer].transpose(1,0).mm(X1[layer])
#         Z2 = X2[layer].transpose(1,0).mm(X2[layer])
#         Z = Z1 + Z2
#         H = Z1.mm(W1[layer]) + Z2.mm(W2[layer])
#         loss.append(torch.norm((Z.mm(W[layer])-H), p='fro').data.cpu())
#         grad = Z.transpose(1,0).mm(Z.mm(W[layer])-H)
#         W[layer] -= step * grad
#     model_fusion.W_update(W)
#     print(f"loss:{[i.data for i in loss]}", end="")
#     return model_fusion

# #%% Linear Fusion
# def linear_fusion2(model1, model2, model_fusion, data1, data2, step=0.01):
#     if model1.ifhook == False:
#         model1.plugin_hook()
#     if model2.ifhook == False:
#         model2.plugin_hook()
#     model1.forward(data1)
#     model2.forward(data2)
#     W = model_fusion.W
#     W1 = model1.W
#     W2 = model2.W
#     X1 = model1.X
#     X2 = model2.X

#     layers = list(W1.keys())
#     loss = []
#     for layer in layers:
#         Z1 = X1[layer].transpose(1,0).mm(X1[layer])
#         Z2 = X2[layer].transpose(1,0).mm(X2[layer])
#         Z = Z1 + Z2
#         H = Z1.mm(W1[layer]) + Z2.mm(W2[layer])
#         loss.append(torch.norm((Z.mm(W[layer])-H), p='fro').data.cpu())
#         grad = Z.transpose(1,0).mm(Z.mm(W[layer])-H)
#         W[layer] -= step * grad
#     model_fusion.W_update(W)
#     print(f"loss:{[i.data for i in loss]}", end="")
#     return model_fusion


# %%
