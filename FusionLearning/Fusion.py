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
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)

    layers = list(fusion_W.keys())
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
        fusion_W[layer].data = Z.pinverse().mm(H)
    model_fusion.W_update(fusion_W)
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
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)
       
    layers = list(fusion_W.keys())
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
        fusion_W[layer].data = Z.pinverse().mm(H)
        model_fusion.W_update(fusion_W)
        X_s = new_X()  
    return model_fusion

#%% Pseudo Inverse Fusion + weight
def pinv_fusion_weight(Tasks, model_fusion, Parm):
    X_s = []
    W_s = []
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)

    layers = list(fusion_W.keys())
    for layer in layers:
        Z_s = []
        H_s = []
        for i in range(len(X_s)):
            X = X_s[i]
            W = W_s[i]
            Z = X[layer].transpose(1,0).mm(X[layer])/X[layer].shape[0]
            Z_s.append(Z)
            H_s.append(Z.mm(W[layer]))
        Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        fusion_W[layer].data = Z.pinverse().mm(H)
    model_fusion.W_update(fusion_W)
    return model_fusion

def pinv_fusion2_weight(Tasks, model_fusion, Parm):
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
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)
       
    layers = list(fusion_W.keys())
    for layer in layers:
        Z_s = []
        H_s = []
        for i in range(len(X_s)):
            X = X_s[i]
            W = W_s[i]
            Z = X[layer].transpose(1,0).mm(X[layer])/X[layer].shape[0]
            Z_s.append(Z)
            H_s.append(Z.mm(W[layer]))
        Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        fusion_W[layer].data = Z.pinverse().mm(H)
        model_fusion.W_update(fusion_W)
        X_s = new_X()  
    return model_fusion
        
#%% Linear Fusion
def linear_fusion(Tasks, model_fusion, Parm, ifprint=True):
    X_s = []
    W_s = [] 
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)

    layers = list(fusion_W.keys())
    loss = []
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
        loss.append((torch.norm((Z.mm(fusion_W[layer])-H), p='fro')**2/2).data.cpu())
        grad = Z.transpose(1,0).mm(Z.mm(fusion_W[layer])-H)
        #print(f"{layer}:{grad[0,0]}", end='| ')
        fusion_W[layer] -= Parm.fusion_lr * grad
    model_fusion.W_update(fusion_W)
    if ifprint:
        print(f"loss:{[i.data for i in loss]}", end="")
    return model_fusion

#%% Linear Fusion+weight
def linear_fusion_weight(Tasks, model_fusion, Parm, ifprint=True):
    X_s = []
    W_s = [] 
    fusion_W = model_fusion.W
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)

    layers = list(fusion_W.keys())
    loss = []
    for layer in layers:
        Z_s = []
        H_s = []
        for i in range(len(X_s)):
            X = X_s[i]
            W = W_s[i]
            Z = X[layer].transpose(1,0).mm(X[layer])/X[layer].shape[0]
            Z_s.append(Z)
            H_s.append(Z.mm(W[layer]))
        Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        loss.append((torch.norm((Z.mm(fusion_W[layer])-H), p='fro')**2/2).data.cpu())
        grad = Z.transpose(1,0).mm(Z.mm(fusion_W[layer])-H)
        fusion_W[layer] -= Parm.fusion_lr * grad
    model_fusion.W_update(fusion_W)
    if ifprint:
        print(f"loss:{[i.data for i in loss]}", end="")
    return model_fusion

#%% Linear Fusion
def linear_fusion_adam(Tasks, model_fusion, Parm, testing, ifprint=True):
    X_s = []
    W_s = [] 
    fusion_W = model_fusion.W
    layers = list(fusion_W.keys())
    optimizer = dict()
    for i, layer in enumerate(layers):
        fusion_W[layer].requires_grad = True
        optimizer[layer] = torch.optim.Adam([fusion_W[layer]], lr=Parm.fusion_lr2[i])
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)
    
    for epoch in range(500):
        print(f"Epoch: {epoch}", end=' |')
        loss = []
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
            loss.append(torch.norm((Z.mm(fusion_W[layer])-H), p='fro')**2/2)
            optimizer[layer].zero_grad()
            loss[-1].backward()
            #print(f"{layer}:{fusion_W[layer].grad[0,0]}",end='| ')
            optimizer[layer].step()
        #Loss = torch.sum(torch.stack(loss))
        model_fusion.W_update(fusion_W)
        if ifprint:
            print(f"loss:{[i.data.cpu() for i in loss]}", end="")

        for i in range(Parm.task_number):
            print(f"Accuray: {testing(model_fusion, Tasks[i].test_loader, Parm)}", end=" |")
        print("")
    return model_fusion

def linear_fusion_adam_weight(Tasks, model_fusion, Parm, testing, ifprint=True):
    X_s = []
    W_s = [] 
    fusion_W = model_fusion.W
    layers = list(fusion_W.keys())
    optimizer = dict()
    for i, layer in enumerate(layers):
        fusion_W[layer].requires_grad = True
        optimizer[layer] = torch.optim.Adam([fusion_W[layer]], lr=Parm.fusion_lr2[i])
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        W_s.append(Task.model.W)
    
    for epoch in range(500):
        print(f"Epoch: {epoch}", end=' |')
        loss = []
        for layer in layers:
            Z_s = []
            H_s = []
            for i in range(len(X_s)):
                X = X_s[i] 
                W = W_s[i]
                Z = X[layer].transpose(1,0).mm(X[layer])/X[layer].shape[0]
                Z_s.append(Z)
                H_s.append(Z.mm(W[layer]))
            Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
            H = torch.sum(torch.stack(H_s, dim=0), dim=0)
            loss.append(torch.norm((Z.mm(fusion_W[layer])-H), p='fro')**2/2)
            optimizer[layer].zero_grad()
            loss[-1].backward()
            #print(f"{layer}:{fusion_W[layer].grad[0,0]}",end='| ')
            optimizer[layer].step()
        #Loss = torch.sum(torch.stack(loss))
        model_fusion.W_update(fusion_W)
        if ifprint:
            print(f"loss:{[i.data.cpu() for i in loss]}", end="")

        for i in range(Parm.task_number):
            print(f"Accuray: {testing(model_fusion, Tasks[i].test_loader, Parm)}", end=" |")
        print("")
    return model_fusion

#%% 



# %%
