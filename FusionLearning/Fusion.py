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
import numpy as np
import copy
import torch.utils.data as Data
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
    for Task in Tasks:
        Task.model.empty_x_y()
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
    for Task in Tasks:
        Task.model.empty_x_y()
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

#%%
def pinv_fusion_lambda(Tasks, model_fusion, lambda_list, Parm):
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
            Z = X[layer].transpose(1,0).mm(X[layer])*lambda_list[i]/X[layer].shape[0]
            Z_s.append(Z)
            H_s.append(Z.mm(W[layer]))
        Z = torch.sum(torch.stack(Z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        fusion_W[layer].data = Z.pinverse().mm(H)
    model_fusion.W_update(fusion_W)
    for Task in Tasks:
        Task.model.empty_x_y()
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
            print(f"Accuracy: {testing(model_fusion, Tasks[i].test_loader, Parm)}", end=" |")
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
            print(f"Accuracy: {testing(model_fusion, Tasks[i].test_loader, Parm)}", end=" |")
        print("")
    return model_fusion

def pinv_fusion_batch(Tasks, model_fusion, Parm, ifbatch=True, ifweight=False, lambda_list=None, rcond=1e-4):
    if lambda_list == None:
        lambda_list = torch.ones(len(Tasks))
    else:
        lambda_list = torch.tensor(lambda_list)
    Z_s = []
    W_s = []
    count = []
    fusion_W = model_fusion.W
    for Task in Tasks:
        Z = dict()
        counter = 0
        if Task.model.ifhook == False:
            Task.model.plugin_hook() 
        train_loader = Data.DataLoader(dataset=Task.train,
                            batch_size=1000,
                            shuffle=True)
        data_loader =  train_loader if ifbatch else [[Task.train[:][0],1]]

        for X, _ in data_loader:
            if Parm.cuda == True: X = X.cuda() 
            Task.model.forward(X)
            counter += X.shape[0]
            X = Task.model.X
            for layer in Task.model.plug_net_name:
                if layer in Z:
                    Z[layer] += X[layer].transpose(1,0).mm(X[layer])
                else:
                    Z[layer] = X[layer].transpose(1,0).mm(X[layer])
        Z_s.append(Z)        
        W_s.append(Task.model.W)
        count.append(counter)

    layers = list(fusion_W.keys())
    for layer in layers:
        H_s = []
        z_s = []
        for i in range(len(Z_s)):
            z = lambda_list[i]*Z_s[i][layer]/count[i] if ifweight else lambda_list[i]*Z_s[i][layer]
            z_s.append(z)
            H_s.append(z.mm(W_s[i][layer]))
        Z = torch.sum(torch.stack(z_s, dim=0), dim=0)
        H = torch.sum(torch.stack(H_s, dim=0), dim=0)
        # print(f"Z.max:{Z.max()}|Z.min:{Z.min()}|Z.std:{Z.std()}")
        # print(f"Z-1.max:{Z.pinverse(rcond=1e-4).max()}|Z-1.min:{Z.pinverse(rcond=1e-4).min()}|Z-1.std:{Z.pinverse(rcond=1e-2).std()}")
        fusion_W[layer].data = Z.pinverse(rcond=rcond).mm(H)
    model_fusion.W_update(fusion_W)
    for Task in Tasks:
        Task.model.empty_x_y()
    return model_fusion


#%% 
def rank(W1, W2, index):
    W1 =  W1[:, index]
    if W1.shape[-1]+1 == W2.shape[0]:
        W2[:-1, :] = W2[:-1, :][index, :]
    else:
        w = W2[:-1, :].reshape(W1.shape[-1], -1)
        W2[:-1, :] = w[index, ...].reshape(-1, W2.shape[-1])
    return W1, W2

# Level Sort
def level_sort(sort_list):
    number, length = sort_list.shape # number: tasks' number; length: cell' number
    part_length = length // number # length of each part
    extra_length = length % number # length of extra part
    index0 = []
    index1 = []
    # index 0 (初始排序: 从乱序变成从小到大的排序)
    for i in range(number):
        index0.append(sort_list[i].argsort())
    # index 1 (交错排序)  
    for i in range(number):
        part = []
        position = 0
        t = list(range(i, i+extra_length))
        if i + extra_length > number:
            t.extend(list(range(0, i+extra_length-number)))
        for j in range(number):
            if j in t:        
                part.append(np.arange(position, position+part_length+1))
                position += part_length+1
            else:
                part.append(np.arange(position, position+part_length))
                position += part_length
        p = copy.deepcopy(part)
        p[0:number-i] = part[i:]
        if i != 0: p[-i:] = part[:i]
        index1.append(p)
    # 逆序
    for i in range(number):
        if i%2 == 1:
            for j in range(len(index1[i])):
                index = index1[i][j]
                index = index[np.arange(len(index)-1,-1,-1)] # Reverse
                index1[i][j] = index
        index1[i] = np.hstack(index1[i])
    # 结合(相对原始顺序进行排序)
    index2 = []
    for i in range(number):
        index2.append(index0[i][index1[i]])      
    return index2, index1

# zero_rank
def zero_rank(Tasks, Parm):
    X_s = []
    Y_s = []
    W_s = []
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
            Task.model.eval()
        X = Task.train[:][0] if Parm.cuda == False else Task.train[:][0].cuda()
        Task.model.forward(X)
        X_s.append(Task.model.X)
        Y_s.append(Task.model.Y)
        W_s.append(Task.model.W)
    layers = list(W_s[0].keys())
    for i, layer in enumerate(layers[:-1]):
        zero_frequency = []
        for Y in Y_s:
            if 'Conv' in layer:
                changed_Y = Y[layer].permute(0,2,3,1).reshape(-1, Y[layer].shape[1])
            else:
                changed_Y = Y[layer]
            # zero_frequency.append(torch.sum(Y[layer]>0, axis=0).float()/Y[layer].shape[0])
            zero_frequency.append(torch.sum(changed_Y>0, 0).float()/changed_Y.shape[0])
        zero_frequency = torch.stack(zero_frequency).cpu().numpy()
        sort_list = np.argsort(np.argsort(zero_frequency, axis=1),axis=1)
        l_sort, _ = level_sort(sort_list)

        for j in range(len(W_s)):
            W_s[j][layers[i]], W_s[j][layers[i+1]] = rank(W_s[j][layers[i]], W_s[j][layers[i+1]], l_sort[j])
    for i in range(len(W_s)):
        Tasks[i].model.W_update(W_s[i])
        Tasks[i].model.empty_x_y()
    return Tasks
#%%
def zero_rank_batch(Tasks, Parm, ifbatch=True):
    Y_s = []
    W_s = []
    zero_frequency = []
    layers = Tasks[0].model.plug_net_name
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.plugin_hook()
            Task.model.eval()
        W_s.append(Task.model.W)
        train_loader = Data.DataLoader(dataset=Task.train,batch_size=1000,shuffle=False)
        data_loader =  train_loader if ifbatch else [[Task.train[:][0],1]]
        zero_frequence = dict()
        count = dict()
        for X, _ in data_loader:
            if Parm.cuda == True: X = X.cuda() 
            Task.model.forward(X)
            Y = Task.model.Y
            for i, layer in enumerate(layers[:-1]):
                if 'Conv' in layer:
                    changed_Y = Y[layer].permute(0,2,3,1).reshape(-1, Y[layer].shape[1])
                else:
                    changed_Y = Y[layer]
                if layer in zero_frequence:
                    zero_frequence[layer] += torch.sum(changed_Y>0, 0).float()
                    count[layer] += changed_Y.shape[0]
                else:
                    zero_frequence[layer]=torch.sum(changed_Y>0, 0).float()
                    count[layer] = changed_Y.shape[0]
        z = dict()
        for layer in layers[:-1]:
            z[layer] = zero_frequence[layer]/count[layer]
        zero_frequency.append(z)        

    for i, layer in enumerate(layers[:-1]):
        zz = []
        for z in zero_frequency:
            zz.append(z[layer])
        zz = torch.stack(zz).cpu().numpy()
        sort_list = np.argsort(np.argsort(zz, axis=1),axis=1)
        l_sort, _ = level_sort(sort_list)
        for j in range(len(W_s)):
            W_s[j][layers[i]], W_s[j][layers[i+1]] = rank(W_s[j][layers[i]], W_s[j][layers[i+1]], l_sort[j])

    for i in range(len(W_s)):
        Tasks[i].model.W_update(W_s[i])
        Tasks[i].model.empty_x_y()
    return Tasks


#%%
def MAN_rank(Tasks, Parm, ifbatch=True):
    Y_s = []
    W_s = []
    importance = []
    layers = list(Tasks[0].model.plug_net_name)
    for Task in Tasks:
        if Task.model.ifhook == False:
            Task.model.plugin_hook()
            Task.model.eval()
        W_s.append(Task.model.W)
        W = dict()
        for layer in layers:
            W[layer] = W_s[-1][layer][:-1,:]
        train_loader = Data.DataLoader(dataset=Task.train, batch_size=1000, shuffle=False)
        data_loader = train_loader if ifbatch else [[Task.train[:][0],1]]
        important = dict()
        count = dict()
        for X, _ in data_loader:
            if Parm.cuda == True: X = X.cuda() 
            Task.model.forward(X)
            Y = Task.model.Y
            for i, layer in enumerate(layers[:-1]):
                if 'Conv' in layers[i+1]:
                    changed_Y = Y[layers[i+1]].permute(0,2,3,1).reshape(-1, Y[layers[i+1]].shape[1])
                else:
                    changed_Y = Y[layers[i+1]].clone()
                changed_Y[changed_Y<0] = 0 # ReLU
                changed_Y = changed_Y.transpose(1,0)
                if layers[i] in important:
                    important[layers[i]] += 2 * torch.sum(torch.abs(W[layers[i+1]].mm(changed_Y)), 1)
                    count[layers[i]] +=  changed_Y.shape[1]
                else:
                    important[layers[i]] = 2 * torch.sum(torch.abs(W[layers[i+1]].mm(changed_Y)), 1)
                    count[layers[i]] = changed_Y.shape[1] 
        for i, layer in enumerate(layers[:-1]):
            important[layer] = important[layer]/count[layer]
            if important[layer].shape[0] != W[layer].shape[1]:
                important[layer] = torch.mean(important[layer].reshape(W[layer].shape[1], -1), 1)
        importance.append(important)

    for i, layer in enumerate(layers[:-1]):
        MAN = []
        for z in importance:   
            MAN.append(z[layer])
        MAN = torch.stack(MAN).cpu().numpy()
        sort_list = np.argsort(np.argsort(MAN, axis=1),axis=1)
        l_sort, _ = level_sort(sort_list)
        for j in range(len(W_s)):
            W_s[j][layers[i]], W_s[j][layers[i+1]] = rank(W_s[j][layers[i]], W_s[j][layers[i+1]], l_sort[j])

    for i in range(len(W_s)):
        Tasks[i].model.W_update(W_s[i])
        Tasks[i].model.empty_x_y()
    return Tasks


#%%
def MAS_omega(X, Y, norm=False):
    layers = list(X.keys())
    omega = dict()
    for layer in layers:
        N = X[layer].shape[0]
        a = X[layer].transpose(1,0)
        b = torch.relu(Y[layer])
        omega[layer] = 2 / N * torch.abs(a.mm(b))
    return omega

def MAS_loss(fusion_W, W_s, Omega):
    layers = list(W_s[0].keys())
    Loss = dict()
    for layer in layers:
        loss = []
        for i in range(len(W_s)):
            loss.append(Omega[i][layer]*(torch.norm(W_s[i][layer]-fusion_W[layer], p='fro')**2))
        Loss[layer] = torch.sum(torch.stack(loss))
    return Loss

#%%
def Loss1(X_s, W_s, fusion_W, layer):
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
    loss = torch.norm((Z.mm(fusion_W[layer])-H), p='fro')**2/2 
    return loss

def Loss2(X_s, W_s, fusion_W, layer, nolinear):
    loss = 0
    N = 0
    for i in range(len(X_s)):
        X = X_s[i][layer]
        W = W_s[i][layer]
        out = nolinear(X.mm(W))
        fusion_out = nolinear(X.mm(fusion_W[layer]))
        loss += torch.norm(fusion_out-out, p='fro')**2
        N += X.shape[0]
    loss /= N
    return loss

#%%
def MAS_fusion(Tasks, model_fusion, Parm, testing, ifprint=True, Test_loader = None):
    X_s = []
    W_s = [] 
    Omega = []
    fusion_W = model_fusion.W
    layers = list(fusion_W.keys())
    optimizer = dict()
    save = []
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
        Omega.append(MAS_omega(Task.model.X, Task.model.Y, False))
    for epoch in range(500):
        print(f"Epoch: {epoch}", end=' |')
        mas_loss = MAS_loss(fusion_W, W_s, Omega)
        Loss = []
        for i, layer in enumerate(layers):
            # loss = Loss1(X_s, W_s, fusion_W, layer)
            loss = Loss2(X_s, W_s, fusion_W, layer, model_fusion.plug_nolinear[i])
            # loss += Parm.Lambda * mas_loss[layer]    
            optimizer[layer].zero_grad()
            loss.backward() 
            Loss.append(loss)
            optimizer[layer].step()
        model_fusion.W_update(fusion_W)
        if ifprint:
            print(f"loss:{[i.data.cpu() for i in Loss]}", end="")

        for i in range(Parm.task_number):
            print(f"Accuracy: {testing(model_fusion, Tasks[i].test_loader, Parm) :.5f}", end=" |")
        
        if Test_loader != None:
            save.append(testing(model_fusion, Test_loader, Parm))
            print(f"Total:{save[-1] :.5f}",end="")
            
        print("")
    return model_fusion, save

#%%
def :




def fine_tune(fusion_model, Tasks, choose_type='kd', layer_wise=False):
    if choose_type == 'unsupervise':
        
    elif choose_type == 'supervise':
        pass
    elif choose_type == 'kd':
        pass