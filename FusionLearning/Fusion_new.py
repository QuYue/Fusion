# -*- encoding: utf-8 -*-
'''
@Time        :2020/07/01 20:07:56
@Author      :Qu Yue
@File        :Fusion.py
@Software    :Visual Studio Code
Introduction: Fusion with Forgetting Events.
'''
#%% Import Packages
import torch
import numpy as np
import copy
import torch.utils.data as Data
from torch.nn import functional as F
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
def pinv_fusion(Tasks, model_fusion, Parm, ifbatch=True, ifweight=False, lambda_list=None, rcond=1e-4):
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

#%%
def AF_rank(Tasks, Parm, ifbatch=True):
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
def interfere_sort(sort_list):
    index_list = np.argsort(sort_list, axis=1)
    return index_list

def Interfere_AF_rank(Tasks, Parm, ifbatch=True):
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
        l_sort = interfere_sort(sort_list)
        for j in range(len(W_s)):
            W_s[j][layers[i]], W_s[j][layers[i+1]] = rank(W_s[j][layers[i]], W_s[j][layers[i+1]], l_sort[j])

    for i in range(len(W_s)):
        Tasks[i].model.W_update(W_s[i])
        Tasks[i].model.empty_x_y()
    return Tasks


def Interfere_MAN_rank(Tasks, Parm, ifbatch=True):
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
        l_sort = interfere_sort(sort_list)
        for j in range(len(W_s)):
            W_s[j][layers[i]], W_s[j][layers[i+1]] = rank(W_s[j][layers[i]], W_s[j][layers[i+1]], l_sort[j])

    for i in range(len(W_s)):
        Tasks[i].model.W_update(W_s[i])
        Tasks[i].model.empty_x_y()
    return Tasks


#%%
def hard_loss(predict_y, y, loss_function):
    loss_func = loss_function['CE']
    loss = loss_func(predict_y, y)
    return loss

def soft_loss(predict_y, teacher_y, loss_function, T):
    loss_func = loss_function['KL']
    loss = loss_func(F.log_softmax(predict_y/T, dim=1), F.softmax(teacher_y/T, dim=1))
    loss *= T**2
    return loss

def soft_loss_layer_wise(Y, teacher_Y, loss_function, T):
    loss = 0
    for layer in Y:
        predict_y = Y[layer]
        teacher_y = teacher_Y[layer]
        loss += soft_loss(predict_y, teacher_y, loss_function, T)
    return loss

def soft_loss_layer_wise2(Y, teacher_Y, loss_function, nolinear, T):
    loss = 0
    for layer in Y:
        predict_y = nolinear[0][layer](Y[layer])
        teacher_y = nolinear[1][layer](teacher_Y[layer])
        loss += soft_loss(predict_y, teacher_y, loss_function, T)
    return loss



#%%
def fine_tune(Fusion_task, Tasks, Parm, choose_type='kd', Lambda=0.5):
    def remove(need_list, remove_list):
        for i in remove_list:
            need_list.remove(i)
    loss_function = {'CE': torch.nn.CrossEntropyLoss(), 'KL': torch.nn.KLDivLoss()}
    train_loader_list = [iter(Task.train_loader) for Task in Tasks]
    available = list(range(len(Tasks)))
    for Task in Tasks:
        Task.model.eval()
    if choose_type in ['unsupervise_layer','kd_layer','kd_layer2']:
        Fusion_task.model.plugin_hook(True,False,False)

    fusion_model = Fusion_task.model
    optimizer = Fusion_task.optimizer
    while len(available) != 0:
        remove_list = []
        data_count = 0
        loss = 0
        for i in available:
            try:
                x, y = next(train_loader_list[i])
            except:
                remove_list.append(i)
                continue
            if Parm.cuda == True: x, y = x.cuda(), y.cuda()
            fusion_model.train()
            data_count += y.shape[0]
            if choose_type == 'unsupervise_layer':
                fusion_model.eval()
                predict_y = fusion_model.forward(x)
                teacher_y = Tasks[i].model.forward(x)
                Y = fusion_model._Y
                teacher_Y = Tasks[i].model.Y
                loss0 = soft_loss_layer_wise(Y, teacher_Y, loss_function, Parm.T)
            elif choose_type == 'unsupervise':
                fusion_model.train()
                predict_y = fusion_model.forward(x)
                teacher_y = Tasks[i].model.forward(x)
                loss0 = soft_loss(predict_y, teacher_y, loss_function, Parm.T)
            elif choose_type == 'supervise':
                fusion_model.train()
                predict_y = fusion_model.forward(x)
                loss0 = hard_loss(predict_y, y, loss_function)
            elif choose_type == 'kd':
                fusion_model.train()
                predict_y = fusion_model.forward(x)
                teacher_y = Tasks[i].model.forward(x)
                loss_s = soft_loss(predict_y, teacher_y, loss_function, Parm.T)
                loss_h = hard_loss(predict_y, y, loss_function)
                loss0 = Lambda * loss_s + (1-Lambda) * loss_h
            elif choose_type == 'kd_layer':
                fusion_model.eval()
                predict_y = fusion_model.forward(x)
                Y = fusion_model._Y
                teacher_y = Tasks[i].model.forward(x)
                teacher_Y = Tasks[i].model.Y
                loss_s = soft_loss_layer_wise(Y, teacher_Y, loss_function, Parm.T)
                fusion_model.train()
                predict_y = fusion_model.forward(x)
                loss_h = hard_loss(predict_y, y, loss_function)
                loss0 = Lambda * loss_s + (1-Lambda) * loss_h
            elif choose_type == 'kd_layer2':
                fusion_model.eval()
                predict_y = fusion_model.forward(x)
                Y = fusion_model._Y
                teacher_y = Tasks[i].model.forward(x)
                teacher_Y = Tasks[i].model.Y
                loss_s = soft_loss_layer_wise2(Y, teacher_Y, loss_function, [Tasks[i].model._plug_nolinear, fusion_model._plug_nolinear], Parm.T)
                fusion_model.train()
                predict_y = fusion_model.forward(x)
                loss_h = hard_loss(predict_y, y, loss_function)
                loss0 = Lambda * loss_s + (1-Lambda) * loss_h
            loss += loss0*data_count
            if Parm.cuda: torch.cuda.empty_cache()  # empty GPU memory
        remove(available, remove_list)
        if data_count != 0:
            optimizer.zero_grad()
            loss /= data_count
            loss.backward()
            optimizer.step()
            del x, y, predict_y
            torch.cuda.empty_cache()
    

