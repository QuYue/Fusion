# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/27 20:36
@Author  : QuYue
@File    : model.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torch
import torch.nn as nn
#%%
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.ifplug = False
        self.layer1 = nn.Linear(28 * 28, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100,100)
        self.layer3 = nn.Linear(100,10)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu(x)
        self.dropout1 = nn.Dropout(0.5)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

    def get_W(self):
        self.W = {}
        keys = list(self.parm)
        for key in keys:
            w = self.parm[key]['weight']
            b = self.parm[key]['bias']
            self.W[key] = torch.cat([w.transpose(1, 0).data, b.unsqueeze(0).data])
        return self.W

    def W_plug(self):
        keys = list(self.parm)
        for key in keys:
            [m, n] = self.W[key].shape
            self.parm[key]['weight'].data = self.W[key][:m-1, :].transpose(1,0)
            self.parm[key]['bias'].data = self.W[key][m-1,:]


    def plug_in(self, ifhook=True):
        self.W = {}
        self.X = {}
        self.Y = {}
        self.parm = {}
        self.ifplug = True
        def get_parm(layer):
            parm = dict(layer.named_parameters())
            return parm
        def get_X(input_data):
            x = input_data[0].data
            X = torch.cat([x, torch.ones([x.shape[0], 1], device=x.device)], 1)
            return X
        def get_Y(output_data):
            Y = output_data.data
            return Y
        def get_hooks(name):
            def hook(model, input_data, output_data):
                self.X[name] = get_X(input_data)
                self.Y[name] = get_Y(output_data)
            return hook
        def plugin(layers, ifhook):
            for i, layer in enumerate(layers):
                name = f'Linear{i}'
                self.parm[name] = get_parm(layer)
                if ifhook:
                    layer.register_forward_hook(get_hooks(name))

        plugin([self.layer1, self.layer2, self.layer3], ifhook)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x

class FNN2(nn.Module):
    def __init__(self):
        super(FNN2, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(28*28, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 10),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x

    @property
    def plug_net(self): # 线性层
        net = [self.network[0], self.network[3], self.network[6]]
        return net


# Fusionable Plugins
class Fusion_plugin(object):
    def __init__(self, model):
        self.model = model # input model by pytorch
        self.plug_net = self.model.plug_net

        self.ifParm = False # 是否参数提取出来 If extract parameter
        self.ifhook = False # 是否有前向hook   If forward hook
        self.norm = False   # 是否经过了norm   If layer normalized
        self.rank = 'No'    # 排序类型

        self.plugin_Parm() # Plugin Parameters

    @property
    def plugin_manager(self):
        manager = { 'ifParm': self.ifParm,
                    'ifhook': self.ifhook,
                    'norm': self.norm,
                    'rank': self.rank}
        return manager

    @property
    def plug_net_name(self):
        name = []
        for i, layer in enumerate(self.plug_net):
            name.append(f'Linear{i}')
        return name

    # Plugin Parameters
    def plugin_Parm(self):
        def get_parm(layer):
            parm = dict(layer.named_parameters())
            return parm
        def plugin(layers):
            name = self.plug_net_name
            for i, layer in enumerate(layers):
                self.parm[name[i]] = get_parm(layer)
        self.parm = {}
        self.ifParm = True
        plugin(self.plug_net)
        self._get_W()

    # Plugin forwark hook
    def plugin_hook(self):
        def get_X(input_data):
            x = input_data[0].data
            X = torch.cat([x, torch.ones([x.shape[0], 1], device=x.device)], 1)
            return X
        def get_Y(output_data):
            Y = output_data.data
            return Y
        def get_hooks(name):
            def hook(model, input_data, output_data):
                self.X[name] = get_X(input_data)
                self.Y[name] = get_Y(output_data)
            return hook
        def plugin(layers):
            name = self.plug_net_name
            for i, layer in enumerate(layers):
                layer.register_forward_hook(get_hooks(name[i]))
        self.X = {}
        self.Y = {}
        self.ifhook = True
        plugin(self.plug_net)

    @property
    def W(self):
        return self._W

    def _get_W(self):
        self._W = {}
        keys = list(self.parm)
        for key in keys:
            w = self.parm[key]['weight']
            b = self.parm[key]['bias']
            self._W[key] = torch.cat([w.transpose(1, 0).data, b.unsqueeze(0).data])

    def W_update(self, W):
        keys = list(self.parm)
        for key in keys:
            [m, n] = W[key].shape
            self.parm[key]['weight'].data = W[key][:m - 1, :].transpose(1, 0)
            self.parm[key]['bias'].data = W[key][m - 1, :]
        self._get_W()

#%%
def average_fusion(models, model_fusion):
    def average(nets):
        aver_net = nets[0].clone()
        for i in range(1, len(nets)):
            aver_net += nets[i].data
        aver_net /= len(nets)
        return aver_net

    layers0 = list(model_fusion.model.named_parameters())
    layers = [list(model.model.named_parameters()) for model in models]

    for i in range(len(layers0)):
        layers0[i][1].data = average([model[i][1].data for model in layers])
    return model_fusion
#%%
def pinv_fusion(model1, model2, model_fusion, data1, data2):
    if model1.ifplug == False:
        model1.plugin_hook()
    if model2.ifplug == False:
        model2.plugin_hook()

    model1.model(data1)
    model2.model(data2)
    W = model_fusion.W
    W1 = model1.W
    W2 = model2.W
    X1 = model1.X
    X2 = model2.X

    layers = list(W1.keys())
    for layer in layers:
        Z1 = X1[layer].transpose(1,0).mm(X1[layer])
        Z2 = X2[layer].transpose(1,0).mm(X2[layer])
        Z = Z1 + Z2
        H = Z1.mm(W1[layer]) + Z2.mm(W2[layer])
        W[layer].data = Z.pinverse().mm(H)
    model_fusion.W_update(W)
    return model_fusion
#%%
def Linear_fusion3(model1, model2, model_fusion, data1, data2, step=0.01):
    if model1.ifplug == False:
        model1.plugin_hook()
    if model2.ifplug == False:
        model2.plugin_hook()
    model1(data1)
    model2(data2)
    W = model_fusion.W
    W1 = model1.W
    W2 = model2.W
    X1 = model1.X
    X2 = model2.X

    layers = list(W1.keys())
    loss = []
    for layer in layers:
        Z1 = X1[layer].transpose(1,0).mm(X1[layer])
        Z2 = X2[layer].transpose(1,0).mm(X2[layer])
        Z = Z1 + Z2
        H = Z1.mm(W1[layer]) + Z2.mm(W2[layer])
        loss.append(torch.norm((Z.mm(W[layer])-H), p='fro').data.cpu())
        grad = Z.transpose(1,0).mm(Z.mm(W[layer])-H)
        W[layer] -= step * grad
    model_fusion.W_update(W)
    print(f"loss:{[i.data for i in loss]}", end="")
    return model_fusion
#%%
def oneshot_rank(model, Parm):
    def rank(W1, W2, layer, data):
        output = layer(data)
        _, index = output[0].sort()
        W1 = W1[:, index]
        W2[:-1, :] = W2[:-1, :][index, :]
        return W1, W2
    def create_data(Parm, dim):
        data = torch.ones([1, dim])
        if Parm.cuda:
            data = data.cuda()
        return data

    layer_list = list(model.W.keys())
    for i in range(len(layer_list)-1):
        name1 = layer_list[i]
        name2 = layer_list[i+1]
        layer1 = model.plug_net[i]
        W = model.W
        W1 = W[name1]
        W2 = W[name2]
        data = create_data(Parm, W1.shape[0]-1)
        W[name1], W[name2] = rank(W1, W2, layer1, data)
        model.W_update(W)
    return model

def out_norm(model, data, Parm):


#%%
if __name__ == '__main__':
    class Parameter():
        def __init__(self):
            self.cuda = False
    Parm = Parameter()
    fnn_1 = FNN2()
    fnn_2 = FNN2()
    fnn_f1 = Fusion_plugin(fnn_1)
    fnn_f2 = Fusion_plugin(fnn_2)
    data = torch.randn([10, 28*28])
    oneshot_rank(fnn_f1, Parm)




