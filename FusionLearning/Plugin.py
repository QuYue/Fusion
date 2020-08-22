# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/27 19:41:49
@Author      :Qu Yue
@File        :Plugin.py
@Software    :Visual Studio Code
Introduction: Fusion Plugin
'''
#%% Import Packages
import torch
import numpy as np

#%% Plugin 
class Plugin(object):
    def __init__(self, model, cpu=False):
        self.model = model # input model by pytorch
        self.cpu = cpu
        self.plug_net = self.model.plug_net # network need plugins
        self.plug_nolinear = self.model.plug_nolinear # nolinear network
        self.ifsynapse = False # If extract synapse
        self.ifhook = False # If forward hook
        self.ify_grad = False # If y grad
        self.norm = False # If layer normalized
        self.rank = 'No' # method for ranking
        self.plug_synapse() # Plugin synapse
        self.forward = self.model.forward
        self.train = self.model.train
        self.eval = self.model.eval
        self.parameters = self.model.parameters
        
    # Plugin Manager 
    @property
    def plugin_manager(self): 
        manager = {'ifsynapse': self.ifsynapse,
                   'ifhook': self.ifhook,
                   'ify_grad': self.ify_grad,
                   'norm': self.norm,
                   'rank': self.rank}
        return manager

    # Plugin network name
    @property
    def plug_net_name(self):
        def get_type(layer):
            type_name = 'None'
            if 'Linear' in str(layer):
                type_name = 'Linear'
            elif 'Conv' in str(layer):
                type_name = 'Conv'
            return type_name
        names = []
        type_count = dict()
        for layer in self.plug_net:
            name = get_type(layer)
            if name in type_count:
                type_count[name] += 1
            else:
                type_count[name] = 1
            names.append(f'{name}{type_count[name]}')
        return names

    # Plugin netwark synapse
    def plug_synapse(self):
        def get_synapse(layer):
            synapse = dict(layer.named_parameters())
            return synapse
        def plugin(layers):
            name = self.plug_net_name
            for i, layer in enumerate(layers):
                self.synapse[name[i]] = get_synapse(layer)
        self.synapse = {}
        self.ifsynapse = True
        plugin(self.plug_net)
        self._get_W()
    
    def _get_W(self):
        self._W = {}
        keys = self.plug_net_name
        for key in keys:
            w = self.synapse[key]['weight']
            b = self.synapse[key]['bias']
            if 'Conv' in key:
                w = w.view(w.shape[0], -1)
            self._W[key] = torch.cat([w.transpose(1, 0).data, b.unsqueeze(0).data])
            if self.cpu:
                self._W[key] = self._W[key].cpu()

    @property
    def W(self):
        # W, which is a matrix for restoring synapses
        self._get_W()
        return self._W

    def W_update(self, new_W):
        # Update W and synapses
        keys = list(self.synapse)
        for key in keys:
            device = self.synapse[key]['weight'].device
            if 'Conv' in key:
                shape = self.synapse[key]['weight'].shape
                self.synapse[key]['weight'].data = new_W[key][:-1, :].transpose(1, 0).data.reshape(shape).to(device)
            elif 'Linear' in key:
                self.synapse[key]['weight'].data = new_W[key][:-1, :].transpose(1, 0).data.to(device)
            self.synapse[key]['bias'].data = new_W[key][-1, :].to(device)
        self._get_W()
    
    def _get_grad(self):
        self._grad = {}
        keys = list(self.synapse)
        for key in keys:
            w = self.synapse[key]['weight'].grad
            b = self.synapse[key]['bias'].grad
            if 'Linear' in key:
                self._grad[key] = torch.cat([w.transpose(1, 0).data, b.unsqueeze(0).data])   
            elif 'Conv' in key:
                self._grad[key] = self.kernel(self.synapse[key])

    @property
    def grad(self):
        # grad of W
        self._get_grad()
        return self._grad

    def conv(self, x, kernel_size, stride, padding):
        padding_func = torch.nn.ZeroPad2d((padding[0], padding[0], padding[1], padding[1]))
        x = padding_func(x)
        X = torch.empty([((x.shape[-2]-kernel_size[0]+1)//stride[0])*((x.shape[-1]-kernel_size[1]+1)//stride[1]),
                    x.shape[0], x.shape[1], kernel_size[0], kernel_size[1]]).to(x.device)
        channel = x.shape[1]
        count = 0
        for i in range(0, x.shape[-2]-kernel_size[0]+1, stride[0]):
            for j in range(0, x.shape[-1]-kernel_size[1]+1, stride[1]):
                data = x[:, :, i:i+kernel_size[0], j:j+kernel_size[1]]
                X[count] = data
                count+=1
        X = X.permute(1,0,2,3,4)
        X = X.reshape(-1, channel*kernel_size[0]*kernel_size[1])
        return X

    # Plugin forward hook
    def plugin_hook(self, y_grad=False):
        def get_X(input_data, model):
            x = input_data[0].data if self.cpu == False else input_data[0].data.cpu()
            if 'Conv' in str(model):
                x = self.conv(x, model.kernel_size, model.stride, model.padding)
            X = torch.cat([x, torch.ones([x.shape[0], 1], device=x.device)], 1)
            return X
        def get_Y(output_data):
            Y = output_data.data  if self.cpu == False else output_data.data.cpu()
            return Y
        def get__Y(output_data):
            return output_data
        def get_hooks(name):
            def hook(model, input_data, output_data):
                self.X[name], self.Y[name] = 0, 0
                self.X[name] = get_X(input_data, model)
                self.Y[name] = get_Y(output_data)
                if self.ify_grad: self._Y[name] = get__Y(output_data)
            return hook
        def plugin(layers):
            name = self.plug_net_name
            for i, layer in enumerate(layers):
                layer.register_forward_hook(get_hooks(name[i]))
     
        self.ify_grad = y_grad
        self.X = {}
        self.Y = {}
        if self.ify_grad: self._Y = {}
        self.ifhook = True
        plugin(self.plug_net)

    # Oneshot Rank
    def oneshot_rank(self, Parm):
        def rank(W1, W2, layer, data):
            output = layer(data)
            _, index = output[0].sort()
            W1 =  W1[:, index]
            W2[:-1, :] = W2[:-1, :][index, :]
            return W1, W2
        def create_data(Parm, dim):
            data = torch.ones([1, dim])
            if Parm.cuda:
                data = data.cuda()
            return data

        layer_list = list(self.W.keys())
        for i in range(len(layer_list)-1):
            name1 = layer_list[i]
            name2 = layer_list[i+1]
            layer1 = self.plug_net[i]
            W = self.W
            W1 = W[name1]
            W2 = W[name2]
            data = create_data(Parm, W1.shape[0]-1)
            W[name1], W[name2] = rank(W1, W2, layer1, data)
            self.W_update(W)
        self.rank = 'OneShot'
        
    # Normalization
    def __normalization(self, weight, layer_number):
        def layer_change(W1, W2, weight):
            W1 = W1 * weight
            W2[:-1, :] = (W2[:-1, :].transpose(1,0) / weight).transpose(1,0)
            return W1, W2
        layer_list = list(self.W.keys())
        name1 = layer_list[layer_number]
        name2 = layer_list[layer_number+1]
        W = self.W
        W1 = W[name1]
        W2 = W[name2]
        W[name1], W[name2] = layer_change(W1, W2, weight)
        self.W_update(W)
        
    def Normalization(self, data, Parm, type='max', scale=1.0):
        if Parm.cuda:
            data = data.cuda()
        layer_list = list(self.W.keys())
        weight = []
        for i in range(len(layer_list)-1):
            label = self.model(data)
            Y = self.Y[layer_list[i]]
            # 使用最大
            if type == 'max':
                max = torch.max(Y, 0).values
            elif type == 'max_abs':
                max = torch.max(torch.abs(Y),0).values
            else:
                max = torch.ones([Y.shape[0]])
            max[max<=0] = scale
            weight.append(scale/max)
            self.__normalization(weight[-1], i)
        self.norm = True
        return weight

    # Empty X and Y
    def empty_x_y(self):
        self.X = {}
        self.Y = {}
        torch.cuda.empty_cache()

