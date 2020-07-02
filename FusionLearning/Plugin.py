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

#%% Plugin 
class Plugin(object):
    def __init__(self, model):
        self.model = model # input model by pytorch
        self.plug_net = self.model.plug_net # network need plugins
        self.ifsynapse = False # If extract synapse
        self.ifhook = False # If forward hook
        self.norm = False # If layer normalized
        self.rank = 'No' # method for ranking
        self.plug_synapse() # Plugin synapse
        self.forward = self.model.forward
        self.train = self.model.train
        self.eval = self.model.eval
        
        # Plugin Manager 
    @property
    def plugin_manager(self):
        manager = {'ifsynapse': self.ifsynapse,
                    'ifhook': self.ifhook,
                    'norm': self.norm,
                    'rank': self.rank}
        return manager

    # Plugin network name
    @property
    def plug_net_name(self):
        name = []
        for i, layer in enumerate(self.plug_net):
            name.append(f'Linear{i}')
        return name

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
        keys = list(self.synapse)
        for key in keys:
            w = self.synapse[key]['weight']
            b = self.synapse[key]['bias']
            self._W[key] = torch.cat([w.transpose(1, 0).data, b.unsqueeze(0).data])
    
    @property
    def W(self):
        # W, which is a matrix for restoring synapses
        return self._W

    def W_update(self, new_W):
        # Update W and synapses
        keys = list(self.synapse)
        for key in keys:
            # [m, n] = new_W[key].shape
            self.synapse[key]['weight'].data = new_W[key][:-1, :].transpose(1, 0)
            self.synapse[key]['bias'].data = new_W[key][-1, :]
        self._get_W()
    
    # Plugin forward hook
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
        


    

        


            




    
    






















