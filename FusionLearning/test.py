#%% 
import torch
import torch.nn as nn
from Plugin import Plugin
import copy

#%%
class FNN(nn.Module): # FNN1
    def __init__(self):
        super(FNN, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(10, 12),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(12, 2),)
    def forward(self, x):
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[3], self.network[6]]
        return net


#%%
fnn = FNN()
model = Plugin(fnn)
model.plugin_hook()
model.model.eval()
# d =  torch.randn([5, 30])
# print(model.model(d))
# Y1 = copy.deepcopy(model.Y)
# model.__normalization(torch.Tensor([1,2,1,2,1,2,1,2,1,2]), 0)
# model.__normalization(torch.Tensor([1,2,1,2,1,2,1,2,1,2,1,2]), 1)
# print(model.model(d))
# Y2 = copy.deepcopy(model.Y)

#%%
data = torch.randn([5, 30])
print(model.forward(data))
Y1 = copy.deepcopy(model.Y)
weight = model.Normalization(data)
print(model.forward(data))
Y2 = copy.deepcopy(model.Y)