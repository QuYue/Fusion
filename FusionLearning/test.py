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
                #nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(10, 12),
                #nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(12, 2),)
    def forward(self, x):
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[1], self.network[2]]
        return net


#%%
fnn = FNN()
model = Plugin(fnn)
model.plugin_hook()
model.model.eval()
d = torch.ones([10, 30])
print(model.model(d))
Y1 = copy.deepcopy(model.Y)
model.Normlization(torch.Tensor([1,2,1,2,1,2,1,2,1,2]), 0, 0)
print(model.model(d))
Y2 = copy.deepcopy(model.Y)


