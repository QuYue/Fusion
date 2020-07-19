# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/27 11:41:19
@Author      :Qu Yue
@File        :model.py
@Software    :Visual Studio Code
Introduction: The models
'''
#%% Import Packages
import torch
import torch.nn as nn
#%% FNN1 model
class FNN1(nn.Module): # FNN1
    def __init__(self):
        super(FNN1, self).__init__()
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
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[3], self.network[6]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.network[1], self.network[4], torch.nn.Softmax()]
        return net

#%% CNN1 model
class CNN1(nn.Module): # CNN1
    def __init__(self):
        super(CNN1, self).__init__()
        self.Conv2d = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(32*7*7, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 10),)

    def forward(self, x):
        x = self.Conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.Conv2d[0], self.Conv2d[4], self.network[0], self.network[3], self.network[6], self.network[9]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.network[1], self.network[4], self.network[7], torch.nn.Softmax()]
        return net

#%%
if __name__ == "__main__":
    data = torch.ones([10, 1, 28, 28])
    model1 = FNN1()
    target = model1(data)
    model2 = CNN1()
    target = model2(data)
    print(target.shape)





# %%
