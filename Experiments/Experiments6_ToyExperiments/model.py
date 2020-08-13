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

class FNN2(nn.Module): # FNN1
    def __init__(self):
        super(FNN2, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(28*28, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 10),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x
    
    @property
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[3]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.network[1], torch.nn.Softmax()]
        return net


#%% CNN1 model
class CNN1(nn.Module): # CNN1
    def __init__(self):
        super(CNN1, self).__init__()
        self.Conv2d = nn.Sequential(       # Input: N * 1 * 28 * 28 (784N * 10)
                nn.Conv2d(in_channels=1,   # Output: N * 16 * 28 * 28  (784N * 16)
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 3, 1),  # Input: N * 16 * 14 * 14 (196N * 145)
                nn.ReLU(),                # Output: N * 32 * 14 * 14 (196N * 32)
                nn.MaxPool2d(2),
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(32*6*6, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                # nn.Linear(512, 256),
                # nn.ReLU(),
                # nn.Dropout(0.5),
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
        net = [self.Conv2d[0], self.Conv2d[4], self.network[0], self.network[3], self.network[6]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.network[1], self.network[4], torch.nn.Softmax()]
        return net

#%% CNN1 model
class CNN2(nn.Module): # CNN1
    def __init__(self):
        super(CNN2, self).__init__()
        self.Conv2d = nn.Sequential(       # Input: N * 1 * 28 * 28 (784N * 10)
                nn.Conv2d(in_channels=1,   # Output: N * 16 * 28 * 28  (784N * 16)
                        out_channels=16,
                        kernel_size=5,
                        stride=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(16, 32, 5, 1),  # Input: N * 16 * 14 * 14 (196N * 145)
                nn.ReLU(),                # Output: N * 32 * 14 * 14 (196N * 32)
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 5, 1),  # Input: N * 16 * 14 * 14 (196N * 145)
                nn.ReLU(),                # Output: N * 32 * 14 * 14 (196N * 32)
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(8192, 512),
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
        net = [self.Conv2d[0], self.Conv2d[3], self.Conv2d[6], self.network[0], self.network[3], self.network[6], self.network[9]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[4], self.Conv2d[7], self.network[1], self.network[4], self.network[7], torch.nn.Softmax()]
        return net


#%%
if __name__ == "__main__":
    data = torch.ones([10, 1, 28, 28])
    model1 = FNN1()
    target = model1(data)
    model2 = CNN1()
    target = model2(data)
    print(target.shape)




