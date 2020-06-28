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
        x = self.network
        return x
    
    @property
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[3], self.network[6]]
        return net










