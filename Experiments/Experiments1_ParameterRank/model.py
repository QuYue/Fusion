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
        self.layer1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

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
        self.z = nn.Sequential(
                nn.Linear(28*28, 1000),
                nn.ReLU(),
                nn.Linear(1000, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.z(x)
        return x



def par_fusion(model1, model2, model0):
    def average(p1, p2):
        p = (p1 + p2)/2
        return p

    net0 = list(model0.named_parameters())
    net1 = list(model1.named_parameters())
    net2 = list(model2.named_parameters())

    for i in range(len(net0)):
        net0[i][1].data = average(net1[i][1].data, net2[i][1].data)
    return model0


#%%
if __name__ == '__main__':
    fnn = FNN()
    cnn = CNN()
    x = torch.ones(5, 1, 28, 28)
    y = fnn(x)
    print(y)
    y = cnn(x)
    print(y)

