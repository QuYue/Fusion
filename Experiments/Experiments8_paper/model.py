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

#%% FNN model
class FNN1(nn.Module): # MNIST
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
        net = [self.network[1], self.network[4], lambda x: x]
        return net


class FNN2(nn.Module): # MNIST Poor
    def __init__(self):
        super(FNN2, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(28*28, 30),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(30, 30),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(30, 10),)
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
        net = [self.network[1], self.network[4], lambda x: x]
        return net
    

class FNN3(nn.Module): # CIFAR 8
    def __init__(self):
        super(FNN3, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(3*32*32, 1000),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(500, 8),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x
    
    @property
    def plug_net(self): # network need plugins
        net = [self.network[0], self.network[3], self.network[6], self.network[9]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.network[1], self.network[4], self.network[7], lambda x: x]
        return net

#%% CNN1 model
class CNN1(nn.Module): # MNIST 
    def __init__(self):
        super(CNN1, self).__init__()
        self.Conv2d = nn.Sequential(       # Input: N * 1 * 28 * 28 (784N * 10)
                nn.Conv2d(in_channels=1,   # Output: N * 32 * 28 * 28 (784N * 32)
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 3, 1, 1),  # Input: N * 32 * 14 * 14 (196N * 289)
                nn.ReLU(),                   # Output: N * 32 * 14 * 14 (196N * 32)
                nn.MaxPool2d(2),
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(32*7*7, 256),
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
        net = [self.Conv2d[0], self.Conv2d[4], self.network[0], self.network[3], self.network[6]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.network[1], self.network[4], lambda x: x]
        return net


class CNN2(nn.Module): # MNIST kernel 5
    def __init__(self):
        super(CNN2, self).__init__()
        self.Conv2d = nn.Sequential(       # Input: N * 1 * 28 * 28 (784N * 26)
                nn.Conv2d(in_channels=1,   # Output: N * 32 * 28 * 28 (784N * 32)
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 5, 1, 2),  # Input: N * 32 * 14 * 14 (196N * 801)
                nn.ReLU(),                   # Output: N * 32 * 14 * 14 (196N * 32)
                nn.MaxPool2d(2),
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(32*7*7, 256),
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
        net = [self.Conv2d[0], self.Conv2d[4], self.network[0], self.network[3], self.network[6]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.network[1], self.network[4], lambda x: x]
        return net


class CNN3(nn.Module): # MNIST kernel 5 stride 2 without maxpooling
    def __init__(self):
        super(CNN3, self).__init__()
        self.Conv2d = nn.Sequential(       # Input: N * 1 * 28 * 28 (196N * 26)
                nn.Conv2d(in_channels=1,   # Output: N * 32 * 14 * 14 (196N * 32)
                        out_channels=32,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 5, 2, 2),  # Input: N * 32 * 14 * 14 (49N * 801)
                nn.ReLU(),                   # Output: N * 32 * 7 * 7 (49N * 32)
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(32*7*7, 256),
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
        net = [self.Conv2d[0], self.Conv2d[3], self.network[0], self.network[3], self.network[6]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[4], self.network[1], self.network[4], lambda x: x]
        return net


class CNN4(nn.Module): # CIFAR 8
    def __init__(self):
        super(CNN4, self).__init__() 
        self.Conv2d = nn.Sequential(           # Input: N * 3 * 32 * 32 (1024N * 28)
                nn.Conv2d(in_channels=3,       # Output: N * 32 * 32 * 32 (1024N * 32)
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1),           
                nn.ReLU(),
                nn.MaxPool2d(2),              
                nn.Dropout(0.5),
                nn.Conv2d(32, 64, 3, 1, 1),    # Input: N * 32 * 16 * 16 (256N * 289)
                nn.ReLU(),                     # Output: N * 64 * 16 * 16 (256N * 64)
                nn.MaxPool2d(2),               
                nn.Dropout(0.5), 
                nn.Conv2d(64, 128, 3, 1, 1),   # Input: N * 64 * 8 * 8 (64N * 577)
                nn.ReLU(),                     # Output: N * 128 * 8 * 8 (64N * 128)
                nn.MaxPool2d(2),             
                nn.Dropout(0.5),
                nn.Conv2d(128, 128, 3, 1, 1),  # Input: N * 128 * 4 * 4 (16N * 1153)
                nn.ReLU(),                     # Output: N * 128 * 4 * 4 (16N * 128)
                nn.MaxPool2d(2),               
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(128*2*2, 400), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(400, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 8),)
    def forward(self, x):
        x = self.Conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.Conv2d[0], self.Conv2d[4], self.Conv2d[8], self.Conv2d[12], self.network[0], self.network[3], self.network[6], self.network[9]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.Conv2d[9], self.Conv2d[13], self.network[1], self.network[4], self.network[7], lambda x: x]
        return net


class CNN5(nn.Module): # CIFAR 8
    def __init__(self):
        super(CNN5, self).__init__() 
        self.Conv2d = nn.Sequential(           # Input: N * 3 * 32 * 32 (1024N * 28)
                nn.Conv2d(in_channels=3,       # Output: N * 32 * 32 * 32 (1024N * 32)
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1),           
                nn.ReLU(),
                nn.MaxPool2d(2),              
                nn.Dropout(0.5),
                nn.Conv2d(16, 32, 3, 1, 1),    # Input: N * 32 * 16 * 16 (256N * 289)
                nn.ReLU(),                     # Output: N * 64 * 16 * 16 (256N * 64)
                nn.MaxPool2d(2),               
                nn.Dropout(0.5), 
                nn.Conv2d(32, 32, 3, 1, 1),   # Input: N * 64 * 8 * 8 (64N * 577)
                nn.ReLU(),                     # Output: N * 128 * 8 * 8 (64N * 128)
                nn.MaxPool2d(2),             
                nn.Dropout(0.5),
                nn.Conv2d(32, 32, 3, 1, 1),   # Input: N * 64 * 8 * 8 (64N * 577)
                nn.ReLU(),                     # Output: N * 128 * 8 * 8 (64N * 128)
                nn.MaxPool2d(2),             
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(128, 30), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(30, 4),)
    def forward(self, x):
        x = self.Conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.Conv2d[0], self.Conv2d[4], self.Conv2d[8], self.Conv2d[12], self.network[0], self.network[3]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.Conv2d[9], self.Conv2d[13], self.network[1], lambda x: x]
        return net


#%% CNN5 model
class CNN5(nn.Module): # CNN1
    def __init__(self):
        super(CNN5, self).__init__()
        self.Conv2d = nn.Sequential( 
                nn.Conv2d(in_channels=3, 
                          out_channels=32,
                          kernel_size=5,
                          stride=1,
                          padding=2),           # N * 16 * 300 * 300
                nn.ReLU(),
                nn.MaxPool2d(2),               # N * 16 * 150 * 150
                nn.Dropout(0.5),
                nn.Conv2d(32, 64, 5, 1, 2),    # N * 32 * 150 * 150
                nn.ReLU(),                     
                nn.MaxPool2d(2),               # N * 32 * 75 * 75
                nn.Dropout(0.5), 
                nn.Conv2d(64, 128, 5, 1, 2),    # N * 64 * 75 * 75 
                nn.ReLU(),
                nn.MaxPool2d(2),               # N * 64 * 37 * 37
                nn.Dropout(0.5),)
        self.network = nn.Sequential(
                nn.Linear(2048, 1000), 
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1000, 400),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(400, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(100, 4),)
    def forward(self, x):
        x = self.Conv2d(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x

    @property
    def plug_net(self): # network need plugins
        net = [self.Conv2d[0], self.Conv2d[4], self.Conv2d[8], self.network[0], self.network[3], self.network[6], self.network[9]]
        return net

    @property
    def plug_nolinear(self): # nolinear network
        net = [self.Conv2d[1], self.Conv2d[5], self.Conv2d[9], self.network[1], self.network[4], self.network[7], lambda x: x]
        return net
    
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#%%
if __name__ == "__main__":
    data1 = torch.ones([10, 1, 28, 28])
    #%%
    model = FNN1()
    target = model(data1)
    model = FNN2()
    target = model(data1)
    model = CNN1()
    target = model(data1)
    model = CNN2()
    target = model(data1)
    model = CNN3()
    target = model(data1)
    #%%
    data2 = torch.ones([10, 3, 32, 32])
    model = FNN3()
    target = model(data2)
    model = CNN4()
    target = model(data2)
    model = CNN5()
    target = model(data2)
    print(target.shape)




