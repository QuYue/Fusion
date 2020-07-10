# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/13 20:18
@Author  : QuYue
@File    : data_input.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torchvision
from sklearn.model_selection import train_test_split

#%% Functions
def data_input(dataset_no=1, download=True):
    # Input Data
    if dataset_no == 1 or 'Disjoint_MNIST':
        dataset_no = 1
        print('Loading dataset 1: Disjoint_MNIST...')
        data_train, data_test = MNIST_input(download)
    elif dataset_no == 2 or 'Split_MNIST':
        dataset_no = 2
        print('Loading dataset 2: Split_MNIST...')
    elif dataset_no == 3 or ''

    else:
        print('Please input right dataset number.')
        return None, None
    print('Data is ready.')
    return data_train, data_test


def MNIST_input(download=True):
    # Data1 MNIST
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (1.0,))])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                            train=True,
                                            transform=trans,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                           train=False,
                                           transform=trans,
                                           download=download)
    return data_train, data_test

def CommodityImage():
    # Data2 CommodityImage
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()],)
    data_1 = torchvision.datasets.ImageFolder(root="../../Data/CommodityImage/amazon",
                                              transform=trans)
    data_2 = torchvision.datasets.ImageFolder(root="../../Data/CommodityImage/caltech",
                                              transform=trans)
    data_3 = torchvision.datasets.ImageFolder(root="../../Data/CommodityImage/Datasets",
                                              transform=trans)
    data_4 = torchvision.datasets.ImageFolder(root="../../Data/CommodityImage/dslr",
                                              transform=trans)
    data_5 = torchvision.datasets.ImageFolder(root="../../Data/CommodityImage/webcam",
                                              transform=trans)

    return data_1, data_2, data_3, data_4, data_5

def data_split(dataset, dataset_no=1):
    # Data Split
    if dataset_no == 1 or 'MNIST':
        dataset1 = []
        dataset2 = []
        for i in dataset:
            if i[1] < 5:
                dataset1.append(i)
            else:
                dataset2.append(i)
    else:
        print('Please input right dataset number.')
        return None, None
    return dataset1, dataset2




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

