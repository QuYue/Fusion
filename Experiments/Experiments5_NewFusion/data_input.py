# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/26 17:29:26
@Author      :Qu Yue
@File        :data_input.py
@Software    :Visual Studio Code
Introduction: Input the data.
'''
#%% Import Packages
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from numpy.random import permutation
from sklearn.model_selection import train_test_split
# %matplotlib auto
#%% Functions
def data_input(Parm):
    """Input the data.
    Input: 
        dataset_ID: Data set number [1, 2, 3]
        download: If download the data set [True, False]
    Output:
        datasets
    """
    # Parameter
    dataset_ID = Parm.dataset_ID
    download = Parm.data.MNIST.download
    # Input Data
    if dataset_ID == 1: # Disjoint MNIST
        print('Loading dataset 1: Disjoint MNIST...')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], Parm)
    elif dataset_ID == 2: # Split MNIST
        print('Loading dataset 2: Split MNIST...')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], Parm)
    elif dataset_ID == 3: # Permuted MNIST
        print('Loading dataset 3: Permuted MNIST')
        data, target = MNIST_input(download)
        datasets = task_split([data, target], Parm)
    elif dataset_ID == 4: # Commodity Image
        print('Loading dataset 4: Commodity Image')
        datasets = CommodityImage_input()
    else:
        print('Please input right dataset number.')
        return None
    print('Data is ready.')
    return datasets

def MNIST_input(download=True):
    """Input MNIST data.
    Input:
        download: If download the data set [True, False]
    Output:
        mnist_data
        mnist_target
    """
    def data_combine(data_train, data_test):
        data = []; target = []
        for d,t in data_train:
            data.append(d), target.append(t)
        for d,t in data_test:
            data.append(d), target.append(t)
        return data, target
    # Data MNIST
    # trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                         torchvision.transforms.Normalize((0.5,), (1.0,))])
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                            train=True,
                                            transform=trans,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST",
                                           train=False,
                                           transform=trans,
                                           download=download)
    # Data Combine
    mnist_data, mnist_target = data_combine(data_train, data_test)
    return mnist_data, mnist_target

def CommodityImage_input():
    """Input CommodityImage data.
    Input: None
    Output:
        datasets
    """
    transform = transforms.Compose([transforms.Scale([300, 300]),
                                    transforms.ToTensor()])
    path = '../../Data/commodityImage/'
    datasetname = ['amazon', 'caltech', 'dslr', 'webcam']
    datasets = []
    for name in datasetname:
        dataset = torchvision.datasets.ImageFolder(root=path+name, transform=transform)
        data = []; target = []
        for d,t in dataset:
            data.append(d), target.append(t)
        datasets.append({'data':data, 'target': target})
    return datasets
    

def task_split(dataset, Parm):
    dataset_ID = Parm.dataset_ID
    # Task Split
    datasets = []
    if dataset_ID == 1: # Disjoint MNIST
        datasets = [{'data':[], 'target':[]} for i in range(2)]
        for i in range(len(dataset[1])):
            if dataset[1][i] < 5:
                datasets[0]['data'].append(dataset[0][i])
                datasets[0]['target'].append(dataset[1][i])
            else: 
                datasets[1]['data'].append(dataset[0][i])
                datasets[1]['target'].append(dataset[1][i])
    elif dataset_ID == 2: # Split MNIST
        datasets = [{'data':[], 'target':[]} for i in range(5)]
        for i in range(len(dataset[1])):
            if dataset[1][i] <= 1:
                datasets[0]['data'].append(dataset[0][i])
                datasets[0]['target'].append(dataset[1][i])
            elif 1 < dataset[1][i] <= 3:
                datasets[1]['data'].append(dataset[0][i])
                datasets[1]['target'].append(dataset[1][i])
            elif 3 < dataset[1][i] <= 5:
                datasets[2]['data'].append(dataset[0][i])
                datasets[2]['target'].append(dataset[1][i])
            elif 5 < dataset[1][i] <= 7:
                datasets[3]['data'].append(dataset[0][i])
                datasets[3]['target'].append(dataset[1][i])
            elif 7 < dataset[1][i] <= 9:
                datasets[4]['data'].append(dataset[0][i])
                datasets[4]['target'].append(dataset[1][i])
    elif dataset_ID == 3: # Permuted MNIST
        tasks_ID = Parm.data.tasks['Permuted MNIST']
        permute_index = [list(range(len(dataset[0][0].view(-1))))]
        for i in range(tasks_ID-1):
            permute_index.append(permutation(permute_index[0]))
        datasets = [{'data':[], 'target':[], 'index': permute_index[i]} for i in range(tasks_ID)]
        for i in range(len(dataset[1])):
            d = dataset[0][i].view(-1)
            for j in range(tasks_ID):
                temp_d = d[datasets[j]['index']]
                temp_d = temp_d.view(dataset[0][0].shape)
                datasets[j]['data'].append(temp_d)
                datasets[j]['target'].append(dataset[1][i])
    else:
        print('Please input right dataset number.')
        return None
    return datasets

def data_split(data, target, test_size=0.2, random_state=0):
    """Data split to train and test.
    Input:
        data: data need to be split
        target: target need to be split
        test_size: ratio of test set to total data set (default=0.2)
        random_state: random seed (default=0)
    Output::
        splitted_data: [[X_train, y_train], [X_test, y_test]]
    """
    X_train,X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return [[X_train, y_train], [X_test, y_test]]


#%% Class
class DATA():
    def __init__(self):
        self.download = True
        self.path = None

class DATASET():
    def __init__(self):
        self.MNIST = DATA()
        self.MNIST.path='../../Data/MNIST'
        self.data_dict = {1:'Disjoint MNIST', 2:'Split MNIST', 3:'Permuted MNIST', 4:'Commodity Image'}
        self.tasks = {'Disjoint MNIST':2, 'Split MNIST':5, 'Permuted MNIST': 3, 'Commodity Image': 4}
if __name__ == "__main__":
    class PARM():
        def __init__(self):
            self.data = DATASET()
            self.dataset_ID = 4
        @property
        def dataset_name(self):
            return self.data.data_dict[self.dataset_ID]  
        @property
        def task_number(self):
            return self.data.tasks[self.dataset_name]            

    Parm = PARM()
    Parm.data.MNIST.download = False
    datasets = data_input(Parm)

