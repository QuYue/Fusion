# -*- encoding: utf-8 -*-
'''
@Time        :2020/08/31 00:55:22
@Author      :Qu Yue
@File        :record.py
@Software    :Visual Studio Code
Introduction: 
'''

#%% 
import pickle
import time
import pandas as pd

#%%
def now():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def record(name, result, ntype='csv'):
    state = True
    name += f"-{now()}.{ntype}"
    if ntype in ['csv', 'txt']:
        result.to_csv(name)
    elif ntype in ['pkl']:
        with open(name, 'wb') as File:
            pickle.dump(result, File)
    else:
        state = False
    return state

def read(name):
    state = True
    ntype = name.split('.')[-1]
    if ntype in ['csv', 'txt']:
        result = pd.read_csv(name)
    elif ntype in ['pkl']:
        with open(name, 'rb') as File:
            result = pickle.load(File)
    else:
        state = False
    return result

#%%
if __name__ == '__main__':
    data = pd.DataFrame([[1,2,3],[4,5,6]])
    record('./result/example', data)
