# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:24:02 2020

@author: QuYue
"""

# -*- encoding: utf-8 -*-
'''
@Time        :2020/09/01 13:29:42
@Author      :Qu Yue
@File        :get_result.py
@Software    :Visual Studio Code
Introduction: 
'''

#%%
import record
import numpy as np
import matplotlib.pyplot as plt
#%%
file_name = ['./result/e9_1-2020-09-01_13-15-34.pkl',
             './result/e9_1-2020-09-01_14-06-50.pkl',
             './result/e9_1-2020-09-01_14-27-14.pkl',
             './result/e9_1-2020-09-01_14-50-11.pkl',
             './result/e9_1-2020-09-01_15-13-06.pkl']

#%%
class PARM:
    def __init__(self):
        self.dataset_ID = 1
        
def mean_std(vector, dim=None):
    vector = np.array(vector)
    if dim == None:
        m = vector.mean()
        n = vector.std()
    else:
        m = vector.mean(dim)
        n = vector.std(dim)
    return m, n
    


num = len(file_name)
Parm = []
#%%
for name in file_name:
    Parm.append(record.read(name))
    
#%%
#for parm in Parm:
#    print(parm.random_seed)
    
#%%
length = len(Parm[0].result['SoloNet'])
Acc = np.empty([length, num])
Total = np.empty([length, num])
for i in range(num):
    parm = Parm[i]
    for j in range(length):
        Acc[j, i] = parm.result['SoloNet'][j]['Acc']
        Total[j, i] = parm.result['SoloNet'][j]['TotalAcc']

print(f"SoloNetAcc: {mean_std(Acc, 1)}")
print(f"SoloNetTotal: {mean_std(Total, 1)}")

# %%
Origin = np.empty([num])
for i in range(num):
    parm = Parm[i]
    Origin[i] = max(parm.result['Origin']['origin'])

print(f"Origin: {mean_std(Origin)}")

#%%
FusionFineTune = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionFineTune[i] = max(parm.result['FusionNet']['FusionFineTune'])

print(f"FusionFineTune: {mean_std(FusionFineTune)}")

# %%
FusionKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionKD[i] = max(parm.result['FusionNet']['FusionKD'])

print(f"FusionKD: {mean_std(FusionKD)}")

#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD'])

print(f"FusionMLKD: {mean_std(FusionMLKD)}")

#%%
FusionMLKD2 = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD2[i] = max(parm.result['FusionNet']['FusionMLKD2'])

print(f"FusionMLKD2: {mean_std(FusionMLKD2)}")











