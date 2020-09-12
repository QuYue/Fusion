# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:53:20 2020

@author: QuYue
"""

#%%
import record
import numpy as np
import matplotlib.pyplot as plt
#%%

file_name = ['./result/results/result1_cnn1.pkl',
             './result/results/result2_cnn1.pkl',
             './result/results/result3_cnn1.pkl',
             './result/results/result4_cnn1.pkl',
             './result/results/result5_cnn1.pkl']
file_name = ['./result/results/task2_result1_fnn1.pkl',
             './result/results/task2_result2_fnn1.pkl',
             './result/results/task2_result3_fnn1.pkl',
             './result/results/task2_result4_fnn1.pkl',
             './result/results/task2_result5_fnn1.pkl']
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
for name in file_name:
    Parm.append(record.read(name))
    
#%%
for parm in Parm:
    print(parm.random_seed)
    
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
print('-------------------------------')
Aver = np.empty([num])
AverAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    Aver[i] = parm.result['FusionNet']['AverFusion']['TotalAcc']
    for j in range(length):
        AverAcc[i,j] = parm.result['FusionNet']['AverFusion']['Acc'][j]

print(f"AverAcc: {mean_std(AverAcc, 0)}")
print(f"AverTotal: {mean_std(Aver)}")


# %%
print('-------------------------------')
pinv = np.empty([num])
pinvAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv[i] = parm.result['FusionNet']['PinvFusion']['TotalAcc']
    for j in range(length):
        pinvAcc[i,j] = parm.result['FusionNet']['PinvFusion']['Acc'][j]

print(f"pinvAcc: {mean_std(pinvAcc, 0)}")
print(f"pinvTotal: {mean_std(pinv)}")

# %%
print('-------------------------------')
pinv_w = np.empty([num])
pinv_wAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_w[i] = parm.result['FusionNet']['PinvFusion_W']['TotalAcc']
    for j in range(length):
        pinv_wAcc[i,j] = parm.result['FusionNet']['PinvFusion_W']['Acc'][j]

print(f"pinvwAcc: {mean_std(pinv_wAcc, 0)}")
print(f"pinvwTotal: {mean_std(pinv_w)}")

# %%
print('-------------------------------')
pinv_AF = np.empty([num])
pinv_AFAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    k = []
    for nn in ['PinvFusion_W+AF','PinvFusion+AF']:
        k.append(parm.result['FusionNet'][nn]['TotalAcc'])
    pinv_AF[i] = max(k)
    for j in range(length):
        pinv_AFAcc[i,j] = parm.result['FusionNet']['PinvFusion+AF']['Acc'][j]

print(f"pinvAFTotal: {mean_std(pinv_AF)}")

#%%
print('-------------------------------')
pinv_AF = np.empty([num])
pinv_AFAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    k = []
    for nn in ['PinvFusion_W+AF+I']:
        k.append(parm.result['FusionNet'][nn]['TotalAcc'])
    pinv_AF[i] = max(k)
    for j in range(length):
        pinv_AFAcc[i,j] = parm.result['FusionNet']['PinvFusion+AF+I']['Acc'][j]

print(f"pinvAFTotal_I: {mean_std(pinv_AF)}")

# %%
print('-------------------------------')
pinv_MAN = np.empty([num])
pinv_MANAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    k = []
    for nn in ['PinvFusion_W+MAN', 'PinvFusion_L+MAN', 'PinvFusion+MAN']:
        k.append(parm.result['FusionNet'][nn]['TotalAcc'])
    pinv_MAN[i] = max(k)
    for j in range(length):
        pinv_MANAcc[i,j] = parm.result['FusionNet']['PinvFusion_L+MAN']['Acc'][j]

print(f"pinvMANTotal: {mean_std(pinv_MAN)}")

#%%
print('-------------------------------')
pinv_MAN = np.empty([num])
pinv_MANAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    k = []
    for nn in ['PinvFusion+MAN=I']:
        k.append(parm.result['FusionNet'][nn]['TotalAcc'])
    pinv_MAN[i] = max(k)
    for j in range(length):
        pinv_MANAcc[i,j] = parm.result['FusionNet']['PinvFusion+MAN=I']['Acc'][j]

print(f"pinvMANTotal_I: {mean_std(pinv_MAN)}")