# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:22:32 2020

@author: QuYue
"""

#%%
import record
import numpy as np
import matplotlib.pyplot as plt
#%%


    
file_name = [
             './result/e3_4-2020-09-02_17-34-45.pkl',
             './result/e3_4-2020-09-02_19-44-58.pkl',
             './result/e3_4-2020-09-02_21-48-35.pkl']

file_name = ['./result/e3_4-2020-09-06_15-12-04.pkl',
             './result/e3_4-2020-09-07_01-39-56.pkl',
             './result/e3_4-2020-09-07_08-04-43.pkl',
             './result/e3_4-2020-09-07_14-52-39.pkl',
             './result/e3_4-2020-09-07_21-48-14.pkl']
file_name = ['./result/e3_4X-2020-09-08_15-59-04.pkl',
             './result/e3_4X-2020-09-08_17-22-05.pkl',
             './result/e3_4X-2020-09-08_19-30-13.pkl',
             './result/e3_4X-2020-09-08_22-11-33.pkl',
             './result/e3_4X-2020-09-08_23-48-33.pkl']

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
pinv_AF = np.empty([num])
pinv_AFAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_AF[i] = parm.result['FusionNet']['PinvFusion+AF']['TotalAcc']
    for j in range(length):
        pinv_AFAcc[i,j] = parm.result['FusionNet']['PinvFusion+AF']['Acc'][j]

print(f"pinvAFAcc: {mean_std(pinv_AFAcc, 0)}")
print(f"pinvAFTotal: {mean_std(pinv_AF)}")

# %%
pinv_MAN = np.empty([num])
pinv_MANAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_MAN[i] = parm.result['FusionNet']['PinvFusion+MAN']['TotalAcc']
    for j in range(length):
        pinv_MANAcc[i,j] = parm.result['FusionNet']['PinvFusion+MAN']['Acc'][j]

print(f"pinvMANAcc: {mean_std(pinv_MANAcc, 0)}")
print(f"pinvMANTotal: {mean_std(pinv_MAN)}")


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
FusionFineTune = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionFineTune[i] = max(parm.result['FusionNet']['FusionFineTune+AF'])

print(f"FusionFineTune+AF: {mean_std(FusionFineTune)}")

# %%
FusionKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionKD[i] = max(parm.result['FusionNet']['FusionKD+AF'])

print(f"FusionKD+AF: {mean_std(FusionKD)}")

#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD+AF'])

print(f"FusionMLKD+AF: {mean_std(FusionMLKD)}")

#%%
FusionFineTune = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionFineTune[i] = max(parm.result['FusionNet']['FusionFineTune+MAN'])

print(f"FusionFineTune+MAN: {mean_std(FusionFineTune)}")

# %%
FusionKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionKD[i] = max(parm.result['FusionNet']['FusionKD+MAN'])

print(f"FusionKD+MAN: {mean_std(FusionKD)}")

#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD+MAN'])

print(f"FusionMLKD+MAN: {mean_std(FusionMLKD)}")
#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i] 
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD2'])
    
print(f"FusionMLKD: {mean_std(FusionMLKD)}")
#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD2+AF'])
    print(FusionMLKD[i])
print(f"FusionMLKD+AF: {mean_std(FusionMLKD)}")
#%%
FusionMLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FusionMLKD[i] = max(parm.result['FusionNet']['FusionMLKD2+MAN'])

print(f"FusionMLKD+MAN: {mean_std(FusionMLKD)}")

