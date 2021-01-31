# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:41:26 2020

@author: QuYue
"""


#%%
import record
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
file_name = ['./result/e4_2-2020-09-04_16-33-41.pkl']
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

# %%
FineTune = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FineTune[i] = max(parm.result['FusionNet']['FusionFineTune'])
    
print(f"FineTune: {mean_std(FineTune)}")

#%%
MLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    MLKD[i] = max(parm.result['FusionNet']['FusionMLKD'])

print(f"MLKD: {mean_std(MLKD)}")

#%%
plt.figure(1)
plt.plot(Parm[0].time['Origin'], Parm[0].result['Origin']['origin'])
plt.plot(Parm[0].time['FusionFineTune'], Parm[0].result['FusionNet']['FusionFineTune'])
plt.plot(Parm[0].time['FusionMLKD'], Parm[0].result['FusionNet']['FusionMLKD'])
plt.legend(['Normal', 'FineTune', 'MLKD'])
plt.grid('on')
plt.show()

#%%
def counter(acc):
    if int(acc//0.1)==9:
        return 9 + int((acc-0.9)//0.01)
    else:
        return int(acc//0.1)
        
def speed(acc_list):
    acc = [counter(i) for i in acc_list]
    acc = np.array(acc)
    return acc

d = speed(Parm[0].result['Origin']['origin'])
d = pd.Series(d).value_counts()
d1 = dict(d)
print(d)

d = speed(Parm[0].result['FusionNet']['FusionFineTune'])
d = pd.Series(d).value_counts()
d2 = dict(d)
print(d)

d = speed(Parm[0].result['FusionNet']['FusionMLKD'])
d = pd.Series(d).value_counts()
d3 = dict(d)
print(d)

#%%
h = {}
for i in range(1,19):
    a = []
    if i in d1:
        a.append(d1[i])
    else:
        a.append(0)
    if i in d2:
        a.append(d2[i])
    else:
        a.append(0)
    if i in d3:
        a.append(d3[i])
    else:
        a.append(0)
    h[i] = a


#%%
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))
name_list = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-91', '91-92','92-93','93-94','94-95','95-96','96-97','97-98','98-99','99-100']
num_list = [h[i][0] for i in range(1, 19)]
num_list1 = [h[i][1] for i in range(1, 19)]
num_list2 = [h[i][2] for i in range(1, 19)]
plt.figure(2)
plt.grid('on')
x = list(range(len(name_list)))
total_width, n = 0.8, 3
width = total_width / n

plt.bar(x, num_list, width=width, label='Normal', fc='b')

for i in range(len(x)):
    x[i] += 2*width
plt.bar(x, num_list2, width=width, label='MLKD', tick_label=name_list, fc='r')

for i in range(len(x)):
    x[i] -= width
plt.bar(x, num_list2, width=width, label='FineTune', tick_label=name_list, fc='g')
plt.legend()
plt.xlabel('Accuray')
plt.ylabel()
plt.show()



















