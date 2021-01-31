# -*- encoding: utf-8 -*-
'''
@Time        :2020/09/03 16:32:47
@Author      :Qu Yue
@File        :get_result6.py
@Software    :Visual Studio Code
Introduction: 
'''
#%% Import Packages
import record
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Read LLL Results
a = []

with open(r'C:\Users\QuYue\Desktop\result.txt', 'r') as file:
    for line in file:
        a.append(line)

data = {}

for i in range(len(a)):
    if i%4 == 0:
        temp = []
        data[a[i][:-2]]=temp
    else:
        h =  a[i].strip('\n').strip(' ').split(' ')
        h = [float(i) for i in h]
        temp.append(h)
        
#%% Read Memory Fusion Results
file_name = ['./result/e6_3-2020-09-03_17-08-11.pkl']

class PARM:
    def __init__(self):
        self.dataset_ID = 1
        

num = len(file_name)
Parm = []

for name in file_name:
    Parm.append(record.read(name))

#%%  Solo Net
data['solonet'] = []
for pi in Parm[0].result['SoloNet']:
    data['solonet'].append(list(pi['History'].values())[0])
    
#%% Fusion and Finetune
data['Pinv'] = Parm[0].result['FusionNet']['PinvFusion_W']['Acc']

data['FineTune'] = Parm[0].result['FusionNet']['FusionFineTune']

#%% Memory Fusion
data['Fusion'] = []

for i in range(len(data['solonet'])):
    h = []
    h += data['solonet'][i]
    h.append(data['Pinv'][i])
    h += data['FineTune']
    data['Fusion'].append(h)
    
#%% Draw figure
font1={'weight': 'bold'}

plt.figure(1)
ax1 = plt.subplot(3,1,1)
plt.plot(data['Fusion'][0], color='r')
plt.plot(data['ori'][0], color='b')
plt.plot(data['ewc'][0], color='orange')
plt.plot(data['mas'][0], color='green')
plt.plot([50, 150], [data['Pinv'][0], data['Pinv'][0]], color='k', linestyle='--')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
plt.vlines(49, 0.5, 1, color ='gray', linestyle='--')
plt.ylim(0.5, 1)
#plt.xlabel('Epoch')
plt.ylabel('Accuracy on Task 1', font1)
plt.grid('on')

ax2 = plt.subplot(3,1,2)
plt.plot(list(range(50, 150)), data['ori'][1], color='b')
plt.plot(list(range(50, 150)), data['ewc'][1], color='orange')
plt.plot(list(range(50, 150)), data['mas'][1], color='green')
plt.plot(data['Fusion'][1], color='r')
plt.plot([50, 150], [data['Pinv'][1], data['Pinv'][1]], color='k', linestyle='--')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.vlines(49, 0.5, 1, color ='gray', linestyle='--')
plt.ylim(0.5, 1)
#plt.xlabel('Epoch')
plt.ylabel('Accuracy on Task 2', font1)
plt.grid('on')

ax3 = plt.subplot(3,1,3)
plt.plot(list(range(100, 150)), data['ori'][2], color='b')
plt.plot(list(range(100, 150)), data['ewc'][2], color='orange')
plt.plot(list(range(100, 150)), data['mas'][2], color='green')
plt.plot(data['Fusion'][2], color='r')
plt.plot([50, 150], [data['Pinv'][2], data['Pinv'][2]], color='k', linestyle='--')
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
plt.vlines(49, 0.5, 1, color ='gray', linestyle='--')
plt.ylim(0.5, 1)
plt.xlabel('Epochs', font1)
plt.ylabel('Accuracy on Task 3', font1)
plt.grid('on')
plt.legend(('Normal', 'EWC', 'MAS', 'Fusion', 'PinvFusion'), prop=font1)

print('Finish')