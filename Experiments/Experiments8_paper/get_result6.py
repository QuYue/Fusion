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
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

#%% Read LLL Results
a = []

with open(r'./result/LLL_results.txt', 'r') as file:
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

x_major_locator=MultipleLocator(50)
plt.figure(1)
ax1 = plt.subplot(3,1,1)
plt.plot(range(1, 151), data['ori'][0], color='blue')
plt.plot(range(1, 151), data['ewc'][0], color='orange')
plt.plot(range(1, 151), data['mas'][0], color='green')
plt.plot(range(1, 51), data['Fusion'][0][0:50], color='red')
plt.plot([150, 180], [data['ori'][0][-1], data['ori'][0][-1]], color='blue', linestyle='--')
plt.plot([150, 180], [data['ewc'][0][-1], data['ewc'][0][-1]], color='orange', linestyle='--')
plt.plot([150, 180], [data['mas'][0][-1], data['mas'][0][-1]], color='green', linestyle='--')
plt.plot([151, 180], [data['Pinv'][0], data['Pinv'][0]], color='r', linestyle='--')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
plt.xlim(0, 180)
plt.ylim(0.5, 1)
ax1.xaxis.set_major_locator(x_major_locator)
plt.axvspan(0, 50, facecolor='green', alpha=0.2)
plt.axvspan(50, 100, facecolor='yellow', alpha=0.2)
plt.axvspan(100, 150, facecolor='blue', alpha=0.2)
plt.axvspan(150, 151, facecolor='red', alpha=0.5)
plt.ylabel('Accuracy on Task1', font1)
plt.grid('on')

ax2 = plt.subplot(3,1,2)
plt.plot(range(51, 151), data['ori'][1], color='blue')
plt.plot(range(51, 151), data['ewc'][1], color='orange')
plt.plot(range(51, 151), data['mas'][1], color='green')
plt.plot(range(51, 101), data['Fusion'][1][0:50], color='r')
plt.plot([150, 180], [data['ori'][1][-1], data['ori'][1][-1]], color='blue', linestyle='--')
plt.plot([150, 180], [data['ewc'][1][-1], data['ewc'][1][-1]], color='orange', linestyle='--')
plt.plot([150, 180], [data['mas'][1][-1], data['mas'][1][-1]], color='green', linestyle='--')
plt.plot([151, 180], [data['Pinv'][1], data['Pinv'][1]], color='r', linestyle='--')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.xlim(0, 180)
plt.ylim(0.5, 1)
ax2.xaxis.set_major_locator(x_major_locator)
plt.axvspan(0, 50, facecolor='green', alpha=0.2)
plt.axvspan(50, 100, facecolor='yellow', alpha=0.2)
plt.axvspan(100, 150, facecolor='blue', alpha=0.2)
plt.axvspan(150, 151, facecolor='red', alpha=0.5)
plt.ylabel('Accuracy on Task2', font1)
plt.grid('on')

ax3 = plt.subplot(3,1,3)
plt.plot(range(101, 151), data['ori'][2], color='blue')
plt.plot(range(101, 151), data['ewc'][2], color='orange')
plt.plot(range(101, 151), data['mas'][2], color='green')
plt.plot(range(101, 151), data['Fusion'][2][0:50], color='r')
plt.plot([150, 180], [data['ori'][2][-1], data['ori'][2][-1]], color='blue', linestyle='--')
plt.plot([150, 180], [data['ewc'][2][-1], data['ewc'][2][-1]], color='orange', linestyle='--')
plt.plot([150, 180], [data['mas'][2][-1], data['mas'][2][-1]], color='green', linestyle='--')
plt.plot([151, 180], [data['Pinv'][2], data['Pinv'][2]], color='r', linestyle='--')
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
plt.xlim(0, 180)
plt.ylim(0.5, 1)
ax3.xaxis.set_major_locator(x_major_locator)
plt.axvspan(0, 50, facecolor='green', alpha=0.2)
plt.axvspan(50, 100, facecolor='yellow', alpha=0.2)
plt.axvspan(100, 150, facecolor='blue', alpha=0.2)
plt.axvspan(150, 151, facecolor='red', alpha=0.5)
plt.xlabel('Epochs', font1)
plt.ylabel('Accuracy on Task3', font1)
plt.grid('on')
plt.legend(('Normal', 'EWC', 'MAS', 'Fusion'), prop=font1)

print('Finish')
plt.show()