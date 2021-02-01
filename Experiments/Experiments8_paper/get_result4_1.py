# -*- encoding: utf-8 -*-
'''
@Time        :2021/01/31 23:30:26
@Author      :Qu Yue
@File        :get_result4_1.py
@Software    :Visual Studio Code
Introduction: 
'''

#%% Import Packages
import record
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

#%% 
font1={'weight': 'bold'}
file_name = ['./result/e4_2-2020-09-04_16-33-41.pkl']
class PARM:
    def __init__(self):
        self.dataset_ID = 1
    @property
    def dataset_name(self):
        return self.data.data_dict[self.dataset_ID]
    @property
    def task_number(self):
        return self.data.tasks[self.dataset_name] 
        
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
    
# for parm in Parm:
    # print(parm.random_seed)
#%%
file_name = ['./result/e4_2-2020-09-05_11-02-00.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

plt.figure(6)
ax1 = plt.subplot(1,5,1)

plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
plt.ylabel('Accuracy', font1)
plt.title('Disjoint MNIST (2 tasks) with FNN', font1)
plt.xlim(-10,500)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
#plt.show()

file_name = ['./result/e4_2-2021-02-01_00-53-41.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax2 = plt.subplot(1,5,2)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy', font1)
plt.title('Split MNIST (5 tasks) with FNN', font1)
plt.xlim(-10,500)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')


file_name = ['./result/e4_2-2021-01-30_23-22-22.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax2 = plt.subplot(1,5,3)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy', font1)
plt.title('Disjoint MNIST (2 tasks) with CNN', font1)
plt.xlim(-10,700)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')

file_name = ['./result/e4_2-2021-01-30_22-38-44.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax2 = plt.subplot(1,5,4)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy', font1)
plt.title('Split MNIST (5 tasks) with CNN', font1)
plt.xlim(-10,700)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')


file_name = ['./result/e5_4-2021-01-31_18-06-39.pkl']
file_name = ['./result/e5_4-2021-02-01_00-02-38.pkl']
file_name = ['./result/e5_4-2020-09-05_17-34-16.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax3 = plt.subplot(1,5,5)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionFineTune'])
plt.plot(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionMLKD'])
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], loc='lower right', prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy')
plt.title('TRANSPORT4 (2 tasks) with CNN', font1)
plt.xlim(-10,700)
plt.ylim(0.1, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
plt.show()