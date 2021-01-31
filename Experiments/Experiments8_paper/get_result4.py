# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:41:26 2020

@author: QuYue
"""


#%% Import Packages
import record
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Font
font1={'weight': 'bold'}

#%% Read Epoch result
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
    
for parm in Parm:
    print(parm.random_seed)
    
#%% SoloNet Results
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

#%% Average Results
Aver = np.empty([num])
AverAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    Aver[i] = parm.result['FusionNet']['AverFusion']['TotalAcc']
    for j in range(length):
        AverAcc[i,j] = parm.result['FusionNet']['AverFusion']['Acc'][j]
print(f"AverAcc: {mean_std(AverAcc, 0)}")
print(f"AverTotal: {mean_std(Aver)}")


#%% PinvFusion Results
pinv = np.empty([num])
pinvAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv[i] = parm.result['FusionNet']['PinvFusion']['TotalAcc']
    for j in range(length):
        pinvAcc[i,j] = parm.result['FusionNet']['PinvFusion']['Acc'][j]
print(f"pinvAcc: {mean_std(pinvAcc, 0)}")
print(f"pinvTotal: {mean_std(pinv)}")

#%% PinvFusion_W Results
pinv_w = np.empty([num])
pinv_wAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_w[i] = parm.result['FusionNet']['PinvFusion_W']['TotalAcc']
    for j in range(length):
        pinv_wAcc[i,j] = parm.result['FusionNet']['PinvFusion_W']['Acc'][j]
print(f"pinvwAcc: {mean_std(pinv_wAcc, 0)}")
print(f"pinvwTotal: {mean_std(pinv_w)}")

#%% PinvFusion_W+AF Results
pinv_AF = np.empty([num])
pinv_AFAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_AF[i] = parm.result['FusionNet']['PinvFusion+AF']['TotalAcc']
    for j in range(length):
        pinv_AFAcc[i,j] = parm.result['FusionNet']['PinvFusion+AF']['Acc'][j]
print(f"pinvAFAcc: {mean_std(pinv_AFAcc, 0)}")
print(f"pinvAFTotal: {mean_std(pinv_AF)}")

#%% PinvFusion+MAN Result
pinv_MAN = np.empty([num])
pinv_MANAcc = np.empty([num, length])
for i in range(num):
    parm = Parm[i]
    pinv_MAN[i] = parm.result['FusionNet']['PinvFusion+MAN']['TotalAcc']
    for j in range(length):
        pinv_MANAcc[i,j] = parm.result['FusionNet']['PinvFusion+MAN']['Acc'][j]
print(f"pinvMANAcc: {mean_std(pinv_MANAcc, 0)}")
print(f"pinvMANTotal: {mean_std(pinv_MAN)}")

#%% Normal Training 
Origin = np.empty([num])
for i in range(num):
    parm = Parm[i]
    Origin[i] = max(parm.result['Origin']['origin'])
print(f"Origin: {mean_std(Origin)}")

#%% Base Fine-tuning
FineTune = np.empty([num])
for i in range(num):
    parm = Parm[i]
    FineTune[i] = max(parm.result['FusionNet']['FusionFineTune'])
print(f"FineTune: {mean_std(FineTune)}")

#%% MLKD
MLKD = np.empty([num])
for i in range(num):
    parm = Parm[i]
    MLKD[i] = max(parm.result['FusionNet']['FusionMLKD'])
print(f"MLKD: {mean_std(MLKD)}")

#%% Draw a demo Figure
plt.figure(1)
plt.plot(Parm[0].time['Origin'], Parm[0].result['Origin']['origin'])
plt.plot(Parm[0].time['FusionFineTune'], Parm[0].result['FusionNet']['FusionFineTune'])
plt.plot(Parm[0].time['FusionMLKD'], Parm[0].result['FusionNet']['FusionMLKD'])
plt.legend(['Normal', 'FineTune', 'MLKD'])
plt.grid('on')
plt.show()

#%% Get the Number of Epoch and time used to reach the Accuracy Interval
def counter(acc):
    if int(acc//0.1)==9:
        return 9 + int((acc-0.9)/0.01)
    else:
        return int(acc/0.1)
        
def speed(acc_list):
    acc = [counter(i) for i in acc_list]
    acc = np.array(acc)
    return acc

def times(acc_list, time_list):
    a = zip(list(range(1,19)), np.ones(19)*-1)
    t = dict(a)
    m =1
    for i in range(len(acc_list)):
        acc = acc_list[i]
        c = counter(acc)
        if m >= c:
            pass
        else:
            for j in range(m, c):
                t[j] = t[m]        
            t[c] = time_list[i]
            m=c
    if t[18] <= 0:
        t[18] = 1000
    return t
        
d = speed(Parm[0].result['Origin']['origin'])
d = pd.Series(d).value_counts()
t1 = times(Parm[0].result['Origin']['origin'], Parm[0].time['Origin'])
d1 = dict(d)
#print(d)
d = speed(Parm[0].result['FusionNet']['FusionFineTune'])
d = pd.Series(d).value_counts()
t2 = times(Parm[0].result['FusionNet']['FusionFineTune'], Parm[0].time['FusionFineTune'])
d2 = dict(d)
#print(d)
d = speed(Parm[0].result['FusionNet']['FusionMLKD'])
d = pd.Series(d).value_counts()
t3 = times(Parm[0].result['FusionNet']['FusionMLKD'], Parm[0].time['FusionMLKD'])
d3 = dict(d)
#print(d)

def accum_sum(d,i):
    z = 0
    for j in range(1,i+1):
        if j in d:
            z += d[j]
    return z

h = {}
for i in range(1,19):
    a = []
    a.append(accum_sum(d1, i))
    a.append(accum_sum(d2, i))
    a.append(accum_sum(d3, i))
    h[i] = a

#%% Draw Figure the Number of Epoch and time used to reach the Accuracy Interval
name_list = ['0-50','50-60', '60-70', '70-80', '80-90', '90-91', '91-92','92-93','93-94','94-95','95-96','96-97','97-98','98-99','99-100']
num_list = [h[i][0] for i in range(4, 19)]
num_list1 = [h[i][1] for i in range(4, 19)]
num_list2 = [h[i][2] for i in range(4, 19)]
tt1 = [t1[i] for i in range(4, 19)]
tt2 = [t2[i] for i in range(4, 19)]
tt3 = [t3[i] for i in range(4, 19)]

fig = plt.figure(4)
ax = fig.add_subplot(111)
x = list(range(len(name_list)))
total_width, n = 0.8, 3
width = total_width / n
ax.bar(x, num_list, width=width, label='Normal', fc='b', alpha=.8)
for i in range(len(x)):
    x[i] += width
ax.bar(x, num_list1, width=width, label='Fusion+FineTune', tick_label=name_list, fc='g', alpha=.8)

for i in range(len(x)):
    x[i] += width
ax.bar(x, num_list2, width=width, label='Fusion+MLKD', fc='r', alpha=.8)
ax.legend(loc=0)
ax2 = ax.twinx()
ax2.plot(np.array(list(range(0,14)))+width , tt1[:-1], '-b', marker='o', label = 'Normal')
ax2.plot(np.array(list(range(0,15)))+width, tt2, '-g', marker='*', label = 'Fusion+FineTune')
ax2.plot(np.array(list(range(0,15)))+width, tt3, '-r', marker='<', label = 'Fusion+MLKD')
ax.legend(loc=0)
ax2.legend(loc='upper left', bbox_to_anchor=(0,0.82))
#ax2.legend()
ax.grid()
ax.set_xlabel('Accuray Interval(%)')
ax.set_ylabel("Number of Epochs")
ax2.set_ylabel("Elapsed Time(s)")
ax2.set_ylim(0,650)
ax.set_ylim(0,200)
plt.xlim(-0.3,14.8)

plt.show()

#%% Draw Figure the Number of Epoch used to reach the Accuracy Interval
name_list = ['0-50','50-60', '60-70', '70-80', '80-90', '90-91', '91-92','92-93','93-94','94-95','95-96','96-97','97-98','98-99','99-100']
num_list = [h[i][0] for i in range(4, 19)]
num_list1 = [h[i][1] for i in range(4, 19)]
num_list2 = [h[i][2] for i in range(4, 19)]
tt1 = [t1[i] for i in range(4, 19)]
tt2 = [t2[i] for i in range(4, 19)]
tt3 = [t3[i] for i in range(4, 19)]

fig = plt.figure(5)
ax = plt.subplot(111)
plt.plot(num_list[:-1], list(range(14)), '-b', marker='o', label = 'Normal')
plt.plot(num_list1, list(range(15)), '-g', marker='*', label = 'Fusion+FineTune')
plt.plot(num_list2, list(range(15)), '-r', marker='<', label = 'Fusion+MLKD')
plt.legend(prop=font1)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
plt.grid('on')
plt.xlabel('Number of Epochs', font1)
plt.ylabel("Accuray Interval(%)", font1)
plt.title('Split MNIST', font1)
plt.yticks(list(range(15)),name_list)
plt.show()

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
ax1 = plt.subplot(1,3,1)

plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
plt.ylabel('Accuracy', font1)
plt.title('Disjoint MNIST (2 tasks)', font1)
plt.xlim(-10,300)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
#plt.show()

file_name = ['./result/e4_2-2020-09-05_10-07-19.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax2 = plt.subplot(1,3,2)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(time1, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionFineTune'])
plt.plot(time2, [Parm.result['FusionNet']['PinvFusion_W']['TotalAcc']]+Parm.result['FusionNet']['FusionMLKD'])
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy', font1)
plt.title('Split MNIST (5 tasks)', font1)
plt.xlim(-10,300)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')


file_name = ['./result/e5_4-2020-09-05_17-34-16.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)

time1 = list(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time2 = list(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number)
time1 = [Parm.time['SoloNet']/Parm.task_number] + time1
time2 = [Parm.time['SoloNet']/Parm.task_number] + time2

ax3 = plt.subplot(1,3,3)
plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionFineTune'])
plt.plot(np.array(Parm.time['FusionMLKD'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionMLKD'])
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune', 'Fusion+MLKD'], loc='lower right', prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy')
plt.title('TRANSPORT4 (2 tasks)', font1)
plt.xlim(-10,900)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
plt.show()


#%% Times number #
print('--------------------')
def find_times(result, time, edge):
    for i in range(len(result)):
        if result[i] > edge:
            break
    if i == (len(result)-1):
        print('Error')
    return time[i]

edge = 0.96
file_name = ['./result/e4_2-2020-09-05_11-02-00.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(f"Dataset: {Parm.dataset_name}| Model: {str(Parm.model).split('.')[1][:-2]}| edge: {edge}")
origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')

print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s {origin_time/fusionfinetune_time}x')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')
print(f'Total: {fusionfinetune_time+time1+time2}s {origin_time/(fusionfinetune_time+time1+time2)}x')
print('--------------------')

edge = 0.96
file_name = ['./result/e4_2-2020-09-05_10-07-19.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(f"Dataset: {Parm.dataset_name}| Model: {str(Parm.model).split('.')[1][:-2]}| edge: {edge}")

origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')

print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s {origin_time/fusionfinetune_time}x')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')
print(f'Total: {fusionfinetune_time+time1+time2}s {origin_time/(fusionfinetune_time+time1+time2)}x')
print('--------------------')

edge = 0.99
file_name = ['./result/e4_2-2021-01-30_23-22-22.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(f"Dataset: {Parm.dataset_name}| Model: {str(Parm.model).split('.')[1][:-2]}| edge: {edge}")

origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')
print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s {origin_time/fusionfinetune_time}x')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')
print(f'Total: {fusionfinetune_time+time1+time2}s {origin_time/(fusionfinetune_time+time1+time2)}x')
print('--------------------')

edge = 0.99
file_name = ['./result/e4_2-2021-01-30_22-38-44.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(f"Dataset: {Parm.dataset_name}| Model: {str(Parm.model).split('.')[1][:-2]}| edge: {edge}")

origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')
print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s {origin_time/fusionfinetune_time}x')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')
print(f'Total: {fusionfinetune_time+time1+time2}s {origin_time/(fusionfinetune_time+time1+time2)}x')
print('--------------------')



edge = 0.911
file_name = ['./result/e5_4-2021-01-31_18-06-39.pkl']
num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(f"Dataset: {Parm.dataset_name}| Model: {str(Parm.model).split('.')[1][:-2]}| edge: {edge}")

origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')

print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s {origin_time/fusionfinetune_time}x')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')
print(f'Total: {fusionfinetune_time+time1+time2}s {origin_time/(fusionfinetune_time+time1+time2)}x')
print('--------------------')



#%%
edge = 0.99
file_name = ['./result/e4_2-2021-01-30_23-22-22.pkl']
# file_name = ['./result/e4_2-2021-01-30_22-38-44.pkl']

num = len(file_name)
for name in file_name:
    Parm= record.read(name)
print(Parm.dataset_name, Parm.model)
origin = Parm.result['Origin']['origin']
origin_time = find_times(origin, Parm.time['Origin'], edge)
print(f'Origin time: {origin_time}s')
print('----------')
fusionfinetune=  Parm.result['FusionNet']['FusionFineTune']
fusionfinetune_time = find_times(fusionfinetune, Parm.time['FusionFineTune'],edge)
time1 = Parm.time['SoloNet']/Parm.task_number
time2 = Parm.time['PinvFusion_W']
print(f'FusionFineTune time: {fusionfinetune_time}s')
print(f'solo time : {time1}s')
print(f'PinvFusion_W: {time2}s')


plt.plot(Parm.time['Origin'], Parm.result['Origin']['origin'])
plt.plot(np.array(Parm.time['FusionFineTune'])+Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, Parm.result['FusionNet']['FusionFineTune'])
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
plt.legend(['Normal', 'Fusion+FineTune'], loc='lower right', prop=font1)
plt.xlabel('Time(s)', font1)
#plt.ylabel('Accuracy')
plt.xlim(-10,900)
plt.ylim(0.4, 1)
plt.vlines(Parm.time['PinvFusion_W']+Parm.time['SoloNet']/Parm.task_number, 0.1, 1, colors = "gray", linestyles = "dashed")
plt.grid('on')
# %%
