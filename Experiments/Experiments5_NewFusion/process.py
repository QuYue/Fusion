# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/29 11:03:46
@Author      :Qu Yue
@File        :process.py
@Software    :Visual Studio Code
Introduction: Train and test process
'''
#%% Import Packages
import torch

#%% 
def training_process(Task, loss_func,Parm):
    true_amount = 0; total_amount = 0
    for step, [x, y] in enumerate(Task.train_loader):
        if Parm.cuda:
            x = x.cuda()
            y = y.cuda()
        predict_y = Task.model(x)
        loss = loss_func(predict_y, y)
        Task.optimizer.zero_grad()
        loss.backward()
        Task.optimizer.step()
        true_amount += int(torch.sum(predict_y.argmax(1).data == y.data))
        total_amount += y.shape[0]
    train_accuracy = true_amount / total_amount
    Task.train_accuracy[Task.ID].append(train_accuracy)

def testing_process(Task, Parm):
    true_amount = 0; total_amount = 0
    for step, [x, y] in enumerate(Task.test_loader):
        if Parm.cuda:
            x = x.cuda()
            y = y.cuda()
        predict_y = Task.model(x)
        true_amount += int(torch.sum(predict_y.argmax(1).data == y.data))
        total_amount += y.shape[0]
    test_accuracy = true_amount / total_amount
    Task.test_accuracy[Task.ID].append(test_accuracy)