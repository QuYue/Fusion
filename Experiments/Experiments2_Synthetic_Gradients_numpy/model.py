# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/27 20:36
@Author  : QuYue
@File    : model.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import torch
import torch.nn as nn
import numpy as np
import sys

#%%
def sigmoid(x):
    z = x.copy()
    z[x>=-100]=1 / (1 + np.exp(-x[x>=-100]))
    z[x<-100]=0
    return z

def sigmoid_out2deriv(out_data, in_data):
    return out_data * (1 - out_data)

def relu(x):
    z = x.copy()
    z[x<=0] = 0
    return z

def relu_out2deriv(out_data, in_data):
    z = np.ones(in_data.shape)
    z[in_data<=0] = 0
    return out_data*z

def MSEloss(output, label):
    loss = 0.5 * np.linalg.norm(output - label, 'fro')
    loss_deriv = output - label
    return loss, loss_deriv


#%% Network
class Layer(object):
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv):
        self.weights = (np.random.randn(input_dim, output_dim) * 0.01)
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv

    def forward(self, input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        return self.output

    def backward(self, output_delta):
        self.weight_output_delta = self.nonlin_deriv(output_delta, self.output)
        self.synthetic_gradient_delta = self.synthetic_gradient - output_delta
        return self.weight_output_delta.dot(self.weights.T)

    def update(self, alpha=0.001):
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        self.weight_synthetic_gradient = self.nonlin_deriv(self.synthetic_gradient, self.output)
        self.weights_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
        self.weights -= self.input.T.dot(self.weight_output_delta) * alpha


class DNI_Layer(object):
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv, alpha=0.001, alpha_sg=0.001):
        self.weights = (np.random.randn(input_dim, output_dim) * 0.01)
        self.weights_synthetic_grads = np.random.randn(output_dim, output_dim) * 0.01
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
        self.alpha_sg = alpha_sg

    def forward(self, input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        return self.output

    def backward(self, output_delta):
        self.output_delta = output_delta
        self.weight_output_delta = self.nonlin_deriv(output_delta, self.output)
        return self.weight_output_delta.dot(self.weights.T)

    def update0(self, times=1):
        self.weight_synthetic_gradient = self.nonlin_deriv(self.synthetic_gradient, self.output)
        self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        for i in range(times):
            self.synthetic_gradient_delta = self.synthetic_gradient - self.output_delta
            self.weights_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha_sg
            self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)

    def update(self, times=1):
        for i in range(times):
            self.synthetic_gradient_delta = self.synthetic_gradient - self.output_delta
            self.weights_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha_sg
            self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        self.weight_synthetic_gradient = self.nonlin_deriv(self.synthetic_gradient, self.output)
        self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha

    def fusion(self, layers ):
        def average(w):
            p = np.zeros(w[0].shape)
            for i in w:
                p += i
            p /= len(w)
            return p
        w, wsg = [], []
        for i in layers:
            w.append(i.weights)
            wsg.append(i.weights_synthetic_grads)

        self.weights_synthetic_grads_fusion = wsg
        self.weights = average(w)

    def forward_and_synthetic_update(self, input, num=0):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads_fusion[num])
        self.weight_synthetic_gradient = self.nonlin_deriv(self.synthetic_gradient, self.output)
        self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        return self.output


class FNN2():
    def __init__(self, nonlin=relu, nonlin_deriv=relu_out2deriv):
        self.layer = []
        self.layer.append(Layer(28 * 28, 1000 , nonlin, nonlin_deriv))
        self.layer.append(Layer(1000, 100, nonlin, nonlin_deriv))
        self.layer.append(Layer(100, 10, nonlin, nonlin_deriv))

    def length(self):
        return len(self.layer)

    def forward(self, x):
        for i in range(self.length()):
            x = self.layer[i].forward(x)
        return x

    def backward(self, output_delta):
        for i in range(self.length()-1, -1, -1):
            output_delta = self.layer[i].backward(output_delta)

    def update(self, alpha=0.01):
        for i in range(self.length()):
            self.layer[i].update(alpha)


class FNN2_DNI():
    def __init__(self, alpha=0.001, alpha_sg=0.001, nonlin=relu, nonlin_deriv=relu_out2deriv):
        self.train = True
        self.layer = []
        self.alpha = alpha
        self.alpha_sg = alpha_sg
        self.layer.append(DNI_Layer(28 * 28, 1000 , nonlin, nonlin_deriv, alpha, alpha_sg))
        self.layer.append(DNI_Layer(1000, 100, nonlin, nonlin_deriv, alpha, alpha_sg))
        self.layer.append(DNI_Layer(100, 10, nonlin, nonlin_deriv, alpha, alpha_sg))

    def length(self):
        return len(self.layer)

    def forward(self, x):
        for i in range(self.length()):
            x = self.layer[i].forward(x)
        return x

    def backward(self, output_delta):
        for i in range(self.length()-1, -1, -1):
            output_delta = self.layer[i].backward(output_delta)

    def update(self, epoch):
        if epoch<=10:
            for i in range(self.length()):
                self.layer[i].update()
        else:
            for i in range(self.length()):
                self.layer[i].update0()

    def fast_forward(self, x, num):
        for i in range(self.length()):
            x = self.layer[i].forward_and_synthetic_update(x, num)
        return x

#%%
def par_fusion(models, model0):
    net0 = list(model0.layer)
    nets = []
    for i in models:
        nets.append(list(i.layer))

    for i in range(len(net0)):
        net0[i].fusion([n[i] for n in nets])
    return model0




