# -*- encoding: utf-8 -*-
'''
@Time        :2020/06/28 14:45:31
@Author      :Qu Yue
@File        :drawing.py
@Software    :Visual Studio Code
Introduction: 
'''

#%% Import Packages
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def draw_result(result, fig, title=[], show=False, others=None):
    #  actionly draw the result
    # others = None or [min, max] or 'same'
    num = len(result)
    result = np.array(result)
    list_max = np.zeros(num)
    list_min = np.zeros(num)
    # check
    if len(title) < num:
        for i in range(len(title), num):
            title.append(str(i))
    xaxis = [list(range(len(i))) for i in result] # axis -x
    subplot = []
    fig.clf()
    for i in range(num):
        result[i] = np.array(result[i])
        list_max[i], list_min[i] = result[i].max(), result[i].min()
        subplot.append(fig.add_subplot(num, 1, i+1))
        subplot[i].plot(xaxis[i], result[i], marker='o')
        subplot[i].grid()
        subplot[i].set_title(title[i])
        if show:
            subplot[i].annotate(s=title[i] + ': %.3f' % result[i][-1], xy=(xaxis[i][-1], result[i][-1]),
                                xytext=(-20, 10), textcoords='offset points')
            subplot[i].annotate(s='Max: %.3f' % result[i].max(), xy=(result[i].argmax(), result[i].max()), xytext=(-20, -10),
                             textcoords='offset points')

    if type(others) == type([]):
        for i in range(num):
            subplot[i].set_ylim(others)
    elif others == 'same':
        min_num = [list_min.min(), list_max.max()]
        for i in range(num):
            subplot[i].set_ylim(min_num)
    plt.pause(0.01)

#%% Main Function
if __name__ == '__main__':
    fig = plt.figure(1)
    plt.ion()
    b = []
    c = []
    d = []
    for i in range(100):
        a = np.random.randn(3)
        b.append(a[0])
        c.append(a[1])
        if i<10:
            d.append(a[2])
        draw_result([b, c ,d], fig, ['b', 'c'], True, 'same')
    plt.ioff()
    plt.show()