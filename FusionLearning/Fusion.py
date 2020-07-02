# -*- encoding: utf-8 -*-
'''
@Time        :2020/07/01 20:07:56
@Author      :Qu Yue
@File        :Fusion.py
@Software    :Visual Studio Code
Introduction: Fusion
'''

# Average Fusion
def average_fusion(models, model_fusion):
    def average(nets):
        aver_net = nets[0].clone()
        for i in range(1, len(nets)):
            aver_net += nets[i].data
        aver_net /= len(nets)
        return aver_net

    layers0 = list(model_fusion.model.named_parameters())
    layers = [list(model.model.named_parameters()) for model in models]

    for i in range(len(layers0)):
        layers0[i][1].data = average([model[i][1].data for model in layers])
    return model_fusion