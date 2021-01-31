#%%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
#%%
transform = transforms.Compose(
    [transforms.Scale([300, 300]),
    transforms.ToTensor()]
)

data1 = torchvision.datasets.ImageFolder(root='../../Data/CommodityImage/amazon', transform=transform)
data2 = torchvision.datasets.ImageFolder(root='../../Data/CommodityImage/caltech', transform=transform)
data3 = torchvision.datasets.ImageFolder(root='../../Data/CommodityImage/dslr', transform=transform)
data4 = torchvision.datasets.ImageFolder(root='../../Data/CommodityImage/webcam', transform=transform)


label = {0: 'bag', 1: 'bike', 2: 'calculator', 3: 'headset', 4: 'keyboard', 5: 'laptop', 6: 'monitor', 7: 'mouse', 8: 'cup', 9: 'projector' }



#%%
a = data1[0]
A = np.array(a[0])
name = a[1]
plt.imshow(A.transpose([1,2,0]))
plt.title(f"{label[name]} {A.shape}")
# %%

Data1 = Data.DataLoader(data1, batch_size=10, shuffle=True)