#%%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
#%%
transform = transforms.Compose(
    [transforms.ToTensor()]
)
data = torchvision.datasets.ImageFolder(root='../../Data/CommodityImage/', transform=transform)
#%%
a = data[0][0]
A = np.array(a)
#%%
plt.imshow(A.transpose([1,2,0]))

# %%
