#%%
# %matplotlib qt5
import numpy as np
import matplotlib.pyplot as plt
#%%
def task1_data():
    x = np.random.randn(100,2)
    x[:,0]+=3
    theta = np.array([-1, 1])
    y = np.sum(x*theta, axis=1)
    #y = np.sum(x**2*theta, axis=1)
    return x, y

def task_loss(y0, y):
    loss = np.linalg.norm(y0-y, 2)**2
    loss /= len(y0)
    return loss

def task2_data():
    x = np.random.randn(100,2)
    theta = np.array([4, 4])
    x[:,1]+=2
    y = np.sum(x*theta, axis=1)
    #y = np.sum(x**2*theta, axis=1)
    return x, y

class Model():
    def __init__(self, theta):
        self.theta = np.array(theta)
    def forward(self, x):
        y = np.sum(x*self.theta, axis=1)
        return y

D = np.ones([150,300])

x1, y1 = task1_data()
x2, y2 = task2_data()

for i in range(0,150):
    theta0 = i/10
    theta0 -= 5
    for j in range(0, 150):
        theta1 = j/10
        theta1 -= 5
        model = Model([theta0, theta1])
        y = model.forward(x1)
        loss1 = task_loss(y1, y)
        y = model.forward(x2)
        loss2 = task_loss(y2, y)
        D[i,j]=loss1
        D[i,j+150]=loss2
        
X,Y = np.meshgrid((np.arange(0,150)/10)-5, (np.arange(0,150)/10)-5)

#%%
x = np.vstack([x1,x2])
z = x.transpose(1,0).dot(x)

h = x1.transpose(1,0).dot(x1).dot(np.array([[-1.],[3]]))+\
    x2.transpose(1,0).dot(x2).dot(np.array([[4.],[4]]))
w = np.linalg.inv(z).dot(h)

# %%
def change(x,y):
    x=(x+5)*10
    y=(y+5)*10
    return x,y

#%%
w=[]
t = [[0,1],[0.05,0.95],[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1],[0.95,0.05],[0.99,0.01],[1,0]]
for i in t: 
    z1 = x1.transpose(1,0).dot(x1)*i[0]
    z2 = x2.transpose(1,0).dot(x2)*i[1]

    h = z1.dot(np.array([[-1.],[1]]))+\
        z2.dot(np.array([[4.],[4]]))
    w.append(np.linalg.inv(z1+z2).dot(h))

z1 = x1.transpose(1,0).dot(x1)
z2 = x2.transpose(1,0).dot(x2)
h = z1.dot(np.array([[-1.],[1]]))+\
    z2.dot(np.array([[4.],[4]]))
fw = (np.linalg.inv(z1+z2).dot(h))
#%%
#画等高线
plt.contour(X,Y,D[:,:150],30, colors='LightPink')
plt.contour(X,Y,D[:,150:],25, colors='LightSteelBlue')

plt.plot(fw[1],fw[0],'go')
x,y =[],[]
for ww in w:
    x.append(ww[1])
    y.append(ww[0])
plt.plot(x,y,'g')

plt.plot(1,-1,'ro')
plt.plot(4,4, 'bo')

plt.show()