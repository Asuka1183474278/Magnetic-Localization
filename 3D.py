import torch
import numpy as np
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


device = torch.device('cuda:0')
x = np.loadtxt('data.txt') / (2**15)# a list of numpy arrays
y = np.loadtxt('position.txt')  # another list of numpy arrays (targets)
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=120)


x_train = x[0:2]
x_test = x[262:284]
y_train = y[0:2]
y_test = y[262:284]

x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(96, 200),
            nn.Tanh(),
            nn.Linear(200, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 3),
        )

    def forward(self, x):
        x = self.model(x)

        return x
#%%
criteon = nn.L1Loss().to(device)

net = torch.load('random_model.pkl')

train_predit = net(x_train.to(device))
print(train_predit)
print(y_train)
train_loss = criteon(train_predit, y_train.to(device))
train_predit = net(x_train.to(device)).cpu().detach().numpy()
print(train_loss.item())
y_train = y_train.numpy()
fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.scatter(train_predit[:,0],train_predit[:,1],train_predit[:,2],'r')
fig2 = plt.figure(2)
ax2 = fig2.gca(projection='3d')
ax2.scatter(y_train[:,0],y_train[:,1],y_train[:,2])


test_predit = net(x_test.to(device))
print(test_predit)
test_loss = criteon(test_predit, y_test.to(device))
test_predit = net(x_test.to(device)).cpu().detach().numpy()
print(test_loss.item())
y_test = y_test.numpy()
fig3 = plt.figure(3)
ax3 = fig3.gca(projection='3d')
ax3.scatter(test_predit[:,0],test_predit[:,1],test_predit[:,2],'r')
fig4 = plt.figure(4)
ax4 = fig4.gca(projection='3d')
ax4.scatter(y_test[:,0],y_test[:,1],y_test[:,2])

plt.show()