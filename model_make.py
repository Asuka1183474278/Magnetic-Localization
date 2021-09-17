import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt

batch_size = 256
learning_rate = 0.01
epochs = 20000
device = torch.device('cuda:0')

x = np.loadtxt('data_右.txt') # a list of numpy arrays
y = np.loadtxt('position_右.txt')-600  # another list of numpy arrays (targets)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=None)


x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)

train_dataset = data.TensorDataset(x_train,y_train)
test_dataset = data.TensorDataset(x_test,y_test)

train_dataloader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
test_dataloader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 200),
            nn.Tanh(),
            nn.Linear(200, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 96),
        )

    def forward(self, x):
        x = self.model(x)

        return x

net = MLP().to(device)
#net = torch.load('random_model.pkl').to(device)
optimizer = optim.ASGD(net.parameters(), lr=learning_rate)
criteon = nn.L1Loss().to(device)

for epoch in range(epochs):
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)

        predit = net(target)
        loss = criteon(data, predit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}:  Loss:{:6f}'.format(
        epoch, loss.item()
    ))
    test_loss = 0
    for (test_data, test_target) in test_dataloader:
        test_data, test_target = test_data.to(device), test_target.to(device)
        predit = net(test_target)
        test_loss += criteon(predit, test_data).item()
    print('Test set : Averge loss: {:.4f}\n'.format(
            test_loss / (int(len(test_dataloader.dataset) / batch_size) + 1)
        ))
torch.save(net, 'random_model.pkl')
"""
train_predit = net(x_train.to(device)).cpu().detach().numpy()
y_train = y_train.numpy()
fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.scatter(train_predit[:,0],train_predit[:,1],train_predit[:,2],'r')
fig2 = plt.figure(2)
ax2 = fig2.gca(projection='3d')
ax2.scatter(y_train[:,0],y_train[:,1],y_train[:,2])


test_predit = net(x_test.to(device)).cpu().detach().numpy()
y_test = y_test.numpy()
fig3 = plt.figure(3)
ax3 = fig3.gca(projection='3d')
ax3.scatter(test_predit[:,0],test_predit[:,1],test_predit[:,2],'r')
fig4 = plt.figure(4)
ax4 = fig4.gca(projection='3d')
ax4.scatter(y_test[:,0],y_test[:,1],y_test[:,2])

plt.show()
"""