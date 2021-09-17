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
epochs = 2000
device = torch.device('cuda:0')

x = np.loadtxt('data_右.txt') / (2**15)# a list of numpy arrays
y = np.loadtxt('position_右.txt')  # another list of numpy arrays (targets)
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

net = MLP().to(device)
#net = torch.load('random_model.pkl').to(device)
optimizer = optim.ASGD(net.parameters(), lr=learning_rate)
criteon = nn.L1Loss().to(device)

for epoch in range(epochs):
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)

        predit = net(data)
        loss = criteon(target, predit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}:  Loss:{:6f}'.format(
        epoch, loss.item()
    ))
    test_loss = 0
    for (test_data, test_target) in test_dataloader:
        test_data, test_target = test_data.to(device), test_target.to(device)
        predit = net(test_data)
        test_loss += criteon(predit, test_target).item()
    print('Test set : Averge loss: {:.4f}\n'.format(
            test_loss / (int(len(test_dataloader.dataset) / batch_size) + 1)
        ))
torch.save(net, 'random_model.pkl')