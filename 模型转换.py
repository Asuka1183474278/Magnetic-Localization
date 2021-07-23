import torch
from torch import nn
import numpy as np

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


device = torch.device('cuda:0')
net = torch.load('random_model.pkl')
x = np.loadtxt('data.txt') /(2**15)
x_train = torch.tensor(x[0:2],dtype=torch.float32).to(device)

print(net(x_train))
net.eval()



example = torch.rand(1,96).to(device)
traced_script_module = torch.jit.trace(net, example)
traced_script_module.save("net.pt")


output = traced_script_module(x_train)
print(output)
