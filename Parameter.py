import torch
from torch import nn

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

net = torch.load('random_model.pkl').to(torch.device('cpu'))
for parameters in net.parameters():
    x = parameters.detach().numpy()
    print(parameters.shape)
