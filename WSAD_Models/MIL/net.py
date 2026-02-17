import torch
import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, input_dim=2048): # feature extractor for demo is [1, 2048] (RGB + Flow)
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.sigmoid(x)