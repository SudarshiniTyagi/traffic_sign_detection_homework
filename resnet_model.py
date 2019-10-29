import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import sys

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18()

        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, nclasses)

    def forward(self, x):
        x = self.resnet(x)

        #64x1000
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        #64x100
        x = self.fc2(x)

        #64x43
        return F.log_softmax(x)