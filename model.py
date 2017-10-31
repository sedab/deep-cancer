import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.dont_write_bytecode = True

class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(nc, 64, 4, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 32, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 16, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(16, 8, 4, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(8)

        # Three classes
        self.linear = nn.Linear(21904, 3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.softmax(self.linear(x))
        return x
