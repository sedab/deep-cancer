import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import numpy as np
from dataloader import *

root_dir = "/beegfs/jmw784/Capstone/LungTilesSorted/"

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = TissueData(root_dir, 'train', transform = transform)
print(len(dataset))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print(dataloader.dataset[0])
