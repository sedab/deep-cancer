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

data = {}
loaders = {}

for dset_type in ['train', 'valid', 'test']:
    data[dset_type] = TissueData(root_dir, dset_type, transform = transform)
    loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=32, shuffle=True)
    print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

print('Class encoding:')
print(data['train'].class_to_idx)
