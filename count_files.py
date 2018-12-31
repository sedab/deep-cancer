import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable

import os
import numpy as np
from PIL import Image
from utils.dataloader import *
from utils.auc import *
from utils import new_transforms
import argparse
import random

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='lung', help='Data to train on (lung/breast/kidney)')


parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels (+ concatenated info channels if metadata = True)')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--dropout', type=float, default=0.5, help='probability of dropout, default=0.5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default=None, help='where to store samples and models')
parser.add_argument('--augment', action='store_true', help='whether to use data augmentation or not')
parser.add_argument('--optimizer',type=str, default='Adam',  help='optimizer: Adam, SGD or RMSprop; default: Adam')
parser.add_argument('--metadata', action='store_true', help='whether to use metadata (default is not)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='nonlinearity to use (selu, prelu, leaky, relu)')
parser.add_argument('--earlystop', action='store_true', help='trigger early stopping (boolean)')
parser.add_argument('--method', type=str, default='average', help='aggregation prediction method (max, average)')
parser.add_argument('--decay_lr', action='store_true', help='activate decay learning rate function')
opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nc = int(opt.nc)
imgSize = int(opt.imgSize)



cudnn.benchmark = True

###############################################################################

"""
Load data
"""

if opt.data == 'breast':
    root_dir = "/beegfs/jmw784/Capstone/BreastTilesSorted/"
    num_classes = 2
    tile_dict_path = '/beegfs/jmw784/Capstone/Breast_FileMappingDict.p'
elif opt.data == 'kidney':
    root_dir = "/beegfs/jmw784/Capstone/KidneyTilesSorted/"
    num_classes = 4
    tile_dict_path = '/beegfs/jmw784/Capstone/Kidney_FileMappingDict.p'
elif opt.data == 'lung_ds1':
    root_dir = "/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1TilesSorted/"
    num_classes = 3
    tile_dict_path = '/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds1_FileMappingDict.p'
elif opt.data == 'lung_ds2':
    root_dir = "/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2TilesSorted/"
    num_classes = 3
    tile_dict_path = '/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds2_FileMappingDict.p'
elif opt.data == 'lung_ds3':
    root_dir = "/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3TilesSorted/"
    num_classes = 3
    tile_dict_path = '/beegfs/sb3923/DeepCancer/alldata/lung_ds/lung_ds3_FileMappingDict.p'
else:
    root_dir = "/beegfs/jmw784/Capstone/LungTilesSorted/"
    num_classes = 3
    tile_dict_path = '/beegfs/jmw784/Capstone/Lung_FileMappingDict.p'

# Random data augmentation
augment = transforms.Compose([new_transforms.Resize((imgSize, imgSize)),
                              transforms.RandomHorizontalFlip(),
                              new_transforms.RandomRotate(),
                              new_transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = {}
loaders = {}
data2 = {}
data3 = {}
data4 = {}
data5 = {}


data['test'] = TissueData(root_dir, 'test', transform = augment, metadata=opt.metadata)

#print('after tissuedata')
with open(tile_dict_path, 'rb') as f:
    tile_dict = pickle.load(f)

classes_test = data['test'].classes 

file_list = data['test'].filenames

#print('/beegfs/jmw784/Capstone/BreastTilesSorted/'+classes_test[0]+'/')

c1 = 0
c2 = 0
c3 = 0
c4 = 0

#c1 = 0
#c2 = 0
#c3 = 0
#c4 = 0

#data2['test'] = TissueData('/beegfs/jmw784/Capstone/BreastTilesSorted/'+classes_test[0]+'/', 'test', transform = augment, metadata=opt.metadata)
#data3['test'] = TissueData('/beegfs/jmw784/Capstone/BreastTilesSorted/'+classes_test[1]+'/', 'test',     transform = augment, metadata=opt.metadata)
#if num_classes>2:
#    data4['test'] = TissueData('/beegfs/jmw784/Capstone/BreastTilesSorted/'+classes_test[2]+'/', 'test',     transform = augment, metadata=opt.metadata)
#if num_classes>3:
#    data5['test'] = TissueData('/beegfs/jmw784/Capstone/BreastTilesSorted/'+classes_test[3]+'/', 'test',     transform = augment, metadata=opt.metadata)

for file in file_list:
    #print(file)
    tile_paths, label = tile_dict[file]
    if label == 0:
        c1 += 1
    elif label == 1:
        c2 += 1
    elif label == 2:
        c3 += 1     
    else:
        c4 += 1

#for tile in data['test']:
#    print(tile)

print('Finished loading %s dataset: %s samples' % ('test', len(data['test'])))
#print('Finished loading test' +classes_test[0]+ 'dataset: %s samples' % ( len(data2['test'])))
#print('Finished loading test'+classes_test[1]+ 'dataset: %s samples' % ( len(data3['test'])))
print('class %s has samples %i' %( classes_test[0], c1))
print('class %s has samples %i' %( classes_test[1], c2))
if num_classes>2:
#    print('Finished loading test'+classes_test[2]+ 'dataset: %s samples' % (len(data4['test'])))
    print('class %s has samples %i' %( classes_test[2], c3))
if num_classes>3:
#    print('Finished loading test'+classes_test[3]+ 'dataset: %s samples' % ( len(data5['test'])))
    print('class %s has samples %i' %( classes_test[3], c4))




data['train'] = TissueData(root_dir, 'train', transform = augment, metadata=opt.metadata)
 
with open(tile_dict_path, 'rb') as f:
    tile_dict = pickle.load(f)

classes_test2 = data['train'].classes
file_list2 = data['train'].filenames
 
 
c1 = 0
c2 = 0
c3 = 0
c4 = 0
 
for file in file_list2:
    tile_paths, label = tile_dict[file]
    if label == 0:
        c1 += 1
    elif label == 1:
        c2 += 1
    elif label == 2:
        c3 += 1
    else:
        c4 += 1
 
print('Finished loading %s dataset: %s samples' % ('train', len(data['train'])))
print('class %s has samples %i' %( classes_test2[0], c1))
print('class %s has samples %i' %( classes_test2[1], c2))
if num_classes>2:
    print('class %s has samples %i' %( classes_test2[2], c3))
if num_classes>3:
    print('class %s has samples %i' %( classes_test2[3], c4))


data['valid'] = TissueData(root_dir, 'valid', transform = augment, metadata=opt.metadata)
  
with open(tile_dict_path, 'rb') as f:
    tile_dict = pickle.load(f)

classes_test3 = data['valid'].classes
file_list3 = data['valid'].filenames
  
c1 = 0
c2 = 0
c3 = 0
c4 = 0
 
for file in file_list3:
    tile_paths, label = tile_dict[file]
    if label == 0:
        c1 += 1
    elif label == 1:
        c2 += 1
    elif label == 2:
        c3 += 1
    else:
        c4 += 1
 
print('Finished loading %s dataset: %s samples' % ('valid', len(data['valid'])))
print('class %s has samples %i' %( classes_test3[0], c1))
print('class %s has samples %i' %( classes_test3[1], c2))
if num_classes>2:
    print('class %s has samples %i' %( classes_test3[2], c3))
if num_classes>3:
    print('class %s has samples %i' %( classes_test3[3], c4))
