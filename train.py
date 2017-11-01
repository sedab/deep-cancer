import argparse
import random
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
from dataloader import *
from comet_ml import Experiment
import new_transforms

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels (+ concatenated info channels if metadata = True)')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--augment', action='store_true', help='Whether to use data augmentation or not')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--metadata', action='store_true', help='Whether to use metadata (default is not)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
parser.add_argument('--evalSize', type=int, default=2000, help='Number of samples to obtain validation loss on')
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity to use (selu, prelu, leaky, relu)')
opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nc = int(opt.nc)
imgSize = int(opt.imgSize)

experiment = Experiment(api_key="qcf4MjyyOhZj7Xw7UuPvZluts", log_code=True)
hyper_params = vars(opt)
experiment.log_multiple_params(hyper_params)

"""
Save experiment 
"""

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))
os.system('mkdir {0}/images'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

###############################################################################

"""
Load data
"""

root_dir = "/beegfs/jmw784/Capstone/LungTilesSorted/"

# Random data augmentation
augment = transforms.Compose([new_transforms.Resize((imgSize, imgSize)),
                              new_transforms.RandomVerticalFlip(),
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

for dset_type in ['train', 'valid', 'test']:
    if dset_type == 'train' and opt.augment:
        data[dset_type] = TissueData(root_dir, dset_type, transform = augment, opt.metadata)
    else:
        data[dset_type] = TissueData(root_dir, dset_type, transform = transform, opt.metadata)

    loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=opt.batchSize, shuffle=True)
    print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

print('Class encoding:')
print(data['train'].class_to_idx)

# Custom weights initialization
if opt.init not in ['normal', 'xavier', 'kaiming']:
    print('Initialization method not found, defaulting to normal')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if opt.init == 'xavier':
            m.weight.data = init.xavier_normal(m.weight.data)
        elif opt.init == 'kaiming':
            m.weight.data = init.kaiming_normal(m.weight.data)
        else:
            m.weight.data.normal_(-0.1, 0.1)
        
        m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:
        if opt.init == 'xavier':
            m.weight.data = init.xavier_normal(m.weight.data)
        elif opt.init == 'kaiming':
            m.weight.data = init.kaiming_normal(m.weight.data)
        else:
            m.weight.data.normal_(-0.1, 0.1)
        
        m.bias.data.fill_(0)

# Define model
class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, nonlinearity, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu

        if nonlinearity == 'selu':
            self.relu = nn.SELU()
        elif nonlinearity == 'prelu':
            self.relu = nn.PReLU()
        elif nonlinearity == 'leaky':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.init_dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(nc, 64, 4, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(256, 128, 4, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        # Three classes
        self.linear = nn.Linear(4096, 3)

    def forward(self, x):
        x = self.init_dropout(self.bn1(self.relu(F.max_pool2d(self.conv1(x),2))))
        x = self.dropout(self.bn2(self.relu(F.max_pool2d(self.conv2(x),2))))
        x = self.dropout(self.bn3(self.relu(F.max_pool2d(self.conv3(x),2))))
        x = x.view(x.size(0), -1)
        x = F.softmax(self.linear(x))
        return x

# Create model objects
model = cancer_CNN(nc, imgSize, opt.nonlinearity, ngpu)
model.apply(weights_init)
model.train()

crossEntropy = nn.CrossEntropyLoss()

# Load checkpoint models if needed
if opt.model != '': 
    model.load_state_dict(torch.load(opt.model))
print(model)

if opt.cuda:
    model.cuda()

# Set up optimizer
if opt.adam:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    optimizer = optim.RMSprop(model.parameters(), lr = opt.lr)

# Define evaluation function for an entire dataset (train, valid, or test)
def evaluate(dset_type, sample_size='full'):

    """
    Note: sample_size will be rounded up to be a multiple of the batch_size
    of the dataloader.
    """

    if sample_size == 'full':
        sample_size = len(data[dset_type])
    elif not isinstance(sample_size, int):
        raise ValueError("Amount should be 'full' or an integer")
    elif sample_size > len(data[dset_type]):
        raise ValueError("Amount cannot exceed size of dataset")    

    model.eval()
    loss = 0
    num_evaluated = 0

    for img, label in loaders[dset_type]:

        if opt.cuda:
            img = img.cuda()
            label = label.cuda()

        eval_input = Variable(img, volatile=True)
        eval_label = Variable(label, volatile=True)

        loss += crossEntropy(model(eval_input), eval_label)

        num_evaluated += img.size(0)

        if num_evaluated >= sample_size:
            model.train()
            return loss / num_evaluated

print('Starting training')

# Training loop
for epoch in range(opt.niter+1):
    data_iter = iter(loaders['train'])
    i = 0

    while i < len(loaders['train']):
        img, label = data_iter.next()
        i += 1

        # Drop the last batch if it's not the same size as the batchsize
        if img.size(0) != opt.batchSize:
            break

        if opt.cuda:
            img = img.cuda()
            label = label.cuda()

        model.zero_grad()

        input_img = Variable(img)
        target_label = Variable(label)

        train_loss = crossEntropy(model(input_img), target_label)
        train_loss.backward()

        optimizer.step()

        experiment.log_metric("Train loss", train_loss.data[0])

        print('[%d/%d][%d/%d] Training Loss: %f'
               % (epoch, opt.niter, i, len(loaders['train']), train_loss.data[0]))

        if i % 200 == 0: # Can change how often to evaluate val set

            eval_size = int(opt.evalSize)
            val_loss = evaluate('valid', sample_size=eval_size)
            experiment.log_metric("Validation loss (%s samples)" % (eval_size), val_loss.data[0])

            print('[%d/%d][%d/%d] Validation Loss: %f'
                   % (epoch, opt.niter, i, len(loaders['valid']), val_loss.data[0]))

    if epoch % 5 == 0:
        torch.save(model.state_dict(), '{0}/epoch_{1}.pth'.format(opt.experiment, epoch))

# Final evaluation
train_loss = evaluate('train')
val_loss = evaluate('valid')
test_loss = evaluate('test')

experiment.log_metric("Test loss", test_loss.data[0])

print('Finished training, train loss: %f, valid loss: %f, test loss: %f'
    % (train_loss.data[0], val_loss.data[0], test_loss.data[0]))
