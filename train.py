import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import numpy as np
import new_transforms
from dataloader import *
from model import *
from comet_ml import Experiment

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgSize', type=int, default=299, help='the height / width of the image to network')
parser.add_argument('--nc', type=int, default=8, help='input image channels + concatenated info channels')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--augment', action='store_true', help='Whether to use data augmentation or not')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nc = int(opt.nc)
imgSize = int(opt.imgSize)

experiment = Experiment(api_key="INSERT_API_KEY_HERE", log_code=True)
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
augment = transforms.Compose([new_transforms.RandomResizedCrop(imgSize),
                                new_transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                new_transforms.ColorJitter(0.5, 0.05, 2, 0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = {}
loaders = {}

for dset_type in ['train', 'valid', 'test']:

	if dset_type == 'train' and opt.augment:
		data[dset_type] = TissueData(root_dir, dset_type, transform = augment)
	else:
    	data[dset_type] = TissueData(root_dir, dset_type, transform = transform)

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
            m.bias.data = init.xavier_normal(m.bias.data)
        elif opt.init == 'kaiming':
            m.weight.data = init.kaiming_normal(m.weight.data)
            m.bias.data = init.kaiming_normal(m.bias.data)
        else:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:
        if opt.init == 'xavier':
            m.weight.data = init.xavier_normal(m.weight.data)
            m.bias.data = init.xavier_normal(m.bias.data)
        elif opt.init == 'kaiming':
            m.weight.data = init.kaiming_normal(m.weight.data)
            m.bias.data = init.kaiming_normal(m.bias.data)
        else:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

# Create model objects
model = cancer_CNN(nc, imgSize, gpu)
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
    optimizer = optim.RMSprop(netD.parameters(), lr = opt.lr)

# Define evaluation function for an entire dataset (train, valid, or test)
def evaluate(dset_type):

	model.eval()
	loss = 0

	for img, label in loaders[dset_type]:

		if opt.cuda:
			img = img.cuda()
			label = label.cuda()

		eval_input = Variable(img, volatile=True)
		eval_label = Variable(label, volatile=True)

		loss += crossEntropy(model(eval_input), eval_label)

	model.train()

	return loss / len(data[dset_type])

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
			val_loss = evaluate('valid')
			experiment.log_metric("Validation loss", val_loss.data[0])

			print('[%d/%d][%d/%d] Validation Loss: %f'
		            % (epoch, opt.niter, i, len(loaders['valid']), val_loss.data[0]))

	if epoch % 10 == 0:
        torch.save(model.state_dict(), '{0}/epoch_{1}.pth'.format(opt.experiment, epoch))

train_loss = evaluate('train')
val_loss = evaluate('valid')
test_loss = evaluate('test')

experiment.log_metric("Test loss", test_loss.data[0])

print('Finished training, train loss: %f, valid loss: %f, test loss: %f'
		% (train_loss.data[0], val_loss.data[0], test_loss.data[0]))