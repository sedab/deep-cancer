from datetime import datetime
from sklearn.manifold import TSNE
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable
import argparse
import copy
import numpy as np
import time
import os
from PIL import Image
from utils.dataloader_all import *
#from utils.auc_test import *
from utils import new_transforms


root_dir = "/beegfs/sb3923/DeepCancer/alldata/AlldataTilesSorted/"
num_classes = 9

tile_dict_path = '/beegfs/sb3923/DeepCancer/alldata/Alldata_FileMappingDict.p'

imgSize = 299

transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#use Tissuedata2 for downsampled data
test_data = TissueData2(root_dir, 'test', transform = transform, metadata=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

classes = test_data.classes


def get_tile_probability(tile_path):

    """
    Returns an array of probabilities for each class given a tile

    @param tile_path: Filepath to the tile
    @return: A ndarray of class probabilities for that tile
    """

    # Some tiles are empty with no path, return nan
    if tile_path == '':
        return np.full(num_classes, np.nan)

    tile_path = root_dir + tile_path

    with open(tile_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(img).unsqueeze(0)

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    output = F.softmax(model(var_img)[0]).data.squeeze(0) 
    return output.numpy()


def get_tile_probability2(tile_path):
    """
    Returns an array of probabilities for each class given a tile

    @param tile_path: Filepath to the tile
    @return: A ndarray of class probabilities for that tile
    """
    # Some tiles are empty with no path, return nan
    if tile_path == '':
        return np.full(5184, np.nan)

    tile_path = root_dir + tile_path

    with open(tile_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(img).unsqueeze(0)

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    viz = (model(var_img)[1]).squeeze(0)#torch.FloatTensor of size 1x5184 
    return viz.data.numpy() #numpy.ndarray



with open(tile_dict_path, 'rb') as f:
    tile_dict = pickle.load(f)


def aggregate(file_list, method):

    """
    Given a list of files, return scores for each class according to the
    method and labels for those files.

    @param file_list: A list of file paths to do predictions on
    @param method: 'average' - returns the average probability score across
                               all tiles for that file
                   'max' - predicts each tile to be the class of the maximum
                           score, and returns the proportion of tiles for
                           each class

    @return: a ndarray of class probabilities for all files in the list
             a ndarray of the labels

    """

    model.eval()
    predictions = []
    true_labels = []
    last_layer = []

    for file in file_list:
        tile_paths, label = tile_dict[file]

        folder = classes[label]

        def add_folder(tile_path):
            if tile_path == '':
                return ''
            else:
                return folder + '/' + tile_path

        # Add the folder for the class name in front
        add_folder_v = np.vectorize(add_folder)
        tile_paths0 = add_folder_v(tile_paths)

        # Get the probability array for the file
        prob_v= np.vectorize(get_tile_probability, otypes=[np.ndarray])
        probabilities = prob_v(tile_paths0)
        
        tile_paths1 = add_folder_v(tile_paths)
        
        prob_v2 = np.vectorize(get_tile_probability2, otypes=[np.ndarray])
        lastlayer = prob_v2(tile_paths1)

        probabilities = np.stack(probabilities.flat)
        prediction = np.nanmean(probabilities, axis = 0)

        tile_label = np.argmax(probabilities,axis=1)
        #last layer
        lastlayer = np.stack(lastlayer.flat)
        a = lastlayer[np.ix_(label == tile_label),:]

        if (np.squeeze(a)).ndim>1:
            lastlayerweights = np.nanmean(np.squeeze(a), axis = 0)
            #lastlayer = np.stack(lastlayer.flat)
            #lastlayerweights = np.nanmean(lastlayer, axis = 0)

            predictions.append(prediction)
            true_labels.append(label)
            last_layer.append(lastlayerweights)

    return np.array(predictions), np.array(true_labels), np.array(last_layer)
    #return np.array(true_labels), np.array(last_layer)



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool, **kwargs):
        super(BasicConv2d, self).__init__()

        self.pool = pool
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv(x)

        if self.pool:
            x = F.max_pool2d(x, 2)
        
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# Define model
class cancer_CNN(nn.Module):
    def __init__(self, nc, imgSize, ngpu):
        super(cancer_CNN, self).__init__()
        self.nc = nc
        self.imgSize = imgSize
        self.ngpu = ngpu
        self.data = 'all'
        self.conv1 = BasicConv2d(nc, 16, False, kernel_size=5, padding=1, stride=2, bias=True)
        self.conv2 = BasicConv2d(16, 32, False, kernel_size=3, bias=True)
        self.conv3 = BasicConv2d(32, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv4 = BasicConv2d(64, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv5 = BasicConv2d(64, 128, True, kernel_size=3, padding=1, bias=True)
        self.conv6 = BasicConv2d(128, 64, True, kernel_size=3, padding=1, bias=True)
        self.linear = nn.Linear(5184, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        llw=x
        x = self.linear(x)
        return x, llw


model = cancer_CNN(3, imgSize, 1)

model_path = "/scratch/jmw784/capstone/deep-cancer/experiments/train_all_downsample_2/epoch_17.pth"
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)

predictions, labels, fw_lastlayer = aggregate(test_data.filenames, method='average')
print('------------------------------------------------------')
print('last-layer')
print(fw_lastlayer)

finalWs = fw_lastlayer
os.chdir("/scratch/jmw784/capstone/deep-cancer/tsne_figures/")
pickle.dump( finalWs, open( "finalWs_all_3.p", "wb" ) )
pickle.dump( predictions, open( "predictions_all_3.p", "wb" ) )
pickle.dump( labels, open( "labels_all_3.p", "wb" ) )

#A function provided by Google in one of their Tensorflow tutorials 
#for visualizing data with t-SNE by plotting it to a graph.

def plot_with_labels(lowDWeights, labels, filename='tsne.png'):
    assert lowDWeights.shape[0] >= len(labels), "More labels than weights"
    plt.figure(figsize=(20, 20))  #in inches
    for i, label in enumerate(labels):
        x, y = lowDWeights[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)
    

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
lowDWeights = tsne.fit_transform(fw_lastlayer)
labels = ['0','1','2','3','4','5','6','7','8']
plot_with_labels(lowDWeights, labels)


