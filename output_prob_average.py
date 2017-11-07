
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from new_transforms import *
import pickle

"""
Assuming model is defined as something already, this is a fake model for now
"""

import torch.nn.functional as F

def model(img):
    #ignore img, simulate data
    return F.softmax(torch.randn(3))

# Take tile path and return probability for that tile

def get_tile_probability(tile_path):
    with open(tile_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = transform(img)
    var_img = Variable(img)

    return model(var_img).data

"""

In the main code, we will have something like this:

file_list = data['train'].filenames
tile_dict = pickle.load('PATH_TO_EDUARDOS_DICT_HERE')

"""

def aggregate(file_list, tile_dict, method):

    predictions = []
    true_labels = []

    for file in file_list:
        tile_paths, label = tile_dict[file]

        # Maybe there's a better way to do this than np.vectorize, you check
        prob_v = np.vectorize(get_tile_probability)
        probabilities = prob_v(tile_paths)

        if method == 'average':
            prediction = 0 # fill in the code here on how to predict
        elif method == 'max':
            prediction = 0 # fill in the code here on how to predict
        else:
            raise ValueError('Method not valid')

        predictions.append(prediction)
        true_labels.append(label)

    return predictions, true_labels

"""
Move these functions below into the method above
"""

def avg_probs(file_name): 
    '''Calculate average probabilities for a given tile
        input: tile name (file name)
        output: mean per class , index of max
    '''
    means = dict_probs[file_name].mean(axis = 0)
    
    return means, np.argmax(means) 


def max_probs(file_name):
    '''Calculate output class by majority of tiles
        input: tile name (file name)
        output: % votes per class , index of max
    '''
    votes = dict_probs[file_name].argmax(axis=1)
    out = np.array([ sum(votes == 0) , sum(votes == 1) , sum(votes == 2)])
    out = np.true_divide(out,sum(out))
    
    return out , np.argmax(out)

