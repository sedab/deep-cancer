
# coding: utf-8


import numpy as np
import random
import torch


#dict_probs = {}


def probs_update( file_name , tensor ): # add self as a parameter: self.dict_probs
    
    ''' Update dictionary containing an array of the predicted probabilities per tile
        One element in the dictionary per slide
    '''
    
    if file_name in dict_probs:
        dict_probs[file_name] = np.append( dict_probs[file_name] ,[tensor.numpy()] , axis = 0)
    else:
        dict_probs[file_name] = np.array([tensor.numpy()])


def avg_probs(file_name): #add self
    '''Calculate average probabilities for a given tile
        input: tile name (file name)
        output: mean per class , index of max
    '''
    means = dict_probs[file_name].mean(axis = 0)
    
    return means , np.argmax(means) 


def max_probs(file_name):
    '''Calculate output class by majority of tiles
        input: tile name (file name)
        output: % votes per class , index of max
    '''
    votes = dict_probs[file_name].argmax(axis=1)
    out = np.array([ sum(votes == 0) , sum(votes == 1) , sum(votes == 2)])
    out = np.true_divide(out,sum(out))
    
    return out , np.argmax(out)

