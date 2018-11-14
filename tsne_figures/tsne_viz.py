from datetime import datetime
from sklearn.manifold import TSNE
import argparse
import numpy as np
import time
import os
from PIL import Image


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

z=np.load('final_weights.out.npy')

print(z)
lowDWeights = tsne.fit_transform(z)




labels = ['0','1','2','3','4','5','6','7','8']
plot_with_labels(lowDWeights, labels)

cd("/scratch/jmw784/capstone/deep-cancer/tsne_figures/")


