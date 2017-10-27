import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import numpy as np
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def parse_json(fname):

    json = self.json[fname]
    age, cigarettes, gender = json['age'], json['cigarettes_per_day'], json['gender']

    return [age, cigarettes, gender]

def make_dataset(dir, dset_type, class_to_idx):
    datapoints = []

    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in os.walk(d):
            for fname in fnames:
                #Parse the filename
                dataset_type, raw_file, x, y = fname.strip('.jpeg').split('_')

                if fname.endswith(".jpeg") and dataset_type == dset_type:
                    path = os.path.join(root, fname)
                    item = (path, parse_json(raw_file + '.svs').extend([int(x), int(y)]), class_to_idx[target])
                    datapoints.append(item)
                    
    return datapoints

class TissueData(data.Dataset):
    def __init__(self, root, dset_type, transform=None):

        classes, class_to_idx = find_classes(root)

        self.json = pickle.load('/scratch/jmw784/capstone/Charrrrtreuse/JsonParser/LungJsonData.p')
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.datapoints = make_dataset(root, dset_type, class_to_idx)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, json information, x pos of original, y pos of original) for the given index
        """

        filepath, info, label = self.datapoints[index]

        img = pil_loader(filepath)

        if self.transform is not None:
            img = self.transform(img)

        info = np.array(info)
        info_length = len(info)
        height, width = img.size[1], img.size[2]
        
        output = np.repeat(info, height*width).reshape((len(info), height, width))

        return img, output, x, y, label

    def __len__(self):
        return len(self.datapoints)
