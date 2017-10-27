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

class TissueData(data.Dataset):
    def __init__(self, root, dset_type, transform=None):

        classes, class_to_idx = find_classes(root)

        with open('/scratch/jmw784/capstone/Charrrrtreuse/JsonParser/LungJsonData.p', 'rb') as f:
            self.json = pickle.load(f)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.datapoints = self.make_dataset(root, dset_type, class_to_idx)
        self.transform = transform

    def parse_json(self, fname):
    
        json = self.json[fname]
        age, cigarettes, gender = json['age_at_diagnosis'], json['cigarettes_per_day'], json['gender']

        return [age, cigarettes, gender]

    def make_dataset(self, dir, dset_type, class_to_idx):
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
                    item = (path, self.parse_json(raw_file + '.svs') + [int(x), int(y)], class_to_idx[target])
                    datapoints.append(item)
                    
        return datapoints

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
        height, width = img.size(1), img.size(2)
        
        reshaped = torch.FloatTensor(np.repeat(info, height*width).reshape((len(info), height, width)))

        print(reshaped.size)

        output = torch.cat((img, reshaped), 0)

        return output, label

    def __len__(self):
        return len(self.datapoints)
