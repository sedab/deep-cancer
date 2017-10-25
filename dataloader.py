import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def name_list(directory, dset_type):
    names = []
    
    for filename in os.listdir(directory): 

        #Parse the filename
        dataset_type, file, x, y = filename.strip('.jpeg').split('_')

        #Only add it if it's the correct dset_type (train, valid, test)
        if filename.endswith(".jpeg") and dataset_type == dset_type:
            names.append([filename, file + '.svs', x, y])

    return names

class TissueData(data.Dataset):
    def __init__(self, root, dset_type, transform=None):
        self.root = root
        self.raw = name_list(root, dset_type)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (lr_image, hr_image) for the given index
        """

        filename, file, x, y = self.names[index]

        img = pil_loader(self.root+filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, file, x, y

    def __len__(self):
        return len(self.raw)