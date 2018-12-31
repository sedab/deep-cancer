import matplotlib
matplotlib.use('Agg')
from datetime import datetime
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
import numpy as np
import time
from PIL import Image
from utils.dataloader import *
#from utils.auc_all_class import * 
#fom utils.auc_test import *
from utils import new_transforms
from matplotlib import *
from pylab import *
from utils.dataloader import *
import matplotlib.pyplot as plt



imgSize = 299
transform = transforms.Compose([new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
    img = img.cuda()

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    output = F.softmax(model(var_img)).data.squeeze(0)

    return output.cpu().numpy()

#with open(tile_dict_path, 'rb') as f:
#    tile_dict = pickle.load(f)

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

    for file in file_list:
        tile_paths, label = tile_dict[file]
        print(tile_paths)
        print(label)

        folder = classes[label]

        def add_folder(tile_path):
            if tile_path == '':
                return ''
            else:
                return folder + '/' + tile_path

        # Add the folder for the class name in front
        add_folder_v = np.vectorize(add_folder)
        tile_paths = add_folder_v(tile_paths)

        # Get the probability array for the file
        prob_v = np.vectorize(get_tile_probability, otypes=[np.ndarray])
        probabilities = prob_v(tile_paths)


        """
        imgSize = probabilities.shape()
        newShape = (imgSize[0], imgSize[1], 3)
        probabilities = np.reshape(np.stack(probabilities.flat), newShape)
        """

        if method == 'average':
            probabilities = np.stack(probabilities.flat)
            prediction = np.nanmean(probabilities, axis = 0)

        elif method == 'max':
            probabilities = np.stack(probabilities.flat)
            probabilities = probabilities[~np.isnan(probabilities).all(axis=1)]
            votes = np.nanargmax(probabilities, axis=1)
            
            out = np.array([sum(votes == i) for i in range(num_classes)])
            prediction = out / out.sum()

        else:
            raise ValueError('Method not valid')

        predictions.append(prediction)
        true_labels.append(label)

    return np.array(predictions), np.array(true_labels)

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
        self.data = 'lung'
        self.conv1 = BasicConv2d(nc, 16, False, kernel_size=5, padding=1, stride=2, bias=True)
        self.conv2 = BasicConv2d(16, 32, False, kernel_size=3, bias=True)
        self.conv3 = BasicConv2d(32, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv4 = BasicConv2d(64, 64, True, kernel_size=3, padding=1, bias=True)
        self.conv5 = BasicConv2d(64, 128, True, kernel_size=3, padding=1, bias=True)
        self.conv6 = BasicConv2d(128, 64, True, kernel_size=3, padding=1, bias=True)
        self.linear = nn.Linear(5184, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x = x6.view(x6.size(0), -1)
        x = self.linear(x)
        return x,x1,x2,x3,x4,x5,x6


model = cancer_CNN(3, imgSize, 1)
model_path = "/scratch/jmw784/capstone/deep-cancer/experiments/train_lung_es/epoch_10.pth"
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)



#following was obtained at Final evaluation and viz notebook
normal_hp_tiles = []
#solid 
normal_hp_tiles.append('test_TCGA-91-6831-11A-01-TS1.4346239f-de98-4c3f-8790-40766161d2c7_39_11.jpeg')
normal_hp_tiles.append('test_TCGA-60-2709-11A-01-BS1.177526d2-5863-4829-ad4b-7de930de16ad_27_9.jpeg')
normal_hp_tiles.append('test_TCGA-34-2600-11A-01-BS1.dd70f786-4ac1-4dfc-b66c-281e19edf374_52_17.jpeg')
normal_hp_tiles.append('test_TCGA-49-4510-11A-01-TS1.7310b502-a637-4912-857b-3c52214ad706_24_4.jpeg')
normal_hp_tiles.append('test_TCGA-44-6145-11A-01-TS1.1ce37c11-0439-4903-be45-68cd55baf942_37_6.jpeg')
normal_hp_tiles.append('test_TCGA-44-3398-11B-01-TS1.0a860872-acfa-45e3-b3ef-9ae4e6b8a4ed_30_9.jpeg')
normal_hp_tiles.append('test_TCGA-22-5477-11A-01-TS1.f67bf6f7-29a4-45ea-b663-a03ebc86391a_18_6.jpeg')
normal_hp_tiles.append('test_TCGA-44-7660-11A-01-TS1.ad09f3a7-cca1-435d-8f25-aa1633903ed8_5_18.jpeg')
normal_hp_tiles.append('test_TCGA-73-4659-11A-01-BS1.1615c681-e9d6-4255-9c48-3fad043abe88_56_2.jpeg')
normal_hp_tiles.append('test_TCGA-66-2755-11A-01-TS1.ead0333e-f9a6-4e21-82c6-9f464342b161_33_16.jpeg')
normal_hp_tiles.append('test_TCGA-51-4079-11A-01-TS1.8a22d88d-afcc-4a61-b267-780d5d738f36_14_13.jpeg')
normal_hp_tiles.append('test_TCGA-33-4583-11A-01-BS1.09818b4f-eb96-4562-abae-ae4a2ed0adfd_31_10.jpeg')
normal_hp_tiles.append('test_TCGA-44-7667-11A-01-TS1.4d2b08b2-07b1-478a-9944-934679cf74bb_21_7.jpeg')
normal_hp_tiles.append('test_TCGA-44-5645-11A-03-TS3.1DF60D63-00A1-42A4-AB5E-5327AC83AB29_63_7.jpeg')
normal_hp_tiles.append('test_TCGA-33-4583-11A-01-TS1.411467f7-1b25-4008-b82f-d508ec5e5680_40_9.jpeg')
normal_hp_tiles.append('test_TCGA-60-2698-11A-01-BS1.ca0fb378-b1f1-4bb7-b078-f56788cbec88_60_11.jpeg')


lusc_hp_tiles = []
#lusc 
lusc_hp_tiles.append('test_TCGA-98-A53A-01A-01-TS1.EADCFE65-613A-4C37-A034-C5B11D452D4F_17_8.jpeg')
lusc_hp_tiles.append('test_TCGA-66-2763-01A-01-BS1.06f7aefa-b361-44ff-a40b-2ccfd128376d_42_8.jpeg')
lusc_hp_tiles.append('test_TCGA-63-A5MP-01A-01-TS1.14673CE6-A47F-4557-BC6D-3E9846E2B0F5_15_15.jpeg')
lusc_hp_tiles.append('test_TCGA-85-8276-01A-01-TS1.2148c209-f5b0-4043-ac67-059aaf578946_16_32.jpeg')
lusc_hp_tiles.append('test_TCGA-66-2763-01A-01-TS1.af1c3100-573f-41a4-a9b9-30e83c0546d2_21_17.jpeg')
lusc_hp_tiles.append('test_TCGA-94-7033-01A-01-TS1.d38f20aa-3a65-4af3-a58d-5837a5a4a631_45_24.jpeg')
lusc_hp_tiles.append('test_TCGA-66-2755-01A-01-BS1.e052d5dd-852d-4835-bace-71bbb1ca3325_9_2.jpeg')
lusc_hp_tiles.append('test_TCGA-22-4593-01A-01-BS1.cd5e248f-9540-4bf6-8d6c-ce86f57bd74d_9_14.jpeg')
lusc_hp_tiles.append('test_TCGA-77-8008-01A-01-BS1.9c362be3-0860-4500-9805-6ae28d23b685_35_8.jpeg')
lusc_hp_tiles.append('test_TCGA-43-6143-01A-01-TS1.6716067a-f179-46cf-8d38-7f7cd8b25d5d_27_7.jpeg')
lusc_hp_tiles.append('test_TCGA-90-7964-01A-01-BS1.e6e284a9-c179-41dc-9c15-28ba4e941111_19_23.jpeg')
lusc_hp_tiles.append('test_TCGA-NC-A5HM-01A-01-TSA.13947B81-4DB8-464A-9225-73DAB87BDABF_66_15.jpeg')
lusc_hp_tiles.append('test_TCGA-34-5239-01A-01-BS1.a1dcf1cd-e343-4310-bba1-d8c8f5014e49_37_3.jpeg')
lusc_hp_tiles.append('test_TCGA-56-A5DR-01A-01-TS1.11A39D71-8B66-4106-B43B-AB5652B25E90_71_8.jpeg')
lusc_hp_tiles.append('test_TCGA-51-6867-01A-01-TS1.6e5b5487-8e8a-4448-bb9f-9e904196566c_6_11.jpeg')
lusc_hp_tiles.append('test_TCGA-77-8148-01A-01-TS1.84ca2884-3397-4362-94a2-fc3bd012094d_10_10.jpeg')


luad_hp_tiles = []
#luad
luad_hp_tiles.append('test_TCGA-91-8499-01A-01-TS1.157b5269-563d-4a4d-b46e-f355362035c0_9_1.jpeg')
luad_hp_tiles.append('test_TCGA-97-8175-01A-01-TS1.805dfbde-6af0-4860-b04e-c6fb6087d8c0_17_9.jpeg')
luad_hp_tiles.append('test_TCGA-44-3398-01A-01-BS1.8347f335-cd9d-46cc-bed5-b28f9dd877a8_36_29.jpeg')
luad_hp_tiles.append('test_TCGA-55-7994-01A-01-TS1.8b666cd9-9f50-4be5-a12f-df25b84934c3_12_18.jpeg')
luad_hp_tiles.append('test_TCGA-86-7714-01A-01-BS1.96ae5077-ba6a-47e1-893d-63e0bcea734d_6_12.jpeg')
luad_hp_tiles.append('test_TCGA-44-5643-01A-01-BS1.13ee8806-e27c-48a4-8d93-9a6a42668cd8_32_3.jpeg')
luad_hp_tiles.append('test_TCGA-78-7536-01A-01-TS1.29490ee7-b0ce-437f-9a5d-fc643c3d38d8_6_1.jpeg')
luad_hp_tiles.append('test_TCGA-44-5645-01B-04-BS4.76ED5128-FB3D-44C7-A679-848860CAA3C5_12_5.jpeg')
luad_hp_tiles.append('test_TCGA-44-7660-01A-01-TS1.c1e1a47f-0836-4cc8-8dac-b91f51a64fac_20_8.jpeg')
luad_hp_tiles.append('test_TCGA-49-4510-01A-01-BS1.1690f93b-764b-43e8-a89c-af3cbe389a40_31_26.jpeg')
luad_hp_tiles.append('test_TCGA-80-5611-01A-01-TS1.b7a91bf2-1222-4745-b089-f2d847cb26dd_14_4.jpeg')
luad_hp_tiles.append('test_TCGA-69-7761-01A-01-BS1.19480ea3-7dbd-4a5c-a775-a736a89eeb9b_8_19.jpeg')
luad_hp_tiles.append('test_TCGA-44-6775-01A-03-TS3.DE1214F1-C47F-4710-BD72-3AFAF3D920E3_13_7.jpeg')
luad_hp_tiles.append('test_TCGA-67-6217-01A-01-BS1.49c857c1-9817-48ed-8b34-5b168c2496c7_12_5.jpeg')
luad_hp_tiles.append('test_TCGA-62-A46V-01A-01-TSA.D5EF7ED6-0460-47F4-B4F0-550843903CC6_62_6.jpeg')
luad_hp_tiles.append('test_TCGA-L9-A743-01A-04-TS4.873E50BA-0B07-4174-9FBE-7A830A85F83E_15_22.jpeg')



lusc_path = '/beegfs/jmw784/Capstone/LungTilesSorted/TCGA-LUSC/'
luad_path = '/beegfs/jmw784/Capstone/LungTilesSorted/TCGA-LUAD/'
normal_path = '/beegfs/jmw784/Capstone/LungTilesSorted/Solid_Tissue_Normal/'


    
#lusc

image_name = 'lusc'
images = []

fig = plt.figure(figsize=(16,16))

for im_ind in range(16):
    tile_path = lusc_path + lusc_hp_tiles[im_ind]
    with open(tile_path, 'rb') as f:
            with Image.open(f) as img:
                im = img.convert('RGB')
                images.append(im)
                
    ax1 = fig.add_subplot(4,4,im_ind+1)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])


fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/original_'+image_name+'.png', format='png', dpi=1200)
fig.clf()
plt.close(fig)

filter_output1 = []
filter_output2 = []
filter_output3 = []
filter_output4 = []
filter_output5 = []
filter_output6 = []

for i in range(16):
    
    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(images[i]).unsqueeze(0)

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    #output = F.softmax(model(var_img)).data.squeeze(0)

    #x1,x2,x3,x4,x5,x6 is the output image after convolution
    x,x1,x2,x3,x4,x5,x6 = model(var_img) 

    filter_output1.append(x1.data.cpu().numpy())
    filter_output2.append(x2.data.cpu().numpy())
    filter_output3.append(x3.data.cpu().numpy())
    filter_output4.append(x4.data.cpu().numpy())
    filter_output5.append(x5.data.cpu().numpy())
    filter_output6.append(x6.data.cpu().numpy())
    
    
for f in range(16):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output1[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L1_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(32):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output2[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L2_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output3[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L3_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output4[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L4_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()    
    plt.close(fig)
for f in range(128):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output5[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L5_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()    
    plt.close(fig)
    
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output6[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L6_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
    
del images, filter_output1, filter_output2, filter_output3, filter_output4, filter_output5, filter_output6
    
#luad

image_name = 'luad'
images = []

fig = plt.figure(figsize=(16,16))

for im_ind in range(16):
    tile_path = luad_path + luad_hp_tiles[im_ind]
    with open(tile_path, 'rb') as f:
            with Image.open(f) as img:
                im = img.convert('RGB')
                images.append(im)
                
    ax1 = fig.add_subplot(4,4,im_ind+1)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])


fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/original_'+image_name+'.png', format='png', dpi=1200)
fig.clf()
plt.close(fig)
filter_output1 = []
filter_output2 = []
filter_output3 = []
filter_output4 = []
filter_output5 = []
filter_output6 = []

for i in range(16):
    
    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(images[i]).unsqueeze(0)

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    #output = F.softmax(model(var_img)).data.squeeze(0)

    #x1,x2,x3,x4,x5,x6 is the output image after convolution
    x,x1,x2,x3,x4,x5,x6 = model(var_img) 

    filter_output1.append(x1.data.cpu().numpy())
    filter_output2.append(x2.data.cpu().numpy())
    filter_output3.append(x3.data.cpu().numpy())
    filter_output4.append(x4.data.cpu().numpy())
    filter_output5.append(x5.data.cpu().numpy())
    filter_output6.append(x6.data.cpu().numpy())
    
    
for f in range(16):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output1[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L1_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(32):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output2[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L2_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output3[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L3_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output4[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L4_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf() 
    plt.close(fig)
for f in range(128):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output5[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L5_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()  
    plt.close(fig)
    
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output6[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L6_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)

del images, filter_output1, filter_output2, filter_output3, filter_output4, filter_output5, filter_output6

    
#normal

image_name = 'normal'
images = []

fig = plt.figure(figsize=(16,16))

for im_ind in range(16):
    tile_path = normal_path + normal_hp_tiles[im_ind]
    with open(tile_path, 'rb') as f:
            with Image.open(f) as img:
                im = img.convert('RGB')
                images.append(im)
                
    ax1 = fig.add_subplot(4,4,im_ind+1)
    ax1.imshow(im)
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    

fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/original_'+image_name+'.png', format='png', dpi=1200)
fig.clf()
plt.close(fig)
filter_output1 = []
filter_output2 = []
filter_output3 = []
filter_output4 = []
filter_output5 = []
filter_output6 = []

for i in range(16):
    
    # Model expects a 4D tensor, unsqueeze first dimension
    img = transform(images[i]).unsqueeze(0)

    # Turn output into probabilities with softmax
    var_img = Variable(img, volatile=True)
    #output = F.softmax(model(var_img)).data.squeeze(0)

    #x1,x2,x3,x4,x5,x6 is the output image after convolution
    x,x1,x2,x3,x4,x5,x6 = model(var_img) 

    filter_output1.append(x1.data.cpu().numpy())
    filter_output2.append(x2.data.cpu().numpy())
    filter_output3.append(x3.data.cpu().numpy())
    filter_output4.append(x4.data.cpu().numpy())
    filter_output5.append(x5.data.cpu().numpy())
    filter_output6.append(x6.data.cpu().numpy())
    
    
for f in range(16):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output1[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L1_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(32):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output2[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L2_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output3[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L3_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()
    plt.close(fig)
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output4[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L4_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf() 
    plt.close(fig)
for f in range(128):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output5[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L5_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf()   
    plt.close(fig)
    
for f in range(64):
    fig = plt.figure(figsize=(16,16))
    for ii in range(16):
        ax1 = fig.add_subplot(4,4,ii+1)
        ax1.imshow(filter_output6[ii][0][f])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.savefig('/scratch/sb3923/deep-cancer/filter_images/filters2/L6_F'+str(f+1)+'_'+image_name+'.png', format='png', dpi=1200)
    fig.clf() 
    plt.close(fig)
    
    
del images, filter_output1, filter_output2, filter_output3, filter_output4, filter_output5, filter_output6
