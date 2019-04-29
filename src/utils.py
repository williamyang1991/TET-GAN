import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import os
from data_augmentation import generate_styles
import random

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

# custom weights initialization called on networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# view images
def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

# load one image in tensor format
def load_image(filename, load_type=0, wd=256, ht=256):
    centerCrop = transforms.CenterCrop((wd, ht))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = transform(centerCrop(Image.open(filename)))
    else:
        img = transform(centerCrop(text_image_preprocessing(filename)))
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

# black and white text image to distance-based text image
def text_image_preprocessing(filename):
    I = np.array(Image.open(filename))
    BW = I[:,:,0] > 127
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel>255]=255
    B_channel = pyimg.distance_transform_edt(1-BW)
    B_channel[B_channel>255]=255
    I[:,:,1] = G_channel.astype('uint8')
    I[:,:,2] = B_channel.astype('uint8')
    return Image.fromarray(I)

# prepare batched filenames of all training data
def load_trainset_batchfnames(filepath, batch_size, usetrainnum=708, trainnum=100000):
    paths = os.listdir(filepath)
    stylenum = len(paths)
    trainnum = (trainnum / batch_size / 2) * batch_size * 2
    fnames = ['%s.png' % (i%usetrainnum+1) for i in range(trainnum)]
    pathid = [(i%stylenum) for i in range(trainnum/2)]
    random.shuffle(pathid)
    random.shuffle(fnames)
    trainbatches = [([]) for _ in range(trainnum/batch_size/2)]
    for i in range(trainnum/batch_size/2):
        traindatas = []
        for j in range(batch_size):
            ii = i * batch_size + j * 2
            traindatas += [[os.path.join(filepath, paths[pathid[ii/2]], 'train', fnames[ii]),
                          os.path.join(filepath, paths[pathid[(ii+1)/2]], 'train', fnames[ii+1])]]  
        trainbatches[i] += traindatas
    return trainbatches

# prepare batched filename of the style image for one shot learning
def load_oneshot_batchfnames(filename, batch_size, trainnum=100000):
    trainnum = (trainnum / batch_size / 2) * batch_size * 2
    trainbatches = [([]) for _ in range(trainnum/batch_size/2)]
    for i in range(trainnum/batch_size/2):
        traindatas = []
        for j in range(batch_size):
            traindatas += [[filename,filename]]  
        trainbatches[i] += traindatas
    return trainbatches

# prepare (x,y,y') at target resolution levels
def prepare_batch(batchfnames, level=3, jitter=0.0, centercropratio=0.5, augementratio=0.0, gpu=True, wd=256, ht=256):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    # level1: input x: 64*64
    # level2: input x: 128*128, x2: 64*64
    # level3: input x: 256*256, x2: 128*128
    downsamplerates = [[2**2], [2**1,2**2], [2**0,2**1]]
    layernum = len(downsamplerates[level-1])
    img_wds = [(wd/downsamplerates[level-1][i]) for i in range(layernum)]
    img_hts = [(ht/downsamplerates[level-1][i]) for i in range(layernum)]

    centerCrop = transforms.CenterCrop((wd, ht))
    img_list = [([]) for _ in range(layernum)]
    img_list2 = [([]) for _ in range(layernum)]
    
    for fname1, fname2 in batchfnames:
        img1 = Image.open(fname1)
        img2 = Image.open(fname2)
        ori_wd, ori_ht = img1.size
        ori_wd = ori_wd / 2
        img1_in = img1.crop((0,0,ori_wd,ori_ht))
        img1_out = img1.crop((ori_wd,0,ori_wd*2,ori_ht))
        img2_in = img2.crop((0,0,ori_wd,ori_ht))
        img2_out = img2.crop((ori_wd,0,ori_wd*2,ori_ht))
        
        if random.random() < augementratio:
            # use augmented data
            img1_out, img2_out = generate_styles(img1_in, img2_in)
        elif jitter != 0.0:
            # add color, contrast, hue jitter
            img1.paste(img2_out,(0,0))
            colorjitter = transforms.ColorJitter(jitter*0.125,jitter*0.125,jitter*0.125,jitter*0.5)
            img1 = colorjitter(img1)
            img2_out = img1.crop((0,0,ori_wd,ori_ht))
            img1_out = img1.crop((ori_wd,0,ori_wd*2,ori_ht))
        
        if random.random() <= centercropratio:
            # use center crop
            img1_in = centerCrop(img1_in)
            img1_out = centerCrop(img1_out)
            img2_in = centerCrop(img2_in)
            img2_out = centerCrop(img2_out)
        else:
            # use random crop
            x = random.randint(0,ori_wd-wd)
            y = random.randint(0,ori_ht-ht)
            img1_in = img1_in.crop((x,y,x+wd,y+ht))
            img1_out = img1_out.crop((x,y,x+wd,y+ht))
            x = random.randint(0,ori_wd-wd)
            y = random.randint(0,ori_ht-ht)            
            img2_in = img2_in.crop((x,y,x+wd,y+ht))
            img2_out = img2_out.crop((x,y,x+wd,y+ht)) 
        
        # create data at different levels
        for i in range(layernum):
            img1_in_ = transform(img1_in.resize((img_wds[i], img_hts[i]),Image.LANCZOS))
            img2_in_ = transform(img2_in.resize((img_wds[i], img_hts[i]),Image.LANCZOS))
            img1_out_ = transform(img1_out.resize((img_wds[i], img_hts[i]),Image.LANCZOS))
            img2_out_ = transform(img2_out.resize((img_wds[i], img_hts[i]),Image.LANCZOS))
            imout = torch.cat((img1_in_,img1_out_), dim = 0)
            imout = imout.unsqueeze(dim=0)
            img_list[i].append(imout)
            imout = torch.cat((img2_in_,img2_out_), dim = 0)
            imout = imout.unsqueeze(dim=0)
            img_list2[i].append(imout)
        
    ins = []    # x
    outs1 = []  # y
    outs2 = []  # y'
    for i in range(layernum):
        data1 = to_var(torch.cat(img_list[i], dim=0))
        data2 = to_var(torch.cat(img_list2[i], dim=0))
        ins += [torch.cat((data1[:,0:3,:,:], data2[:,0:3,:,:]),dim=0)]
        outs1 += [torch.cat((data1[:,3:,:,:], data2[:,3:,:,:]),dim=0)]
        outs2 += [torch.cat((data2[:,3:,:,:], data1[:,3:,:,:]),dim=0)]
    
    return ins, outs1, outs2