import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg

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

def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    
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