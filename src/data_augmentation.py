import torch

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import random

# generating random text effects on the distance map
# grayimg1, gratimg2: two distance maps to be colorized using the same text effects
# maxcolornum: determine the richness of color
def colorize_two(grayimg1, grayimg2, maxcolornum):   
    # number of color anchors
    if maxcolornum == 0:
        colornum = 2
    else:
        colornum = random.randint(1,maxcolornum) + 2
    
    # max distance
    maxdist = max(np.max(grayimg1), np.max(grayimg2))

    # checkpoints to create color anchors 
    checkpoints = []
    checkpoints+=[int(np.round(i*(maxdist/(colornum-1.0)))) for i in range(colornum-1)]
    checkpoints+=[int(maxdist)]
    checkpoints+=[int(256)]
    
    # color map
    cmap = np.random.randint(0,255,(colornum+1,3))

    mx = 256  # if gray.dtype==np.uint8 else 65535
    lut = np.empty(shape=(256, 3))

    lastval = checkpoints[0]
    lastcol = cmap[0]
    for i in range(colornum):
        col = cmap[i+1]
        val = checkpoints[i+1]
        for i in range(3):
            lut[lastval:val, i] = np.linspace(
                lastcol[i], col[i], val - lastval)
        lastcol = col
        lastval = val

    # generating text effects on grayimg1
    [w1, h1] = grayimg1.shape
    colorimg1 = np.empty(shape=(w1, h1, 3), dtype=np.uint8)

    for i in range(3):
        colorimg1[..., i] = cv2.LUT(grayimg1, lut[:, i])
            
    colorimg1 = np.clip(cv2.GaussianBlur(colorimg1, (3,3), 3, 1), a_min=0, a_max=255) / 255.0
    colorimg1 = np.transpose(colorimg1, axes=(2,0,1))
    
    # generating text effects on grayimg2
    [w2, h2] = grayimg2.shape
    colorimg2 = np.empty(shape=(w2, h2, 3), dtype=np.uint8)

    for i in range(3):
        colorimg2[..., i] = cv2.LUT(grayimg2, lut[:, i])
    
    colorimg2 = np.clip(cv2.GaussianBlur(colorimg2, (3,3), 3, 1), a_min=0, a_max=255) / 255.0
    colorimg2 = np.transpose(colorimg2, axes=(2,0,1))
    
    return [torch.tensor(colorimg1, dtype=torch.float32), torch.tensor(colorimg2, dtype=torch.float32)]

# generating random text effects based on the pixel distance from the glyph
# img1, img2: two PIL images to be renderred
def generate_styles(img1, img2):  
    r, g, b = img1.split()
    mask1 = transforms.ToTensor()(r).repeat(3,1,1)
    fg1 = transforms.ToTensor()(g)*255
    bg1 = transforms.ToTensor()(b)*255
    r, g, b = img2.split()
    mask2 = transforms.ToTensor()(r).repeat(3,1,1)
    fg2 = transforms.ToTensor()(g)*255 
    bg2 = transforms.ToTensor()(b)*255
    [fgc1, fgc2] = colorize_two(fg1.squeeze().numpy().astype(np.uint8), 
                                fg2.squeeze().numpy().astype(np.uint8), 0)
    [bgc1, bgc2] = colorize_two(bg1.squeeze().numpy().astype(np.uint8), 
                                bg2.squeeze().numpy().astype(np.uint8), 3)
    texteffects1 = fgc1*mask1+bgc1*(1-mask1)
    texteffects2 = fgc2*mask2+bgc2*(1-mask2) 
    return transforms.ToPILImage()(texteffects1), transforms.ToPILImage()(texteffects2)