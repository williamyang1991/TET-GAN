import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d
import torch.nn.functional as F
import numpy as np

# Relu + Conv + BN block
class RCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, useIn = False):
        super(RCBBlock, self).__init__()
        
        kernel_size = 4
        padding = 1
        self.relu = LeakyReLU(0.2)
        self.conv = Conv2d(out_channels=out_channels, kernel_size=kernel_size, stride=2,
                              padding=padding, in_channels=in_channels)
        if useIn:
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        else:
            self.bn = BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(self.relu(x))) 

# Relu + DeConv + BN block
class RDCBBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, useIn = False):
        super(RDCBBlock, self).__init__()
        
        kernel_size = 4
        padding = 1
        self.relu = ReLU()
        self.dconv = ConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size, stride=2,
                              padding=padding, in_channels=in_channels)
        if useIn:
            self.bn = InstanceNorm2d(num_features=out_channels, affine=True)
        else:
            self.bn = BatchNorm2d(num_features=out_channels)
            
    def forward(self, x):
        return self.bn(self.dconv(self.relu(x)))

class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        modelList = []       
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=padding, in_channels=in_channels))
        modelList.append(LeakyReLU(0.2))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modelList.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=padding, in_channels=ndf * nf_mult_prev))
            modelList.append(InstanceNorm2d(num_features=ndf * nf_mult, affine=True))
            modelList.append(LeakyReLU(0.2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modelList.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        modelList.append(InstanceNorm2d(num_features=ndf * nf_mult, affine=True))
        modelList.append(LeakyReLU(0.2))
        modelList.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult))

        self.model = nn.Sequential(*modelList)
    def forward(self, x):
        out = self.model(x)
        return out.view(-1)

# Encoder without sharing layers    
class Encoder(nn.Module):
    def __init__(self, nef=32, useIn = False):
        super(Encoder, self).__init__()
        
        # for 256*256 input in progressive training
        self.conv0 = Conv2d(out_channels=nef, kernel_size=3, stride=1,
                              padding=1, in_channels=3)
        # for 128*128 input in progressive training
        self.conv1 = Conv2d(out_channels=nef*2, kernel_size=3, stride=1,
                              padding=1, in_channels=3)
        # for 64*64 input in progressive training
        self.conv2 = Conv2d(out_channels=nef*4, kernel_size=3, stride=1,
                              padding=1, in_channels=3)          
        self.rcb0 = RCBBlock(in_channels=nef, out_channels=nef*2, useIn = useIn)
        self.rcb1 = RCBBlock(in_channels=nef*2, out_channels=nef*4, useIn = useIn)            
        self.rcb2 = RCBBlock(in_channels=nef*4, out_channels=nef*8, useIn = useIn)         
        self.rcb3 = RCBBlock(in_channels=nef*8, out_channels=nef*16, useIn = useIn)
        self.rcb4 = RCBBlock(in_channels=nef*16, out_channels=nef*16, useIn = useIn)

    def forward(self, x):
        return self.rcb4(self.rcb3(self.rcb2(self.rcb1(self.rcb0(self.conv0(x))))))

# ShareLayers are the last few layers of Encoders and the first few layers of Generators
class ShareLayer(nn.Module):
    def __init__(self, nsf=512):
        super(ShareLayer, self).__init__()
                            
        self.rcb1 = RCBBlock(in_channels=nsf, out_channels=nsf)
        self.rcb2 = RCBBlock(in_channels=nsf, out_channels=nsf)            
        self.relu = LeakyReLU(0.2)
        self.conv = Conv2d(out_channels=nsf, kernel_size=4, stride=2,
                              padding=1, in_channels=nsf)
        self.rdcb3 = RDCBBlock(in_channels=nsf, out_channels=nsf)
        self.rdcb2 = RDCBBlock(in_channels=nsf*2, out_channels=nsf)
        self.rdcb1 = RDCBBlock(in_channels=nsf*2, out_channels=nsf)
            
    def forward(self, x):
        x2 = self.rcb1(x)
        x3 = self.rcb2(x2)
        x4 = self.conv(self.relu(x3))
        x5 = torch.cat((self.rdcb3(x4), x3), dim=1)
        x6 = torch.cat((self.rdcb2(x5), x2), dim=1)
        return  torch.cat((self.rdcb1(x6), x), dim=1)
    
# Generator without sharing layers 
class Generator(nn.Module):
    def __init__(self, ngf=32):
        super(Generator, self).__init__()
        
        self.rdcb4 = RDCBBlock(in_channels=ngf*32, out_channels=ngf*16)
        self.rdcb3 = RDCBBlock(in_channels=ngf*16, out_channels=ngf*8)
        self.rdcb2 = RDCBBlock(in_channels=ngf*8, out_channels=ngf*4)
        self.rdcb1 = RDCBBlock(in_channels=ngf*4, out_channels=ngf*2)
        self.rdcb0 = RDCBBlock(in_channels=ngf*2, out_channels=ngf)

        # for 64*64 output in progressive training
        self.conv2 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf*4)
        # for 128*128 output in progressive training
        self.conv1 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf*2)
        # for 256*256 output in progressive training
        self.conv0 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf)
        self.tanh = Tanh()
            
    def forward(self, x):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(x)))
        return self.tanh(self.conv0(F.relu(self.rdcb0(self.rdcb1(x2)))))

# Generator without sharing layers 
class ConcatGenerator(nn.Module):
    def __init__(self, ngf=32):
        super(ConcatGenerator, self).__init__()

        self.rdcb4 = RDCBBlock(in_channels=ngf*64, out_channels=ngf*16)
        self.rdcb3 = RDCBBlock(in_channels=ngf*16, out_channels=ngf*8)
        self.rdcb2 = RDCBBlock(in_channels=ngf*8, out_channels=ngf*4)
        self.rdcb1 = RDCBBlock(in_channels=ngf*4, out_channels=ngf*2)
        self.rdcb0 = RDCBBlock(in_channels=ngf*2, out_channels=ngf)                        

        # for 64*64 output in progressive training
        self.conv2 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf*4)
        # for 128*128 output in progressive training
        self.conv1 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf*2)
        # for 256*256 output in progressive training
        self.conv0 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1,
                              padding=1, in_channels=ngf)
        self.tanh = Tanh()   
        
    def forward(self, x, y):   
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(torch.cat((x, y), dim=1))))
        return self.tanh(self.conv0(F.relu(self.rdcb0(self.rdcb1(x2)))))
    
class TETGAN(nn.Module):
    def __init__(self, nef=32, ndf=64):
        super(TETGAN, self).__init__()
        
        self.nef = nef
        self.ngf = nef
        self.nsf = nef * 16
        self.ndf = ndf
        
        self.E_x = Encoder(self.nef)
        self.E_yc = Encoder(self.nef, useIn=True)
        self.S_x = ShareLayer(self.nsf)
        self.G_x = Generator(self.ngf)
        self.E_ys = Encoder(self.nef)
        self.S_y = ShareLayer(self.nsf)
        self.G_y = ConcatGenerator(self.ngf) 
        self.D_x = Discriminator(in_channels=6, ndf=self.ndf)
        self.D_y = Discriminator(in_channels=9, ndf=self.ndf)
        
    # style transfer
    def forward(self, x, y):
        content_feat = self.S_x(self.E_x(x))
        style_feat = self.S_y(self.E_ys(y))
        return self.G_y(content_feat, style_feat)
    
    # destylization
    def desty_forward(self, y):
        return self.G_x(self.S_x(self.E_yc(y)))