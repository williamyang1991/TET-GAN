import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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
    
    # input x: 64*64
    def forward_level1(self, x):
        return self.rcb4(self.rcb3(self.rcb2(self.conv2(x))))
    
    # input x: 128*128, x2: 64*64, w: fading out weight
    def forward_level2(self, x, x2, w):
        return self.rcb4(self.rcb3(self.rcb2(self.conv2(x2)*w
                                                 +self.rcb1(self.conv1(x))*(1-w))))
    
    # input  x: 256*256, x2: 128*128, w: fading out weight
    def forward_level3(self, x, x2, w):
        if w == 0:
            return self.rcb4(self.rcb3(self.rcb2(self.rcb1(self.rcb0(self.conv0(x))))))
        else:
            return self.rcb4(self.rcb3(self.rcb2(self.rcb1(self.conv1(x2)*w
                                                        +self.rcb0(self.conv0(x))*(1-w)))))
        

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
    
    # output 64*64
    def forward_level1(self, x):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(x)))
        return self.tanh(self.conv2(F.relu(x2)))
    
    # output 128*128
    def forward_level2(self, x):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(x)))
        return self.tanh(self.conv1(F.relu(self.rdcb1(x2))))
    
    # output 256*256
    def forward_level3(self, x):
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
    
    # output 64*64
    def forward_level1(self, x, y):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(torch.cat((x, y), dim=1))))
        return self.tanh(self.conv2(F.relu(x2)))
    
    # output 128*128
    def forward_level2(self, x, y):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(torch.cat((x, y), dim=1))))
        return self.tanh(self.conv1(F.relu(self.rdcb1(x2))))
    
    # output 256*256
    def forward_level3(self, x, y):
        x2 = self.rdcb2(self.rdcb3(self.rdcb4(torch.cat((x, y), dim=1))))
        return self.tanh(self.conv0(F.relu(self.rdcb0(self.rdcb1(x2)))))
    
    
class TETGAN(nn.Module):
    def __init__(self, nef=32, ndf=64, gpu = True):
        super(TETGAN, self).__init__()
        
        self.nef = nef
        self.ngf = nef
        self.nsf = nef * 16
        self.ndf = ndf
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.gpu = gpu
        
        self.E_x = Encoder(self.nef)
        self.E_yc = Encoder(self.nef, useIn=True)
        self.S_x = ShareLayer(self.nsf)
        self.G_x = Generator(self.ngf)
        self.E_ys = Encoder(self.nef)
        self.S_y = ShareLayer(self.nsf)
        self.G_y = ConcatGenerator(self.ngf) 
        self.D_x = Discriminator(in_channels=6, ndf=self.ndf)
        self.D_y = Discriminator(in_channels=9, ndf=self.ndf)
        
        self.trainerE_x = torch.optim.Adam(self.E_x.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerE_yc = torch.optim.Adam(self.E_yc.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerS_x = torch.optim.Adam(self.S_x.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG_x = torch.optim.Adam(self.G_x.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerE_ys = torch.optim.Adam(self.E_ys.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerS_y = torch.optim.Adam(self.S_y.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG_y = torch.optim.Adam(self.G_y.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_x = torch.optim.Adam(self.D_x.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_y = torch.optim.Adam(self.D_y.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.loss = nn.L1Loss()
    
    # FOR TESTING
    # style transfer
    def forward(self, x, y):
        content_feat = self.S_x(self.E_x(x))
        style_feat = self.S_y(self.E_ys(y))
        return self.G_y(content_feat, style_feat)
    
    # destylization
    def desty_forward(self, y):
        return self.G_x(self.S_x(self.E_yc(y)))
    
    
    # FOR TRAINING
    # content autoencoder reconstruction
    def autoencoder_level1(self, x):
        return self.G_x.forward_level1(self.S_x(self.E_x.forward_level1(x)))
    def autoencoder_level2(self, x, x2, w):
        return self.G_x.forward_level2(self.S_x(self.E_x.forward_level2(x, x2, w)))
    def autoencoder_level3(self, x, x2, w):
        return self.G_x.forward_level3(self.S_x(self.E_x.forward_level3(x, x2, w)))
    
    # destylization
    def destylization_level1(self, y):
        content_feat = self.S_x(self.E_yc.forward_level1(y))
        return [self.G_x.forward_level1(content_feat), content_feat]
    def destylization_level2(self, y, y2, w):
        content_feat = self.S_x(self.E_yc.forward_level2(y, y2, w))  
        return [self.G_x.forward_level2(content_feat), content_feat]
    def destylization_level3(self, y, y2, w):
        content_feat = self.S_x(self.E_yc.forward_level3(y, y2, w))
        return [self.G_x.forward_level3(content_feat), content_feat]
    def get_guidance_content_feature_level1(self, x):
        return self.S_x(self.E_x.forward_level1(x))
    def get_guidance_content_feature_level2(self, x, x2, w):
        return self.S_x(self.E_x.forward_level2(x, x2, w))
    def get_guidance_content_feature_level3(self, x, x2, w):
        return self.S_x(self.E_x.forward_level3(x, x2, w))
    
    # stylization
    def stylization_level1(self, x, y):
        content_feat = self.S_x(self.E_x.forward_level1(x))
        style_feat = self.S_y(self.E_ys.forward_level1(y))
        return self.G_y.forward_level1(content_feat, style_feat)
    def stylization_level2(self, x, x2, y, y2, w):
        content_feat = self.S_x(self.E_x.forward_level2(x, x2, w))
        style_feat = self.S_y(self.E_ys.forward_level2(y, y2, w))        
        return self.G_y.forward_level2(content_feat, style_feat)
    def stylization_level3(self, x, x2, y, y2, w):
        content_feat = self.S_x(self.E_x.forward_level3(x, x2, w))
        style_feat = self.S_y(self.E_ys.forward_level3(y, y2, w))        
        return self.G_y.forward_level3(content_feat, style_feat)
    
    # style autoencoder resconstruction
    def style_autoencoder(self, y):
        content_feat = self.S_x(self.E_yc(y))
        style_feat = self.S_y(self.E_ys(y))
        return self.G_y(content_feat, style_feat)
    
    # init weight
    def init_networks(self, weights_init):
        self.E_x.apply(weights_init)
        self.E_yc.apply(weights_init)
        self.S_x.apply(weights_init)
        self.G_x.apply(weights_init)
        self.E_ys.apply(weights_init)
        self.S_y.apply(weights_init)
        self.G_y.apply(weights_init)
        self.D_x.apply(weights_init)
        self.D_y.apply(weights_init)
    
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if torch.cuda.is_available() else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def update_autoencoder(self, x, x2, level, w):
        if level == 1:
            x_fake = self.autoencoder_level1(x)
        elif level == 2:
            x_fake = self.autoencoder_level2(x, x2 ,w)
        else:
            x_fake = self.autoencoder_level3(x, x2 ,w)
        Lrec = self.loss(x_fake, x) * self.lambda_l1            
        self.trainerG_x.zero_grad()            
        self.trainerS_x.zero_grad()
        self.trainerE_x.zero_grad()
        Lrec.backward() 
        self.trainerG_x.step()
        self.trainerS_x.step()
        self.trainerE_x.step()
        return Lrec.data.mean()
    
    def update_destylization_discriminator(self, y, y2, x, level, w):
        with torch.no_grad():
            if level == 1:
                x_fake = self.destylization_level1(y)[0]
            elif level == 2:
                x_fake = self.destylization_level2(y, y2 ,w)[0]
            else:
                x_fake = self.destylization_level3(y, y2 ,w)[0]
            fake_concat = torch.cat((y, x_fake), dim=1)
        fake_output = self.D_x(fake_concat)
        real_concat = torch.cat((y, x), dim=1)
        real_output = self.D_x(real_concat)        
        gradient_penalty = self.calc_gradient_penalty(self.D_x, real_concat.data, fake_concat.data)
        Ldadv = fake_output.mean() - real_output.mean() + self.lambda_gp * gradient_penalty        
        self.trainerD_x.zero_grad() 
        Ldadv.backward()
        self.trainerD_x.step()
        return (real_output.mean() - fake_output.mean()).data.mean()
    
    def update_destylization_generator(self, y, y2, x, x2, level, w):
        with torch.no_grad():
            if level == 1:
                z = self.get_guidance_content_feature_level1(x)
            elif level == 2:
                z = self.get_guidance_content_feature_level2(x, x2, w)
            else:
                z = self.get_guidance_content_feature_level3(x, x2, w)
        if level == 1:
            output = self.destylization_level1(y)  
        elif level == 2:
            output = self.destylization_level2(y, y2 ,w)
        else:
            output = self.destylization_level3(y, y2 ,w)
        x_fake = output[0]
        x_feature = output[1]
        fake_concat = torch.cat((y, x_fake), dim=1)
        fake_output = self.D_x(fake_concat)
        Ldfeat = self.loss(x_feature, z) * self.lambda_l1
        Ldadv = -fake_output.mean()
        Ldpix = self.loss(x_fake, x) * self.lambda_l1
        Ldesty = Ldfeat + Ldadv + Ldpix        
        self.trainerE_yc.zero_grad()
        self.trainerS_x.zero_grad()
        self.trainerG_x.zero_grad()
        Ldesty.backward()
        self.trainerE_yc.step()
        self.trainerS_x.step()
        self.trainerG_x.step()  
        return Ldesty.data.mean()
        
    def update_stylization_discriminator(self, x, x2, y, y2, y_real, level, w):
        with torch.no_grad():
            if level == 1:
                y_fake = self.stylization_level1(x, y)
            elif level == 2:
                y_fake = self.stylization_level2(x, x2, y, y2, w)
            else:
                y_fake = self.stylization_level3(x, x2, y, y2, w)
            fake_concat = torch.cat((x, y, y_fake), dim=1)
        fake_output = self.D_y(fake_concat)
        real_concat = torch.cat((x, y, y_real), dim=1)
        real_output = self.D_y(real_concat)        
        gradient_penalty = self.calc_gradient_penalty(self.D_y, real_concat.data, fake_concat.data)
        Lsadv = fake_output.mean() - real_output.mean() + self.lambda_gp * gradient_penalty        
        self.trainerD_y.zero_grad() 
        Lsadv.backward()
        self.trainerD_y.step()
        return (real_output.mean() - fake_output.mean()).data.mean()
    
    def update_stylization_generator(self, x, x2, y, y2, y_real, level, w):
        if level == 1:
            y_fake = self.stylization_level1(x, y)
        elif level == 2:
            y_fake = self.stylization_level2(x, x2, y, y2, w)
        else:
            y_fake = self.stylization_level3(x, x2, y, y2, w)
        fake_concat = torch.cat((x, y, y_fake), dim=1)
        fake_output = self.D_y(fake_concat)
        Lsadv = -fake_output.mean()
        Lspix = self.loss(y_fake, y_real) * self.lambda_l1
        Lsty = Lsadv + Lspix        
        self.trainerE_x.zero_grad()
        self.trainerS_x.zero_grad()
        self.trainerE_ys.zero_grad()
        self.trainerS_y.zero_grad()   
        self.trainerG_y.zero_grad()   
        Lsty.backward()
        self.trainerE_x.step()
        self.trainerS_x.step()
        self.trainerE_ys.step()  
        self.trainerS_y.step()
        self.trainerG_y.step() 
        return Lsty.data.mean()    

    def update_style_autoencoder(self, y):
        y_fake = self.style_autoencoder(y)
        Lsrec = self.loss(y_fake, y) * self.lambda_l1            
        self.trainerE_yc.zero_grad()            
        self.trainerE_ys.zero_grad()
        self.trainerS_x.zero_grad()
        self.trainerS_y.zero_grad()
        self.trainerG_y.zero_grad()
        Lsrec.backward() 
        self.trainerE_yc.step()
        self.trainerE_ys.step()
        self.trainerS_x.step()
        self.trainerS_y.step()
        self.trainerG_y.step()     
        return Lsrec.data.mean()        
    
    # (x, y_real, y) represents (x, y, y')
    # x, y, y_real: images at current level
    # x2, y2, y2_real: images at last level
    # w: fading out weight. 
    def one_pass(self, x, x2, y, y2, y_real, y2_real, level, w):
        Lrec = self.update_autoencoder(x, x2, level, w)
        Ldadv = self.update_destylization_discriminator(y_real, y2_real, x, level, w)
        Ldesty = self.update_destylization_generator(y_real, y2_real, x, x2, level, w)
        Lsadv = self.update_stylization_discriminator(x, x2, y, y2, y_real, level, w)
        Lsty = self.update_stylization_generator(x, x2, y, y2, y_real, level, w)
        return [Lrec, Ldadv, Ldesty, Lsadv, Lsty]
        
