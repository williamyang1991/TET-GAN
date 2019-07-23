from __future__ import print_function
from options import TrainOptions
import torch
from models import TETGAN
from utils import load_image, to_data, to_var, visualize, save_image, load_trainset_batchfnames, prepare_batch, weights_init
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # data loader
    print('--- load parameter ---')
    outer_iter = opts.outer_iter
    fade_iter = max(1.0, float(outer_iter / 2))
    epochs = opts.epoch
    batchsize = opts.batchsize
    datasize = opts.datasize
    datarange = opts.datarange
    augementratio = opts.augementratio
    centercropratio = opts.centercropratio      
    
    # model
    print('--- create model ---')
    tetGAN = TETGAN(gpu = (opts.gpu!=0))
    if opts.gpu != 0:
        tetGAN.cuda()
    tetGAN.init_networks(weights_init)
    tetGAN.train()

    print('--- training ---')
    stylenames = os.listdir(opts.train_path)
    print('List of %d styles:'%(len(stylenames)), *stylenames, sep=' ')
    
    if opts.progressive == 1:
        # proressive training. From level1 64*64, to level2 128*128, to level3 256*256
        # level 1
        for i in range(outer_iter):
            jitter = min(1.0, i / fade_iter)
            fnames = load_trainset_batchfnames(opts.train_path, batchsize*4, datarange, datasize*2)
            for epoch in range(epochs):
                for fname in fnames:
                    x, y_real, y = prepare_batch(fname, 1, jitter, 
                                                 centercropratio, augementratio, opts.gpu)
                    losses = tetGAN.one_pass(x[0], None, y[0], None, y_real[0], None, 1, None)
                print('Level1, Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4]))
        # level 2
        for i in range(outer_iter):
            w = max(0.0, 1 - i / fade_iter)
            fnames = load_trainset_batchfnames(opts.train_path, batchsize*2, datarange, datasize*2)
            for epoch in range(epochs):
                for fname in fnames:
                    x, y_real, y = prepare_batch(fname, 2, 1, 
                                                 centercropratio, augementratio, opts.gpu)
                    losses = tetGAN.one_pass(x[0], x[1], y[0], y[1], y_real[0], y_real[1], 2, w)
                print('Level2, Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4]))
        # level 3
        for i in range(outer_iter):
            w = max(0.0, 1 - i / fade_iter)
            fnames = load_trainset_batchfnames(opts.train_path, batchsize, datarange, datasize)
            for epoch in range(epochs):
                for fname in fnames:
                    x, y_real, y = prepare_batch(fname, 3, 1, 
                                                 centercropratio, augementratio, opts.gpu)
                    losses = tetGAN.one_pass(x[0], x[1], y[0], y[1], y_real[0], y_real[1], 3, w)
                print('Level3, Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4])) 
    else:
        # directly train on level3 256*256
        for i in range(outer_iter):
            fnames = load_trainset_batchfnames(opts.train_path, batchsize, datarange, datasize)
            for epoch in range(epochs):
                for fname in fnames:
                    x, y_real, y = prepare_batch(fname, 3, 1, 
                                                 centercropratio, augementratio, opts.gpu)
                    losses = tetGAN.one_pass(x[0], None, y[0], None, y_real[0], None, 3, 0)
                print('Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4]))        
    
    print('--- save ---')
    torch.save(tetGAN.state_dict(), opts.save_model_name)

if __name__ == '__main__':
    main()
