from __future__ import print_function
from options import FinetuneOptions
import torch
from models import TETGAN
from utils import load_image, to_data, to_var, visualize, save_image, load_oneshot_batchfnames, prepare_batch, weights_init
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = FinetuneOptions()
    opts = parser.parse()

    # data loader
    print('--- load parameter ---')
    outer_iter = opts.outer_iter
    epochs = opts.epoch
    batchsize = opts.batchsize
    datasize = opts.datasize
    stylename = opts.style_name
    
    # model
    print('--- create model ---')
    tetGAN = TETGAN(gpu = (opts.gpu!=0))        
    if opts.gpu != 0:
        tetGAN.cuda()
    tetGAN.load_state_dict(torch.load(opts.load_model_name))
    tetGAN.train()

    print('--- training ---')
    # supervised one shot learning
    if opts.supervise == 1:
        for i in range(outer_iter):
            fnames = load_oneshot_batchfnames(stylename, batchsize, datasize)
            for epoch in range(epochs):
                for fname in fnames:
                    x, y_real, y = prepare_batch(fname, 3, 0, 0, 0, opts.gpu)
                    losses = tetGAN.one_pass(x[0], None, y[0], None, y_real[0], None, 3, 0)
                print('Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4]))      
    # unsupervised one shot learning
    else:
        for i in range(outer_iter):
            fnames = load_oneshot_batchfnames(stylename, batchsize, datasize)
            for epoch in range(epochs):
                for fname in fnames:
                    # no ground truth x provided
                    _, y_real, _ = prepare_batch(fname, 3, 0, 0, 0, opts.gpu)
                    Lsrec = tetGAN.update_style_autoencoder(y_real[0])
                for fname in fnames:
                    # no ground truth x provided
                    _, y_real, y = prepare_batch(fname, 3, 0, 0, 0, opts.gpu)
                    with torch.no_grad():
                        x_auxiliary = tetGAN.desty_forward(y_real[0])
                    losses = tetGAN.one_pass(x_auxiliary, None, y[0], None, y_real[0], None, 3, 0)
                print('Iter[%d/%d], Epoch [%d/%d]' %(i+1, outer_iter, epoch+1, epochs))
                print('Lrec: %.3f, Ldadv: %.3f, Ldesty: %.3f, Lsadv: %.3f, Lsty: %.3f, Lsrec: %.3f'
                         %(losses[0], losses[1], losses[2], losses[3], losses[4], Lsrec))
    
    print('--- save ---')
    torch.save(tetGAN.state_dict(), opts.save_model_name)

if __name__ == '__main__':
    main()