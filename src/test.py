from options import TestOptions
import torch
from models import TETGAN
from utils import load_image, to_data, to_var, visualize, save_image
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('--- load data ---')
    style = load_image(opts.style_name)
    if opts.gpu != 0:
        style = to_var(style)
    if opts.c2s == 1:
        content = load_image(opts.content_name, opts.content_type)
        if opts.gpu != 0:
            content = to_var(content)
    
    # model
    print('--- load model ---')
    tetGAN = TETGAN()
    tetGAN.load_state_dict(torch.load(opts.model))
    if opts.gpu != 0:
        tetGAN.cuda()
    tetGAN.eval()
    
    print('--- testing ---')
    if opts.c2s == 1:
        result = tetGAN(content, style)
    else:
        result = tetGAN.desty_forward(style)
    if opts.gpu != 0:
        result = to_data(result)
        
    print('--- save ---')
    # directory
    result_filename = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)
    save_image(result[0], result_filename)

if __name__ == '__main__':
    main()
