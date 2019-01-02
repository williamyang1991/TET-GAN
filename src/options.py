import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--style_name', type=str, default='../data/style/26.jpg', help='path of the style image')
        self.parser.add_argument('--content_name', type=str, default='../data/content/3.png', help='path of the content image')
        self.parser.add_argument('--c2s', type=int, default=1, help='translation direction, 1 for stlization, 0 for destylization')
        self.parser.add_argument('--content_type', type=int, default=0, help='0 for distance-based text image, 1 for black and white text image')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='output.png', help='file name of the outputs')
        self.parser.add_argument('--result_dir', type=str, default='../output/', help='path for saving result images')

        # model related
        self.parser.add_argument('--model', type=str, default='../save/tetgan-aaai.ckpt', help='specified the dir of saved models')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt