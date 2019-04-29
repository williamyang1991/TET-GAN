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
        self.parser.add_argument('--model', type=str, default='../save/tetgan-aaai.ckpt', help='specify the model name to load')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--train_path', type=str, default='../data/dataset/', help='path of the training images')

        # train related
        self.parser.add_argument('--outer_iter', type=int, default=50, help='number of iteration for fading in progressive training')
        self.parser.add_argument('--epoch', type=int, default=5, help='number of epoch for each outer iteration')
        self.parser.add_argument('--progressive', type=int, default=1, help='1 for using progressive training, 0 for using normal training')        
        self.parser.add_argument('--batchsize', type=int, default=8, help='batchsize for level3. level3 use batchsize, level2 use 2*batchsize, level1 use 4*batchsize')   
        self.parser.add_argument('--datasize', type=int, default=12800, help='number of sampled data for each epoch')           
        self.parser.add_argument('--datarange', type=int, default=708, help='data sampling range for each style (data is sampled from 1.png ~ datarange.png)')
        self.parser.add_argument('--augementratio', type=float, default=0.25, help='ratio of augmented style during training')
        self.parser.add_argument('--centercropratio', type=float, default=0.5, help='ratio of center cropping')
        
        # model related
        self.parser.add_argument('--save_model_name', type=str, default='../save/tetgan.ckpt', help='specify the model name to save')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
class FinetuneOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--style_name', type=str, default='../data/oneshotstyle/1-train.png', help='path of the style image')

        # train related
        self.parser.add_argument('--outer_iter', type=int, default=20, help='number of iteration for fading in progressive training')
        self.parser.add_argument('--epoch', type=int, default=1, help='number of epoch for each outer iteration')      
        self.parser.add_argument('--batchsize', type=int, default=8, help='batchsize')   
        self.parser.add_argument('--datasize', type=int, default=80, help='number of sampled data for each epoch')
        self.parser.add_argument('--supervise', type=int, default=1, help='1 for supervised learning, 0 for unsupervised learning')
        
        # model related
        self.parser.add_argument('--save_model_name', type=str, default='../save/tetgan-oneshot.ckpt', help='specified the model name to save')
        self.parser.add_argument('--load_model_name', type=str, default='../save/tetgan-aaai.ckpt', help='specify the model name to load')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu, 0 for cpu, 1 for gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt