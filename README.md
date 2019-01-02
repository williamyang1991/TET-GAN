<img src="https://github.com/williamyang1991/TET-GAN/blob/master/imgs/teaser.png" width="80%" height="80%">

# TET-GAN

This is a pytorch implementation of the paper.

Shuai Yang, Jiaying Liu, Wenjing Wang and Zongming Guo.
TET-GAN: Text Effects Transfer via Stylization and Destylization, 
Accepted by AAAI Conference on Artificial Intelligence (AAAI), 2019.

[[Project]](http://www.icst.pku.edu.cn/struct/Projects/TETGAN.html) | [[Paper]](https://arxiv.org/abs/1812.06384) | [[Dataset]](http://www.icst.pku.edu.cn/struct/Projects/TETGAN.html)

**This code currently only provides functions for testing. We are cleanning up the training code and the full code will be released soon.**

It is provided for educational/research purpose only. Please consider citing our paper if you find the software useful for your work.


## Usage: 

#### Prerequisites
- Python 2.7
- Pytorch 0.4.1
- matplotlib
- scipy
- opencv-python
- Pillow

#### Install
- Clone this repo:
```
git clone https://github.com/williamyang1991/TET-GAN.git
cd TET-GAN/src
```
## Testing Example

- Download a pre-trained model from [[Google Drive]](https://drive.google.com/file/d/1pNOE4COeoXp_-N4IogNS-GavCBbZJtw1/view?usp=sharing) or [[Baidu Cloud]](https://pan.baidu.com/s/1yK6wM0famUwu25s1v92Emw) to `./save/`
- Style Transfer with default parameters
  - Results can be found in `./output/`
  
<img src="https://github.com/williamyang1991/TET-GAN/blob/master/imgs/example.jpg" width="50%" height="50%">
  
```
python test.py
```
- Destylization with default parameters
```
python test.py --c2s 0
```
- Transfer the style of `26.jpg` onto the text image `2.png` and save the result as `26_2.png`
```
python test.py --style_name ../data/style/1.png --content_name ../data/content/2.png --name 26_1.png
```
- For black and white text images, use option `--content_type 1`
```
python test.py --style_name ../data/style/1.png --content_name ../data/content/4.png --content_type 1
```

## Training Examples

We are cleanning up the related code and it will be coming soon.

### Contact

Shuai Yang

williamyang@pku.edu.cn
