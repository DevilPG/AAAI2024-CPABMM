# Continuous Piecewise-Affine Based Motion Model for Image Animation (AAAI 2024)

###  [Paper](https://doi.org/10.48550/arXiv.2401.09146 )
<!-- <br> -->
[Hexiang Wang](https://github.com/DevilPG), 
[Fengqi Liu](liufengqi@sjtu.edu.cn), 
[Qianyu Zhou](https://qianyuzqy.github.io/),
[Ran Yi](https://yiranran.github.io/), 
[Xin Tan](https://tanxincs.github.io/), 
 and [Lizhuang Ma](https://dmcv.sjtu.edu.cn/) 
<!-- <br> -->

![image](imgs/framework.jpg)

## Introduction

### Abstract
>Stroke-based rendering aims to recreate an image with a set of strokes. Most existing methods render complex images using an uniform-block-dividing strategy, which leads to boundary inconsistency artifacts. To solve the problem, we propose Compositional Neural Painter, a novel stroke-based rendering framework which dynamically predicts the next painting region based on the current canvas, instead of dividing the image plane uniformly into painting regions. We start from an empty canvas and divide the painting process into several steps. At each step, a compositor network trained with a phasic RL strategy first predicts the next painting region, then a painter network trained with a WGAN discriminator predicts stroke parameters, and a stroke renderer paints the strokes onto the painting region of the current canvas. Moreover, we extend our method to stroke-based style transfer with a novel differentiable distance transform loss, which helps preserve the structure of the input image during stroke-based stylization. Extensive experiments show our model outperforms the existing models in both stroke-based neural painting and stroke-based stylization.

This work has been accepted by ACM MM 2023. 

### The Boundary Inconsistency Problem in Existing Works
![image](imgs/boundary%20inconsistency.jpg)

### Demos
<div class="half">
    <img src="imgs/1.gif" width="180"><img src="imgs/2.gif" width="180"><img src="imgs/3.gif" width="180"><img src="imgs/4.gif" width="180">
</div>
<div class="half">
    <img src="imgs/5.gif" width="180"><img src="imgs/6.gif" width="180"><img src="imgs/7.gif" width="180"><img src="imgs/8.gif" width="180">
</div>

## Todo (Latest update: 2023/10/16)
- [x] **Release the training code
- [x] **Release the pretrained model

## Pretrained Models
The pretrained models can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1MFa7RNEREvn8nsLWmwmYpy_EUGpGbztt?usp=sharing) and [百度网盘](https://pan.baidu.com/s/11to0MHjXqzxJU89rqFAHiA) (提取码：0306)

You can put painter.pkl under painter/checkpoints, compositor.pkl under compositor/checkpoints and renderer-oil.pkl under the main folder.


## Training Step

### (0) Prepare
Data prepare: download the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset or [ImageNet](https://image-net.org) dataset

Dependencies:
```
pip install torch==1.6.0
```

### (1) Train the stroke renderer


To train the neural stroke renderer, you can run the following code
```
python3 compositor/train_renderer_FCN.py
```

### (2) Train the painter network

After the neural renderer is trained, you can train the painter network by:
```
$ cd painter
$ python3 train.py
```

### (3) Train the compositor network

With the trained stroke renderer and painter network, you can train the compositor network by:
```
$ cd compositor
$ python3 train.py --dataset=path_to_your_dataset
```

## Testing Step

### Image to painting
After all the training steps are finished, you can paint an image by:
```
$ cd compositor
$ python3 test.py --img_path=path_to_your_test_img --stroke_num=number_of_strokes
```

If you want to save the painting process as a video, you can use ```--video```.

If the painted results are not so good, you can try different painting mode by ```--mode=2``` or ```--mode=3```.

### Test the DT loss

We provide the differentiable distance transform code in compositor/DT.py, you can have a test by:

```
$ cd compositor
$ python3 DT.py --img_path=path_to_your_test_img
```

## Citation

If you find this code helpful for your research, please cite:

```
@inproceedings{hu2023stroke,
      title={Stroke-based Neural Painting and Stylization with Dynamically Predicted Painting Region}, 
      author={Teng Hu and Ran Yi and Haokun Zhu and Liang Liu and Jinlong Peng and Yabiao Wang and Chengjie Wang and Lizhuang Ma},
      booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
      year={2023}
      }
```
