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

![image](imgs/framework.png)

## Introduction

### Abstract
>Image animation aims to bring static images to life according to driving videos and create engaging visual content that can be used for various purposes such as animation, entertainment, and education. Recent unsupervised methods utilize affine and thin-plate spline transformations based on keypoints to transfer the motion in driving frames to the source image. However, limited by the expressive power of the transformations used, these methods always produce poor results when the gap between the motion in the driving frame and the source image is large. To address this issue, we propose to model motion from the source image to the driving frame in highly-expressive diffeomorphism spaces. Firstly, we introduce Continuous Piecewise-Affine based (CPAB) transformation to model the motion and present a well-designed inference algorithm to generate CPAB transformation from control keypoints. Secondly, we propose a SAM-guided keypoint semantic loss to further constrain the keypoint extraction process and improve the semantic consistency between the corresponding keypoints on the source and driving images. Finally, we design a structure alignment loss to align the structure-related features extracted from driving and generated images, thus helping the generator generate results
that are more consistent with the driving action. Extensive experiments on four datasets demonstrate the effectiveness of our method against state-of-the-art competitors quantitatively and qualitatively. 

This work has been accepted by AAAI 2024. 


### Examples of video reconstruction and image animation
![image](imgs/image_recon_comp.pdf)
![image](imgs/main_comp_new.pdf)


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
