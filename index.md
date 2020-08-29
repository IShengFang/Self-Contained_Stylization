
 Hung-Yu Chen†[1] and [I-Sheng Fang†](https://ishengfang.github.io)[1] and Chia-Ming Cheng[2] and [Wei-Chen Chiu†](https://walonchiu.github.io)[1]

 [1]National Chiao Tung University, Taiwan, [2] MediaTek Inc., Taiwan
 
† Both authors contribute equally.

---------------------------------------
[[arxiv]](https://arxiv.org/pdf/1812.03910.pdf)[[WACV2020]](https://openaccess.thecvf.com/content_WACV_2020/html/Chen_Self-Contained_Stylization_via_Steganography_for_Reverse_and_Serial_Style_Transfer_WACV_2020_paper.html)[[poster]](./poster.pdf)

![](https://github.com/IShengFang/Self-Contained_Stylization/raw/master/teaser.png)

## Abstract 
Style transfer has been widely applied to give real-world images a new artistic look. However, given a stylized image, the attempts to use typical style transfer methods for de-stylization or transferring it again into another style usually lead to artifacts or undesired results. We realize that these issues are originated from the content inconsistency between the original image and its stylized output. Therefore, in this paper we advance to keep the content information of the input image during the process of style transfer by the power of steganography, with two approaches proposed: a two-stage model and an end-to-end model. We conduct extensive experiments to successfully verify the capacity of our models, in which both of them are able to not only generate stylized images of quality comparable with the ones produced by typical style transfer methods, but also effectively eliminate the artifacts introduced in reconstructing original input from a stylized image as well as performing multiple times of style transfer in series. 

## Introduction

![](./intro.gif)
Typical style transfer approach aims to transfer the style of the content photo according to the one of the reference style image. However, the problem of reverse style transfer for removing the stylization of a stylized image to recover the original content photo is rarely discussed. Besides, no prior work has discussed the problem of serial style transfer which transfers a stylized image into another style but preserves the content of original image. None of the existing methods can easily tackle these two problems, where the original photo and its stylized image have *content inconsistency* due to the stylization.

## Method
To resolve the *content inconsistency*, we propose to integrate the idea of **steganography** with **style transfer**. We call it **Self-Contained Stylization**, where the content information of the original photo is hidden into the stylized output. We have two methods proposed in our paper, Two-stage model and end-to-end model.

### Two-Stage Model
Our two-stage model is a pipeline built upon a straightforward integration of style transfer and steganography networks, as shown in Figure 3(a). In the first stage, we stylize the content image according to the style image based on a style transfer model. Afterward in the second stage, the steganography network learns an encoder to hide the content information of into the stylized image It from the previous stage, as well as a paired decoder which is able to retrieve the hidden information from the encoded image.

![](https://github.com/IShengFang/Self-Contained_Stylization/raw/master/two-stage/model.png)

### End-to-End Model

Aside from the two-stage model which can take several style transfer methods as its base (please refer to our supplementary material), our end-to-end model digs deeply into the characteristic of AdaIN for enabling image stylization and content information encryption simultaneously in a single network.
![](https://github.com/IShengFang/Self-Contained_Stylization/raw/master/end-to-end/model.png)

## Results
We compare our proposed models to the baselines from Gatys *et al.* and AdaIN with qualitative and quantitative evaluations.

### Qualitative Evaluation
![](https://github.com/IShengFang/Self-Contained_Stylization/raw/master/result.gif)
In regular style transfer, given content and style images, our model can well perform typical style transfer with the content feature seamlessly hidden into the resultant stylized images. 

In reverse style transfer, we are able to easily reconstruct the original photos solely based on these encoded stylized images. Although the results of our two-stageand end-to-end models have some mild color patches andslight color shift respectively, they both well reconstruct theoverall structure of the content image. Please note that our proposed methods **do not need the original image** to achieve reverse style transfer.

In serial style transfer, our proposed methods are more similar to their respective expectations than the ones from baselines whichare deeply influenced by the previous stylization. Our model can perform *multiple style transfers* in series without requiring the original content photo.



### Quantitative Evaluation
![](https://github.com/IShengFang/Self-Contained_Stylization/raw/master/quant.png)

We perform reverse and serial style transfer with different models and comparethe outputs with respect to their corresponding expectations. The averaged L2 distance, structural similarity (SSIM), and learned perceptual image patch similarity (LPIPS) are used to measure the difference and the results are shown in Table 2. 

Both our models perform better than the base-lines. Particularly, our two-stage model performs the bestfor reverse style transfer while the end-to-end model doesso for serial style transfer. We believe that our two stage model benefits from its larger amount of encrypted information and the design of identity mapping, leading to the better result in reverse style transfer, and the end-to-end modelshows its advantage in having less information to hide, making it more robust to the propagated error caused by serialstyle transfer.
## Citation
```
@inproceedings{chen20wacv,
 title = {Self-Contained Stylization via Steganography for Reverse and Serial Style Transfer},
 author = {Hung-Yu Chen and I-Sheng Fang and Chia-Ming Cheng and Wei-Chen Chiu},
 booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
 year = {2020}
} 
```

## Acknowledgements
Part of the code is based on [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

This project is supported by MediaTek Inc., MOST-108-2636-E-009-001, MOST-108- 2634-F-009 -007, and MOST-108-2634-F-009-013. We are grateful to the National Center for Highperformance Computing for computer time and facilities.
