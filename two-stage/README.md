# Two-Stage Model of Self-Contained Stylization
Our two-stage model is a pipeline built upon a straight-forward integration of style transfer and steganography networks.

![](./model.png)

## How to run

### Test
```
python test.py --content_dir <content dir path> --style_dir <style dir path> --output_dir <output dir path> --cpu(if run without GPU)
```

### Train
```
python train.py --content_dir <content dir path> --style_dir <style dir path> --result_dir <image result dir path> --log_dir <log dir path for tensorboard>  --cpt_dir <model checkpoint dir path> --vgg_path <pretrained vgg path> 
```

## Note
In the pytorch 0.4, which paper implemented version, the code may have a F.mse_loss reduction mode bug. Therefore, in the paper, we set the image loss weight 2000 and message loss wieght 0.00001.
In this code default, we set the image loss weight 50 and message loss wieght 1.