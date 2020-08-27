
import time
import os 
import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

import model

def argumnet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='../input/content/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='../input/style/',
                        help='Directory path to a batch of style images')

    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Directory path for output images')

    parser.add_argument('--vgg', type=str, default='./model_weights/vgg_normalised.pth')
    parser.add_argument('--tune_vgg', type=str, default='./model_weights/tuned_vgg.pth')
    parser.add_argument('--encrypter', type=str, default='./model_weights/encrypter.pth')
    parser.add_argument('--plain_decoder', type=str, default='./model_weights/plain_decoder.pth')
    parser.add_argument('--decrypter', type=str, default='./model_weights/decrypter.pth')
    
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    return args

def check_output_subdir_exist(output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    subdir_names = ['regular', 'reverse', 'serial']
    for dir_name in subdir_names:
        subdir_path = os.path.join(output_dir,dir_name)
        if not os.path.isdir(subdir_path):
            os.mkdir(subdir_path)



def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def main():
    args = argumnet()
    check_output_subdir_exist(args.output_dir)
    tf = test_transform(256, True) # End-to-end pretrained model only support 256x256
        
    stylization = model.SelfContained_Style_Transfer()
    stylization.load_model(vgg_path=args.vgg, tune_vgg_path=args.tune_vgg, encrypter_path=args.encrypter, plain_decoder_path=args.plain_decoder, decrypter_path=args.decrypter)
    stylization.eval()

    if not args.cpu:
        stylization = stylization.cuda()

    content_names = [f for f in os.listdir(args.content_dir)]
    style_names = [f for f in os.listdir(args.style_dir)]
    
    for content_name in content_names:
        print(content_name)
        content_image = tf(Image.open(args.content_dir + content_name)).unsqueeze(0)
        if not args.cpu:
            content_image = content_image.cuda()
        for style_name in style_names:
            print(style_name)
            style_image = tf(Image.open(args.style_dir + style_name)).unsqueeze(0)
            if not args.cpu:
                style_image = style_image.cuda()
            # style transfer
            stylized = stylization.style_transfer(content_image, style_image)

            output = stylized.cpu()
            output_name = args.output_dir + 'regular/regular-' + content_name.strip('.jpg') + '__' + style_name
            save_image(output, output_name, normalize=True)
            print(output_name)

            # reverse style transfer
            recon_img = stylization.reverse_style_transfer(stylized)

            output = recon_img.cpu()
            output_name = args.output_dir + 'reverse/reverse-' + content_name.strip('.jpg') + '__' + style_name
            save_image(output, output_name, normalize=True)
            print(output_name)
            # reverse and serial style transfer
            for serial_style_name in style_names:
                print(serial_style_name)
                serial_style_image = tf(Image.open(args.style_dir + serial_style_name)).unsqueeze(0)
                if not args.cpu:
                    serial_style_image = serial_style_image.cuda()
                serial_stylized = stylization.serial_style_tansfer(stylized, serial_style_image)

                output = serial_stylized.cpu()
                output_name = args.output_dir + 'serial/serial-' + content_name.strip('.jpg') + '__' + style_name.strip('.jpg') + '__' + serial_style_name
                save_image(output, output_name, normalize=True)
                print(output_name)

if __name__ == "__main__":
    with torch.no_grad():
        main()