import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

def Encrypt_Decoder():
    net = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(1536, 1024, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(1024, 1024, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(1024, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),    
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )

    return net

def Plain_Decoder():
    net = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    return net

def VGG():
    full_vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    return full_vgg

def Decrypter():
    net = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 1024, (3, 3)),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    )
    return net

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def inverse_AdaIN(content_feat, content_std, content_mean):
    size = content_feat.size()
    cur_mean, cur_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - cur_mean.expand(size)) / cur_std.expand(size)
    return normalized_content_feat * content_std + content_mean

def encrypted_decoding(decoder, feat, en_std, en_mean):        
    # add the layers of content std and mean info
    std_layer = en_std.expand(feat.size())
    mean_layer = en_mean.expand(feat.size())
    feat_cat = torch.cat((feat, std_layer, mean_layer), 1)

    # decode the image
    g_t = decoder(feat_cat)

    return g_t

class Feature_extractor(nn.Module):
    def __init__(self, encoder, tune):
        super(Feature_extractor, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = tune

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input    

    def forward(self, image):
        style_feats = self.encode_with_intermediate(image)
        content_feat = self.encode(image)
        return content_feat, style_feats

# loss computation functions

# import function

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)
    
def calc_content_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)




class SelfContained_Style_Transfer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encrypter = Encrypt_Decoder()
        self.plain_decoder = Plain_Decoder()
        self.decrypter = Decrypter()

    def load_model(self, vgg_path=None, tune_vgg_path=None, encrypter_path=None, plain_decoder_path=None, decrypter_path=None):
        '''
        method for loading pretrained model weight
        '''
        if encrypter_path:
            self.encrypter.load_state_dict(torch.load(encrypter_path))
        if plain_decoder_path:
            self.plain_decoder.load_state_dict(torch.load(plain_decoder_path))
        if decrypter_path:
            self.decrypter.load_state_dict(torch.load(decrypter_path))

        vgg = VGG().eval()
        if vgg_path:
            vgg.load_state_dict(torch.load(vgg_path))
            vgg = nn.Sequential(*list(vgg.children())[:31])
        self.fixed_feat_extractor = Feature_extractor(vgg, False)

        vgg_tune = VGG().eval()
        if tune_vgg_path:
            vgg_tune = nn.Sequential(*list(vgg_tune.children())[:31])
            vgg_tune.load_state_dict(torch.load(tune_vgg_path))
        else:
            vgg_tune.load_state_dict(torch.load(vgg_path))
            vgg_tune = nn.Sequential(*list(vgg_tune.children())[:31])
        self.tune_feat_extractor = Feature_extractor(vgg_tune, True)

    def style_transfer(self,content, style):
        content_feat = self.fixed_feat_extractor.encode(content)
        style_feat = self.fixed_feat_extractor.encode(style)
        c_mean, c_std = self.calc_mean_std(content_feat)

        feat = self.adaptive_instance_normalization(content_feat, style_feat)
        self_contained_stylized = self.encrypted_decoding(feat, c_mean, c_std)
        return self_contained_stylized

    def reverse_style_transfer(self, stylized):
        recon_feat = self.reconturct_content_feature(stylized)
        recon_img = self.plain_decoder(recon_feat)
        return recon_img

    def serial_style_tansfer(self, stylized, serial_style):
        serial_style_feat = self.fixed_feat_extractor.encode(serial_style)
        recon_feat, decry_c_mean, decry_c_std = self.reconturct_content_feature(stylized, return_mean_std=True)
        serial_feat = adaptive_instance_normalization(recon_feat, serial_style_feat)
        serial_stylized = self.encrypted_decoding(serial_feat, decry_c_mean, decry_c_std)
        return serial_stylized

    def reverse_and_serial_style_transfer(self,stylized, serial_style):
        recon_feat, decry_c_mean, decry_c_std = self.reconturct_content_feature(stylized, return_mean_std=True)
        #reverse
        recon_feat = self.reconturct_content_feature(stylized)
        recon_img = self.plain_decoder(recon_feat)
        #serial
        serial_style_feat = self.fixed_feat_extractor.encode(serial_style)
        serial_feat = adaptive_instance_normalization(recon_feat, serial_style_feat)
        serial_stylized = self.encrypted_decoding(serial_feat, decry_c_mean, decry_c_std)

        return recon_img, serial_stylized

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def inverse_AdaIN(self,content_feat, content_std, content_mean):
        size = content_feat.size()
        cur_mean, cur_std = calc_mean_std(content_feat)
        normalized_content_feat = (content_feat - cur_mean.expand(size)) / cur_std.expand(size)
        return normalized_content_feat * content_std + content_mean

    def encrypted_decoding(self, feat, en_mean, en_std):        
        # add the layers of content std and mean info
        std_layer = en_std.expand(feat.size())
        mean_layer = en_mean.expand(feat.size())
        feat_cat = torch.cat((feat, std_layer, mean_layer), 1)

        # decode the image
        g_t = self.encrypter(feat_cat)

        return g_t

    def reconturct_content_feature(self, stylized, return_mean_std=False):
        stylized_content_feat = self.tune_feat_extractor.encode(stylized)
        decry_info = self.decrypter(stylized)
        decry_c_std = decry_info[:, 0:512, :, :]
        decry_c_mean = decry_info[:, 512:1024, :, :]
        recon_feat = self.inverse_AdaIN(stylized_content_feat, decry_c_std, decry_c_mean)
        if return_mean_std:
            return recon_feat, decry_c_mean, decry_c_std
        else:
            return recon_feat