import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#vgg definition that conveniently let's you grab the outputs from any layer
def VGG():
    vgg = nn.Sequential(
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
    return vgg

    
        
def Decoder():
    decoder = nn.Sequential( # Sequential,
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512,256,(3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256,256,(3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256,256,(3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256,256,(3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256,128,(3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128,128,(3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128,64,(3, 3)),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64,64,(3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64,3,(3, 3)),
    )
    return decoder
    
    
class VGG_Encoder(nn.Module):
    def __init__(self, pretrain_nets):
        super(VGG_Encoder, self).__init__()
        enc_layers = list(pretrain_nets.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

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

    def forward(self, style_img, content_img=None, with_intermediate=True):
        if with_intermediate:
            feats = self.encode_with_intermediate(style_img)
            return feats[-1], feats
        else:
            content_feat = self.encode(content_img)
            style_feats = self.encode_with_intermediate(style_img)
            return content_feat, style_feats
    
class Message_Encoder(nn.Module):
    def __init__(self, input_channel=3, input_width=256, kernel_size=3, encode_depth=3, messenge_channel=8, dim=64):
        super(Message_Encoder, self).__init__()
        
        self.conv_in =nn.Conv2d(input_channel, dim, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
    
        self.conv_in_bn = nn.BatchNorm2d(dim)
        
        self.conv_en = nn.Sequential()
        self.conv_en.add_module('conv_en_0', nn.Conv2d(dim+messenge_channel, dim, kernel_size, stride=1, padding=1, dilation=1, bias=True))
        self.conv_en.add_module('conv_en_0_bn', nn.BatchNorm2d(dim))
        self.conv_en.add_module('conv_en_0_relu', nn.ReLU())
        for i in range(encode_depth-1):
            self.conv_en.add_module('conv_en_{}'.format(i+1), nn.Conv2d(dim, dim, kernel_size, stride=1, padding=1, dilation=1, bias=True))
            self.conv_en.add_module('conv_en_{}_bn'.format(i+1), nn.BatchNorm2d(dim))
            self.conv_en.add_module('conv_en_{}_relu'.format(i+1), nn.ReLU())
        
        for layer in self.conv_en:
            try:
                nn.init.zeros_(layer.bias)
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
            
        self.conv_out = nn.Conv2d(dim+3, 3, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
        self.conv_out_bn = nn.BatchNorm2d(3)
        
    def forward(self, x, messenge):
        h = F.relu(self.conv_in_bn(self.conv_in(x)))
        h = torch.cat((h, messenge), dim=1)
        h = self.conv_en(h)
        h = torch.cat((h,x),dim=1)
        x_encoded = F.relu(self.conv_out_bn(self.conv_out(h)))
        
        return x_encoded
    
class Message_Decoder(nn.Module):
    def __init__(self, input_channel=3, input_width=256, kernel_size=3, decode_depth=6, dim=64,
                 messenge_channel=8, content_feat_shape=(512,32,32)):
        super(Message_Decoder, self).__init__()
        
        self.input_width= input_width
        
        self.conv_in =nn.Conv2d(input_channel, dim, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv_in_bn = nn.BatchNorm2d(dim)
        
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        
        self.conv_de = nn.Sequential()
        for i in range(decode_depth):
            self.conv_de.add_module('conv_de_{}'.format(i), nn.Conv2d(dim, dim, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True))
            self.conv_de.add_module('conv_de_{}_bn'.format(i), nn.BatchNorm2d(dim))
            self.conv_de.add_module('conv_de_{}_relu'.format(i), nn.ReLU())

        for layer in self.conv_de:
            try:
                nn.init.zeros_(layer.bias)
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
            
        self.conv_out = nn.Conv2d(dim, messenge_channel, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
                                  
        self.conv_out_bn = nn.BatchNorm2d(messenge_channel)
        self.content_feat_shape = content_feat_shape
        self.C, self.H, self.W = content_feat_shape
                
    def forward(self, x, use_train_content_feat_shape=True, content_feat_shape=None):
        h = F.relu(self.conv_in_bn(self.conv_in(x)))
        h = self.conv_de(h)
        h = F.relu(self.conv_out_bn(self.conv_out(h)))
        h = F.adaptive_avg_pool2d(h,(self.input_width, self.input_width))       
        if use_train_content_feat_shape:
            h = h.view(x.size(0), self.C, self.H, self.W )
        elif content_feat_shape:
            h = h.view(message_shape)
        return h    


class SelfContained_Style_Transfer(nn.Module):
    vgg_relu_4_1_dim = 512
    messenge_channel = 8
    down_time = 3
    down_scale = 2**down_time

    def __init__(self):
        super(SelfContained_Style_Transfer, self).__init__()
    
    def load_model(self, vgg_path, decoder_path, message_encoder_path, message_decoder_path, image_size, ):
        
        self.image_size = image_size

        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.decoder = Decoder()
        self.decoder.load_state_dict(torch.load(decoder_path))

        self.message_encoder = Message_Encoder()
        self.message_encoder.load_state_dict(torch.load(message_encoder_path))

        self.message_decoder = Message_Decoder(
            input_width = self.image_size, 
            content_feat_shape=( self.vgg_relu_4_1_dim, int(self.image_size/self.down_scale), int(self.image_size/self.down_scale)))
        self.message_decoder.load_state_dict(torch.load(message_decoder_path))

    def style_transfer_stage(self,content, style):
        content_feat = self.vgg(content)
        style_feat = self.vgg(style)
        feat = self.adaptive_instance_normalization(content_feat, style_feat)
        stylized = self.decoder(feat)
        return F.hardtanh(stylized, 0,1), content_feat

    def steganography_stage(self,image, msg):
        msg = torch.reshape(msg, (msg.size(0), self.messenge_channel, self.image_size,self.image_size))
        self_contained_stylized = self.message_encoder(image, msg)
        return F.hardtanh(self_contained_stylized, 0,1)

    def style_transfer(self,content, style):
        stylized, msg = self.style_transfer_stage(content, style)
        self_contained_stylized = self.steganography_stage(stylized, msg)
        return self_contained_stylized

    def reverse_style_transfer(self, encoded, return_msg=False):
        recon_feat = self.message_decoder(encoded)
        recon_img = F.hardtanh(self.decoder(recon_feat),0,1)
        if return_msg:
            return recon_img, recon_feat
        else:
            return recon_img

    def serial_style_tansfer(self, encoded, serial_style, keep_hidding=False):
        recon_feat = self.message_decoder(encoded)
        style_feat = self.vgg(serial_style)
        feat = self.adaptive_instance_normalization(recon_feat, style_feat)
        serial_stylized = F.hardtanh(self.decoder(feat), 0,1)
        if keep_hidding:
            return self.steganography_stage(serial_stylized, recon_feat)
        else:
            return serial_stylized

    def serial_style_tansfer_w_msg(self, msg, serial_style, keep_hidding=False):
        serial_style_feat = self.vgg(serial_style)
        feat = self.adaptive_instance_normalization(msg, serial_style_feat)
        serial_stylized = F.hardtanh(self.decoder(feat), 0,1)
        if keep_hidding:
            return self.steganography_stage(serial_stylized, msg)
        else:
            return serial_stylized

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

    
