import time
import os 
import argparse

import model
import dataset
import utils

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    #for older versioin pytorch which w/o torch.utils.tensorboard.SummaryWriter
    from tensorboardX import SummaryWriter

from torch import optim
import copy

import torchvision
from torchvision import transforms

from tqdm import tqdm


from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

from collections import OrderedDict

class StyleTransferStageTrainer(nn.Module):
    def __init__(self,train_identity=True):
        super().__init__()
        self.n_iter = 0
        self.train_identity = train_identity
    
    def init_tensorboard(self,writer):
        self.writer = writer

    def load_model(self, vgg_path, decoder_path=None):
        self.vgg = model.VGG()
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.vgg_encoder = model.VGG_Encoder(self.vgg)

        self.decoder = model.Decoder()
        if decoder_path:
            self.decoder.load_state_dict(torch.load(decoder_path))

    def init_optim(self, lr, lr_decay, beta_1, beta_2):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.opt = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(beta_1, beta_2))

    def init_train_data(self,content_dir, style_dir, batch_size, num_workers, identity_radio,image_size):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.content_dir = content_dir
        self.style_dir = style_dir

        if self.train_identity:
            pair_dataset = dataset.FlatFolderDatasetPair(args.content_dir, args.style_dir, 
                                                dataset.train_transform(image_size), identity_radio=5)
            self.pair_iter = iter(data.DataLoader(
                            pair_dataset, batch_size=args.batch_size,
                            sampler=dataset.InfiniteSamplerWrapper(pair_dataset),
                            num_workers=num_workers))
        else:
            content_tf = dataset.train_transform(image_size)
            style_tf = dataset.train_transform(image_size)
            content_dataset = dataset.FlatFolderDataset(content_dir, content_tf)
            style_dataset = dataset.FlatFolderDataset(style_dir, style_tf)
            self.content_iter = iter(data.DataLoader(content_dataset, batch_size=self.batch_size,
                                sampler=dataset.InfiniteSamplerWrapper(content_dataset),
                                num_workers=self.num_workers))
            self.style_iter = iter(data.DataLoader(style_dataset, batch_size=self.batch_size,
                            sampler=dataset.InfiniteSamplerWrapper(style_dataset),
                            num_workers=self.num_workers))

    def init_test_data(self, test_content_dir, test_style_dir, test_size=256, crop=True):
        self.test_content_tf = dataset.test_transform(test_size, crop)
        self.test_style_tf = dataset.test_transform(test_size, crop)
        test_content_dataset = dataset.FlatFolderDataset(test_content_dir, self.test_content_tf)
        test_style_dataset = dataset.FlatFolderDataset(test_style_dir, self.test_style_tf )
        self.test_content = next(iter(data.DataLoader(test_content_dataset, batch_size=self.batch_size,
                                      sampler=dataset.InfiniteSamplerWrapper(test_content_dataset),
                                      num_workers=self.num_workers)))
        self.test_style = next(iter(data.DataLoader(test_style_dataset, batch_size=self.batch_size,
                                    sampler=dataset.InfiniteSamplerWrapper(test_style_dataset),
                                    num_workers=self.num_workers)))

        with torch.no_grad():
            # get the relu4_1 feature 
            test_content_feat = self.vgg_encoder.encode(self.test_content.cuda())
            test_style_feat = self.vgg_encoder.encode(self.test_style.cuda())
            # transfer the style
            self.test_stylized_feat = self.adaptive_instance_normalization(test_content_feat, test_style_feat).cpu()
            self.test_content_feat = test_content_feat.cpu()

        output = self.test_content.cpu()
        output_name = self.result_dir + 'content' + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('first_stage/content', output, self.n_iter)
        # save the style image
        output = self.test_style.cpu()
        output_name = self.result_dir + 'style' '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('first_stage/style', output, self.n_iter)

    def init_loss_weight(self, content_weight=1.0, style_weight=10.0):
        self.content_weight = content_weight
        self.style_weight = style_weight

    def init_cpt(self, cpt_interval, cpt_dir, result_dir,):
        self.cpt_interval = cpt_interval
        self.cpt_dir = cpt_dir
        utils.check_and_make_dir(self.cpt_dir)
        self.result_dir = os.path.join(result_dir, 'first_stage')
        utils.check_and_make_dir(self.result_dir)

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

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return F.mse_loss(input, target)

    def adjust_learning_rate(self, optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = self.lr / (1.0 + self.lr_decay * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _fit_step(self,):
        # adjust the learning rate
        self.adjust_learning_rate(self.opt, iteration_count=self.n_iter)
        # get the images
        if self.train_identity:
            content_images, style_images= next(self.pair_iter)
            content_images = content_images.cuda()
            style_images = style_images.cuda()
        else:
            content_images = next(self.content_iter).cuda()
            style_images = next(self.style_iter).cuda()
        
        # get the relu4_1 feature 
        content_feat, style_feats = self.vgg_encoder(style_images, content_images, with_intermediate=False)
        # transfer the style
        stylized_feat = self.adaptive_instance_normalization(content_feat, style_feats[-1])
        stylized_image  = self.decoder(stylized_feat)

        content_feat_re, style_feats_re = self.vgg_encoder(stylized_image)

        # compute the style transfer losses
        loss_c = self.calc_content_loss(content_feat_re, content_feat)
        loss_s = self.calc_style_loss(style_feats_re[0], style_feats[0])
        for s_layer in range(1, 4):
            loss_s += self.calc_style_loss(style_feats_re[s_layer], style_feats[s_layer])
        
        # multiply the losses by weights
        loss_c = self.content_weight * loss_c
        loss_s = self.style_weight * loss_s
        
        total_loss = loss_c + loss_s 
        self.opt.zero_grad()
        total_loss.backward()
        
        self.opt.step()

        self.n_iter+=1

        #tensorboard
        self.writer.add_scalar('first_stage/loss_content', loss_c, self.n_iter)
        self.writer.add_scalar('first_stage/loss_style', loss_s, self.n_iter)
        self.writer.add_scalar('first_stage/total_loss', total_loss, self.n_iter)

        self.loss_c = loss_c.item()
        self.loss_s = loss_s.item()
        self.total_loss = total_loss.item()

    def _save_cpt(self,):
        # save the decoder parameters
        state_dict = self.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}AdaIN_decoder{:d}.pth'.format(self.cpt_dir, self.n_iter))
        
    def _test(self,):
        # set the mode of the models
        self.decoder.eval()

        stylized_image  = self.decoder(self.test_stylized_feat.cuda())
        # save the stylized image
        output = stylized_image.cpu()
        output_name = self.result_dir + 'stylized_' + str(self.n_iter) + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('first_stage/stylized', output, self.n_iter)
        
        # content feature reconstruction
        recon_img = self.decoder(self.test_content_feat.cuda())
        output = recon_img.cpu()
        output_name =  self.result_dir + 'recon_' + str(self.n_iter) + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('first_stage/reconstructed', output, self.n_iter)
        # set the mode of the models
        self.decoder.train()

    def fit(self,max_iter):
        self.max_iter = max_iter
        pbar = tqdm(range(self.max_iter))
        for n_iter in pbar:
            self._fit_step()
            pbar.set_description('c:%4.2f|s:%4.2f|total:%4.2f'
                %(self.loss_c, self.loss_s, self.total_loss))
            if self.n_iter % self.cpt_interval == 0 or self.n_iter == max_iter:
                self._save_cpt()
                with torch.no_grad():
                    self._test()

class SteganographyStageTrainer(model.SelfContained_Style_Transfer):
    def __init__(self,):
        super().__init__()
        self.n_iter = 0
    def init_tensorboard(self,writer):
        self.writer = writer

    def load_model(self, vgg_path=None, decoder_path=None, message_encoder_path=None, message_decoder_path=None):
        self.vgg = model.VGG()
        if vgg_path:
            self.vgg.load_state_dict(torch.load(vgg_path))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.decoder = model.Decoder()
        if decoder_path:
            self.decoder.load_state_dict(torch.load(decoder_path))

        self.message_encoder = model.Message_Encoder()
        if message_encoder_path:
            self.message_encoder.load_state_dict(torch.load(message_encoder_path))

        self.message_decoder = model.Message_Decoder(
            input_width = self.image_size, 
            content_feat_shape=(self.vgg_relu_4_1_dim, int(self.image_size/self.down_scale), int(self.image_size/self.down_scale)))
        if message_decoder_path:
            self.message_decoder.load_state_dict(torch.load(message_decoder_path))

    def init_optim(self, lr, lr_decay, beta_1, beta_2):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.opt = torch.optim.Adam(list(self.message_encoder.parameters())+list(self.message_decoder.parameters()), lr=lr, betas=(beta_1, beta_2))

    def init_train_data(self,content_dir, style_dir, batch_size, num_workers,image_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.content_dir = content_dir
        self.style_dir = style_dir

        content_tf = dataset.train_transform(self.image_size)
        style_tf = dataset.train_transform(self.image_size)
        content_dataset = dataset.FlatFolderDataset(content_dir, content_tf)
        style_dataset = dataset.FlatFolderDataset(style_dir, style_tf)
        self.content_iter = iter(data.DataLoader(content_dataset, batch_size=self.batch_size,
                            sampler=dataset.InfiniteSamplerWrapper(content_dataset),
                            num_workers=self.num_workers))
        self.style_iter = iter(data.DataLoader(style_dataset, batch_size=self.batch_size,
                        sampler=dataset.InfiniteSamplerWrapper(style_dataset),
                        num_workers=self.num_workers))

    def init_test_data(self, test_content_dir, test_style_dir, test_size=256, crop=True):
        self.test_content_tf = dataset.test_transform(test_size, crop)
        self.test_style_tf = dataset.test_transform(test_size, crop)
        test_content_dataset = dataset.FlatFolderDataset(test_content_dir, self.test_content_tf)
        test_style_dataset = dataset.FlatFolderDataset(test_style_dir, self.test_style_tf )
        self.test_content = next(iter(data.DataLoader(test_content_dataset, batch_size=self.batch_size,
                                      sampler=dataset.InfiniteSamplerWrapper(test_content_dataset),
                                      num_workers=self.num_workers)))
        self.test_style = next(iter(data.DataLoader(test_style_dataset, batch_size=self.batch_size,
                                    sampler=dataset.InfiniteSamplerWrapper(test_style_dataset),
                                    num_workers=self.num_workers)))

        with torch.no_grad():
            # generate the stylized image
            self.test_stylized, self.test_content_feat = self.style_transfer_stage(self.test_content.cuda(), self.test_style.cuda())
            self.test_stylized = self.test_stylized.cpu()
            self.test_content_feat = self.test_content_feat.cpu()

        # save the content image
        output = self.test_content.cpu()
        output_name = self.result_dir + 'content' + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('second_stage/content', output, self.n_iter)
        # save the style image
        output = self.test_style.cpu()
        output_name = self.result_dir + 'style' '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('second_stage/style', output, self.n_iter)
        # save the stylized image
        output = self.test_stylized.cpu()
        output_name = self.result_dir + 'stylized'+ '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('second_stage/stylized', output, self.n_iter)

    def init_loss_weight(self, img_weight=1.0, msg_weight=1.0):
        self.img_weight = img_weight
        self.msg_weight = msg_weight

    def init_cpt(self, cpt_interval, cpt_dir, result_dir,):
        self.cpt_interval = cpt_interval
        self.cpt_dir = cpt_dir
        utils.check_and_make_dir(self.cpt_dir)
        self.result_dir = os.path.join(result_dir, 'second_stage')
        utils.check_and_make_dir(self.result_dir)

    def adjust_learning_rate(self, optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = self.lr / (1.0 + self.lr_decay * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _fit_step(self,):
        # adjust the learning rate
        self.adjust_learning_rate(self.opt, iteration_count=self.n_iter)
        # get the images
        content_images = next(self.content_iter).cuda()
        style_images = next(self.style_iter).cuda()
        
        #style_transfer_stage
        with torch.no_grad():
            #save gpu mem and make it faster
            stylized, content_feat = self.style_transfer_stage(content_images, style_images)
        
        #encode content feature in image
        encoded = self.steganography_stage(stylized, content_feat)
        
        # compute the image recontruction losses
        loss_img = F.mse_loss(encoded, stylized)

        # decode image
        recon_feat = self.message_decoder(encoded)
        # compute the message recontruction losses
        loss_msg = F.mse_loss(recon_feat, content_feat)

        # multiply the losses by weights
        loss_img = self.img_weight * loss_img
        loss_msg = self.msg_weight * loss_msg
        
        total_loss = loss_img + loss_msg 

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        self.n_iter+=1

        #tensorboard
        self.writer.add_scalar('second_stage/loss_img', loss_img, self.n_iter)
        self.writer.add_scalar('second_stage/loss_msg', loss_msg, self.n_iter)
        self.writer.add_scalar('second_stage/total_loss', total_loss, self.n_iter)

        self.loss_img = loss_img.item()
        self.loss_msg = loss_msg.item()
        self.total_loss = total_loss.item()

    def _save_cpt(self,):
        # save the msg encoder parameters
        state_dict = self.message_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}msg_encoder{:d}.pth'.format(self.cpt_dir, self.n_iter))
        # save the msg decoder parameters
        state_dict = self.message_decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}msg_decoder{:d}.pth'.format(self.cpt_dir, self.n_iter))  
         
    def _test(self,):
        # set the mode of the models
        self.message_encoder.eval()
        self.message_decoder.eval()

        # generate the encoded image
        encoded = self.steganography_stage(self.test_stylized.cuda(), self.test_content_feat.cuda())
        # reverse style transfer
        recon_img = self.reverse_style_transfer(encoded)
        
        # save the encoded image
        output = encoded.cpu()
        output_name =  self.result_dir + 'encoded_' + str(self.n_iter) + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('second_stage/encoded', output, self.n_iter)
        
        # save the recontructed image
        output = recon_img.cpu()
        output_name =  self.result_dir + 'reconst_' + str(self.n_iter) + '.png'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('second_stage/reconstructed', output, self.n_iter)


        self.writer.add_scalar('second_stage_eval/reverse_l1', F.l1_loss(recon_img, self.test_content.cuda()), self.n_iter)
        self.writer.add_scalar('second_stage_eval/reverse_l2', F.mse_loss(recon_img, self.test_content.cuda()), self.n_iter)
        # set the mode of the models
        self.message_encoder.train()
        self.message_decoder.train()
    def fit(self,max_iter):
        self.max_iter = max_iter
        pbar = tqdm(range(self.max_iter))
        for n_iter in pbar:
            self._fit_step()
            pbar.set_description('L_img:%4.2f|L_msg:%4.2f|total:%4.2f'
                %(self.loss_img, self.loss_msg, self.total_loss))
            if self.n_iter % self.cpt_interval == 0 or self.n_iter == max_iter:
                self._save_cpt()
                with torch.no_grad():
                    self._test()
def argument():
    parser = argparse.ArgumentParser()
    #data
    parser.add_argument('--content_dir', type=str, default='/home/ishengfang/hdd/mscoco2017/train2017/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='/home/ishengfang/hdd/painter_by_numbers/train/',
                        help='Directory path to a batch of style images')

    parser.add_argument('--test_content_dir', type=str, default='/home/ishengfang/hdd/mscoco2017/test2017/',
                        help='Directory path to a test content images')
    parser.add_argument('--test_style_dir', type=str, default='/home/ishengfang/hdd/painter_by_numbers/train/',
                        help='Directory path to a test style images')

    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='Directory path for output images')

    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Directory path for tensorboard logs')
    parser.add_argument('--cpt_dir', type=str, default='/home/ishengfang/hdd/Self-Contained_Stylization/cpts/2st/',
                        help='Directory path for model checkpoints')

    parser.add_argument('--vgg_path', type=str, default='./model_weights/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--msg_encoder_path', type=str, default=None)
    parser.add_argument('--msg_decoder_path', type=str, default=None)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8,)
    parser.add_argument('--num_workers', type=int, default=32,)

    parser.add_argument('--first_stage_iter', type=int, default=160000,)
    parser.add_argument('--second_stage_iter', type=int, default=160000,)
    parser.add_argument('--cpt_iter', type=int, default=1000,)
    parser.add_argument('--lr', type=float, default=1e-4 )
    parser.add_argument('--beta_1', type=float, default=0.9, )
    parser.add_argument('--beta_2', type=float, default=0.999, )
    parser.add_argument('--lr_decay', type=float, default=5e-5 )

    #loss weight
    parser.add_argument('--content_weight', type=float, default=1.0 )
    parser.add_argument('--style_weight', type=float, default=10.0 )

    parser.add_argument('--img_weight', type=float, default=50.0 )
    parser.add_argument('--msg_weight', type=float, default=1.0 )
    #train mode
    parser.add_argument('--stage', type=str, default='both',
                        help='choose training stage: style, stegano, both')

    #first stage args
    parser.add_argument('--identity_radio', type=int, default=5,)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True #use cudnn.
    print('loading argument')
    args =  argument()
    print(args)
    print('initializing tensorboard')
    writer = SummaryWriter(args.log_dir)
    #Style Transfer Stage
    if args.stage=='both' or args.stage=='style':
        print('start training Style Transfer Stage')
        print('initializing Style Transfer Stage Trainer')
        trainer = StyleTransferStageTrainer()
        print('initializing tensorboard')
        trainer.init_tensorboard(writer)
        print('initializing model')
        trainer.load_model(args.vgg_path)
        print('moving model to gpu')
        trianer = trainer.cuda()
        print('initializing optimizer')
        trainer.init_optim(args.lr, args.lr_decay, args.beta_1, args.beta_2)
        print('initializing train data')
        trainer.init_train_data(args.content_dir, args.style_dir, args.batch_size, args.num_workers, args.identity_radio, args.image_size)
        print('initializing chekpoint setting')
        trainer.init_cpt(args.cpt_iter, args.cpt_dir, args.result_dir)
        print('initializing test data')
        trainer.init_test_data(args.test_content_dir, args.test_style_dir)
        print('initializing loss weight')
        trainer.init_loss_weight(args.content_weight, args.style_weight)
        print('start fitting')
        trainer.fit(args.first_stage_iter)

    # Steganography Stage
    if args.stage=='both' or args.stage=='stegano':
        print('start training Steganography Stage')
        if args.stage=='both' or args.stage=='style':
            vgg = trainer.vgg
            decoder = trainer.decoder
        print('initializing Steganography Stage Trainer')
        trainer = SteganographyStageTrainer()
        print('initializing tensorboard')
        trainer.init_tensorboard(writer)
        print('initializing train data')
        trainer.init_train_data(args.content_dir, args.style_dir, args.batch_size, args.num_workers, args.image_size)
        print('initializing model')
        trainer.load_model(args.vgg_path, args.decoder_path, args.msg_encoder_path, args.msg_decoder_path)
        if args.stage=='both' or args.stage=='style':
            trainer.vgg = vgg
            trainer.decoder = decoder
        print('moving model to gpu')
        trianer = trainer.cuda()
        print('initializing chekpoint setting')
        trainer.init_cpt(args.cpt_iter, args.cpt_dir, args.result_dir)
        print('initializing test data')
        trainer.init_test_data(args.test_content_dir, args.test_style_dir)
        print('initializing optimizer')
        trainer.init_optim(args.lr, args.lr_decay, args.beta_1, args.beta_2)
        print('initializing loss weight')
        trainer.init_loss_weight(args.img_weight, args.msg_weight)
        print('start fitting')
        trainer.fit(args.second_stage_iter)