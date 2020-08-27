import time
import os 
import argparse

import model
import dataset
import utils

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import copy

import torchvision
from torchvision import transforms

from tqdm import tqdm


from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

from collections import OrderedDict

class Trainer(model.SelfContained_Style_Transfer):
    def __init__(self,):
        super().__init__()
        self.n_iter = 0
        self.mse_loss = nn.MSELoss()
    
    def init_tensorboard(self,log_dir):
        self.writer = SummaryWriter(log_dir)

    def load_model(self, vgg_path, plain_decoder_path, tune_vgg_path=None, encrypter_path=None, decrypter_path=None):
        '''
        load pretrained vgg and plain decoder
        other default is None for cpt training
        '''
        if encrypter_path is not None:
            self.encrypter.load_state_dict(torch.load(encrypter_path))

        self.plain_decoder.load_state_dict(torch.load(plain_decoder_path))
        vgg = model.VGG().eval()
        vgg.load_state_dict(torch.load(vgg_path))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.fixed_feat_extractor = model.Feature_extractor(vgg, False)

        self.vgg_tune = model.VGG()
        if tune_vgg_path:
            self.vgg_tune = nn.Sequential(*list(self.vgg_tune.children())[:31])
            self.vgg_tune.load_state_dict(torch.load(tune_vgg_path))
        else:
            self.vgg_tune.load_state_dict(torch.load(vgg_path))
            self.vgg_tune = nn.Sequential(*list(self.vgg_tune.children())[:31])            
        self.tune_feat_extractor = model.Feature_extractor(self.vgg_tune, True)

        if decrypter_path is not None:
            decrypter.load_state_dict(torch.load(decrypter_path))

    def move_to_cuda(self,):
        self.encrypter = self.encrypter.cuda()
        self.plain_decoder = self.plain_decoder.cuda()
        self.fixed_feat_extractor = self.fixed_feat_extractor.cuda()
        self.tune_feat_extractor = self.tune_feat_extractor.cuda()
        self.decrypter = self.tune_feat_extractor.cuda()

    def init_optim(self, lr, lr_decay):
        self.lr = lr
        self.lr_decay = lr_decay
        self.encrypter_opt = torch.optim.Adam(self.encrypter.parameters(), lr=lr)
        self.vgg_tune_opt = torch.optim.SGD(self.vgg_tune.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
        self.decryter_opt = torch.optim.Adam(self.decrypter.parameters(), lr=lr)

    def init_train_data(self,content_dir, style_dir, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.content_dir = content_dir
        self.style_dir = style_dir

        content_tf = dataset.train_transform()
        style_tf = dataset.train_transform()

        content_dataset = dataset.FlatFolderDataset(content_dir, content_tf)
        style_dataset = dataset.FlatFolderDataset(style_dir, style_tf)
        self.content_iter = iter(data.DataLoader(content_dataset, batch_size=self.batch_size,
                            sampler=dataset.InfiniteSamplerWrapper(content_dataset),
                            num_workers=self.num_workers))
        self.style_iter = iter(data.DataLoader(style_dataset, batch_size=self.batch_size,
                          sampler=dataset.InfiniteSamplerWrapper(style_dataset),
                          num_workers=self.num_workers))

    def init_test_data(self, test_content_img, test_style_img, test_size=256, crop=True):
        self.test_content_tf = dataset.test_transform(test_size, crop)
        self.test_style_tf = dataset.test_transform(test_size, crop)
        self.test_content = self.test_content_tf(Image.open(test_content_img))
        self.test_style = self.test_style_tf(Image.open(test_style_img))
        self.test_content = self.test_content.cuda().unsqueeze(0)
        self.test_style = self.test_style.cuda().unsqueeze(0)
        self.test_content_f, _ = self.fixed_feat_extractor(self.test_content)
        _, self.test_style_f = self.fixed_feat_extractor(self.test_style)
        self.test_c_mean, self.test_c_std = self.calc_mean_std(self.test_content_f)
        self.test_feat = self.adaptive_instance_normalization(self.test_content_f, self.test_style_f[-1])

        self.writer.add_images('style', self.test_style.cpu(), self.n_iter)
        self.writer.add_images('content', self.test_content.cpu(), self.n_iter)

    def init_loss_weight(self, content_weight=2.0, style_weight=10.0, decry_weight=30.0, inv_weight=5.0, vgg_weight=5.0):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.decry_weight = decry_weight
        self.inv_weight = inv_weight
        self.vgg_weight = vgg_weight

    def init_cpt(self, cpt_interval, cpt_dir, result_dir):
        self.cpt_interval = cpt_interval
        self.cpt_dir = cpt_dir
        utils.check_and_make_dir(self.cpt_dir)
        self.result_dir = result_dir
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

    def inverse_AdaIN(self,content_feat, content_std, content_mean):
        size = content_feat.size()
        cur_mean, cur_std = self.calc_mean_std(content_feat)
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

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        mse_loss = nn.MSELoss()
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        mse_loss = nn.MSELoss()
        return mse_loss(input, target)

    def adjust_learning_rate(self, optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = self.lr / (1.0 + self.lr_decay * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _fit_step(self,):
        # adjust the learning rate
        self.adjust_learning_rate(self.encrypter_opt, iteration_count=self.n_iter)
        self.adjust_learning_rate(self.decryter_opt, iteration_count=self.n_iter)
        # get the images
        content_images = next(self.content_iter).cuda()
        style_images = next(self.style_iter).cuda()
        
        # get the feature informaiton
        content_feat, content_style = self.fixed_feat_extractor(content_images)
        _, style_feats = self.fixed_feat_extractor(style_images)
        c_mean, c_std = self.calc_mean_std(content_feat)
        
        # transfer the style
        feat = self.adaptive_instance_normalization(content_feat, style_feats[-1])
        
        # decode the image
        st = self.encrypted_decoding(feat, c_std, c_mean)
        
        # compute the style transfer losses
        st_content_feat, st_style_feats = self.fixed_feat_extractor(st)
        loss_c = self.calc_content_loss(st_content_feat, feat)
        loss_s = self.calc_style_loss(st_style_feats[0], style_feats[0])
        for s_layer in range(1, 4):
            loss_s += self.calc_style_loss(st_style_feats[s_layer], style_feats[s_layer])
        
        # compute the vgg loss
        st_content_feat_tune, _ = self.tune_feat_extractor(st)
        loss_vgg = self.calc_content_loss(st_content_feat_tune, feat)
        
        # decrypt encoded content from st and compute the loss
        decry_info = self.decrypter(st)
        decry_c_std = decry_info[:, 0:512, :, :]
        decry_c_mean = decry_info[:, 512:1024, :, :]
        loss_decry = self.mse_loss(decry_c_std, c_std) + self.mse_loss(decry_c_mean, c_mean)
        
        # apply inverse AdaIN and compute the loss
        recon_feat = self.inverse_AdaIN(st_content_feat_tune, decry_c_std, decry_c_mean)
        loss_inv = self.mse_loss(recon_feat, content_feat)
        
        # multiply the losses by weights
        loss_c = self.content_weight * loss_c
        loss_s = self.style_weight * loss_s
        loss_decry = self.decry_weight * loss_decry
        loss_vgg = self.vgg_weight * loss_vgg
        loss_inv = self.inv_weight * loss_inv    
        
        loss_st = loss_c + loss_s + loss_decry + loss_vgg + loss_inv
    
        # get the gradients
        self.encrypter_opt.zero_grad()
        self.decryter_opt.zero_grad()
        self.vgg_tune_opt.zero_grad()
        
        loss_st.backward()
        
        self.encrypter_opt.step()
        self.decryter_opt.step()
        self.vgg_tune_opt.step()

        self.n_iter+=1

        #tensorboard
        self.writer.add_scalar('loss_content', loss_c, self.n_iter)
        self.writer.add_scalar('loss_style', loss_s, self.n_iter)
        self.writer.add_scalar('loss_decry', loss_decry, self.n_iter)
        self.writer.add_scalar('loss_vgg', loss_vgg, self.n_iter)
        self.writer.add_scalar('loss_inv', loss_inv, self.n_iter)
        self.writer.add_scalar('loss_st', loss_st, self.n_iter)

        self.loss_c = loss_c.item()
        self.loss_s = loss_s.item()
        self.loss_decry = loss_decry.item()
        self.loss_vgg = loss_vgg.item()
        self.loss_inv = loss_inv.item()
        self.loss_st = loss_st.item()

    def _save_cpt(self,):
        # save the st_decoder parameters
        state_dict = self.encrypter.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}encrypter_{:d}.pth'.format(self.cpt_dir, self.n_iter))
        
        # save the content_decrypter parameters
        state_dict = self.decrypter.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}decrypter_{:d}.pth'.format(self.cpt_dir, self.n_iter))
        
        # save the vgg_tune parameters
        state_dict = self.vgg_tune.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,'{:s}tuned_vgg_{:d}.pth'.format(self.cpt_dir, self.n_iter))
        
    def _test(self,):
        # set the mode of the models
        self.encrypter.eval()
        self.decrypter.eval()

        # generate the encoded style-transferred image
        test_st = self.encrypted_decoding(self.test_feat, self.test_c_std, self.test_c_mean)

        # save the st image
        output = test_st.cpu()
        output_name = self.result_dir + 'st_' + str(self.n_iter) + '.jpg'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('encrypted', output, self.n_iter)
        
        # decrypt content mean and std from st
        decry_info = self.decrypter(test_st)
        decry_c_std = decry_info[:, 0:512, :, :]
        decry_c_mean = decry_info[:, 512:1024, :, :]

        # get the content of st
        recon_content_f, _ = self.tune_feat_extractor(test_st)

        # apply inverse AdaIN to transfer the style
        recon_feat = self.inverse_AdaIN(recon_content_f, decry_c_std, decry_c_mean)

        # decode the inverse AdaIN image
        recon_img = self.plain_decoder(recon_feat)

        output = recon_img.cpu()
        output_name =  self.result_dir + 'recon_' + str(self.n_iter) + '.jpg'
        torchvision.utils.save_image(output, output_name)
        self.writer.add_images('reconstructed', output, self.n_iter)
        
        # set the mode of the models
        self.encrypter.train()
        self.decrypter.train()

    def fit(self,max_iter):
        self.max_iter = max_iter
        pbar = tqdm(range(self.max_iter))
        for n_iter in pbar:
            self._fit_step()
            pbar.set_description('c:%4.2f|s:%4.2f|decry:%4.2f|vgg:%4.2f|inv:%4.2f|total:%4.2f'
                %(self.loss_c, self.loss_s, self.loss_decry, self.loss_vgg, self.loss_inv, self.loss_st))
            if self.n_iter % self.cpt_interval == 0 or self.n_iter == max_iter:
                self._save_cpt()
                self._test()

def argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content_dir', type=str, default='/home/ishengfang/hdd/mscoco2017/train2017/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='/home/ishengfang/hdd/painter_by_numbers/train/',
                        help='Directory path to a batch of style images')

    parser.add_argument('--test_content', type=str, default='../input/content/avril.jpg',
                        help='Directory path to a test content images')
    parser.add_argument('--test_style', type=str, default='../input/style/antimonocromatismo.jpg',
                        help='Directory path to a test style images')

    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='Directory path for output images')

    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help='Directory path for tensorboard logs')
    parser.add_argument('--cpt_dir', type=str, default='./cpts/',
                        help='Directory path for model checkpoints')

    parser.add_argument('--vgg_path', type=str, default='./model_weights/vgg_normalised.pth')
    parser.add_argument('--plain_decoder_path', type=str, default='./model_weights/plain_decoder.pth')

    parser.add_argument('--batch_size', type=int, default=4,)
    parser.add_argument('--num_workers', type=int, default=8,)

    parser.add_argument('--max_iter', type=int, default=1000000,)
    parser.add_argument('--cpt_iter', type=int, default=10000,)
    parser.add_argument('--lr', type=float, default=1e-4 )
    parser.add_argument('--lr_decay', type=float, default=5e-5 )

    #loss weight
    parser.add_argument('--content_weight', type=float, default=2.0 )
    parser.add_argument('--style_weight', type=float, default=10.0 )
    parser.add_argument('--decry_weight', type=float, default=30.0 )
    parser.add_argument('--inv_weight', type=float, default=5.0 )
    parser.add_argument('--vgg_weight', type=float, default=5.0 )


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print('loading argument')
    args =  argument()
    print(args)
    print('initializing trainer')
    trainer = Trainer()
    print('initializing tensorboard')
    trainer.init_tensorboard(args.log_dir)
    print('initializing model')
    trainer.load_model(args.vgg_path, args.plain_decoder_path)
    print('moving model to gpu')
    trianer = trainer.cuda()
    print('initializing optimizer')
    trainer.init_optim(args.lr, args.lr_decay)
    print('initializing train data')
    trainer.init_train_data(args.content_dir, args.style_dir, args.batch_size, args.num_workers)
    print('initializing test data')
    trainer.init_test_data(args.test_content, args.test_style)
    print('initializing loss weight')
    trainer.init_loss_weight(args.content_weight, args.style_weight, args.decry_weight, 
                             args.inv_weight, args.vgg_weight)
    print('initializing chekpoint setting')
    trainer.init_cpt(args.cpt_iter, args.cpt_dir, args.result_dir)
    print('start fitting')
    trainer.fit(args.max_iter)