from __future__ import print_function
from six.moves import range
import os
import time
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
import datetime
import dateutil.tz
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import TEXT_TRANSFORMER_ENCODERv2, CNN_ENCODER
# from InceptionScore import calculate_inception_score

from miscc.losses import sent_loss, words_loss, caption_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import timeit

from catr.models import utils, caption
from catr.datasets import coco
from catr.cfg_damsm_bert import Config
# from catr.engine import train_one_epoch, evaluate

from transformers import BertTokenizer
from nltk.tokenize import RegexpTokenizer

config = Config() # initialize catr config here
tokenizer = BertTokenizer.from_pretrained(config.vocab, do_lower=True)
retokenizer = BertTokenizer.from_pretrained("catr/damsm_vocab.txt", do_lower=True)
# reg_tokenizer = RegexpTokenizer(r'\w+')
frozen_list_image_encoder = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3']

# @torch.no_grad()
# def evaluate(cnn_model, trx_model, cap_model, batch_size, cap_criterion, dataloader_val):
#     cnn_model.eval()
#     trx_model.eval()
#     cap_model.eval() ### 
#     s_total_loss = 0
#     w_total_loss = 0
#     c_total_loss = 0 ###
#     ### add caption criterion here. #####
# #     cap_criterion = torch.nn.CrossEntropyLoss() # add caption criterion here
#     labels = torch.LongTensor(range(batch_size)) # used for matching loss
#     if cfg.CUDA:
#         labels = labels.cuda()
# #         cap_criterion = cap_criterion.cuda() # add caption criterion here
# #     cap_criterion.eval()
#     #####################################

#     val_data_iter = iter(dataloader_val)
#     for step in tqdm(range(len(val_data_iter)),leave=False):
#         data = val_data_iter.next()

#         real_imgs, captions, cap_lens, class_ids, keys, cap_imgs, cap_img_masks, sentences, sent_masks = prepare_data(data)

#         words_features, sent_code = cnn_model(cap_imgs)

#         words_emb, sent_emb = trx_model(captions)

#         ##### add catr here #####
#         cap_preds = cap_model(words_features, cap_img_masks, sentences[:, :-1], sent_masks[:, :-1]) # caption model feedforward

#         cap_loss = caption_loss(cap_criterion, cap_preds, sentences)

#         c_total_loss += cap_loss.item()
#         #########################

#         w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
#                                             cap_lens, class_ids, batch_size)
#         w_total_loss += (w_loss0 + w_loss1).item()

#         s_loss0, s_loss1 = \
#             sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
#         s_total_loss += (s_loss0 + s_loss1).item()

# #             if step == 50:
# #                 break

#     s_cur_loss = s_total_loss / step
#     w_cur_loss = w_total_loss / step
#     c_cur_loss = c_total_loss / step

#     return s_cur_loss, w_cur_loss, c_cur_loss

            
# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader,dataloader_val, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = data_loader.batch_size
        self.val_batch_size = dataloader_val.batch_size
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.dataloader_val = dataloader_val
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        ####################################################################
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        print('Load image encoder from:', img_encoder_path)
        state_dict = torch.load(img_encoder_path, map_location='cpu')
        image_encoder.load_state_dict(state_dict['model'])
        for p in image_encoder.parameters(): # make image encoder grad on
            p.requires_grad = True
        for k,v in image_encoder.named_children(): # freeze the layer1-5 (set eval for BNlayer)
            if k in frozen_list_image_encoder:
                v.train(False)
                v.requires_grad_(False)
        
#         image_encoder.eval()

        ###################################################################
        text_encoder = TEXT_TRANSFORMER_ENCODERv2(emb=cfg.TEXT.EMBEDDING_DIM
                                            ,heads=8
                                            ,depth=1
                                            ,seq_length=cfg.TEXT.WORDS_NUM
                                            ,num_tokens=self.n_words)
#         state_dict = torch.load(cfg.TRAIN.NET_E)
#         text_encoder.load_state_dict(state_dict)
#         print('Load ', cfg.TRAIN.NET_E)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        state_dict = torch.load(cfg.TRAIN.NET_E,map_location='cpu')
        text_encoder.load_state_dict(state_dict['model'])
        for p in text_encoder.parameters():
            p.requires_grad = True
        
#         text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict['model'])
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = torch.load(Dname, map_location='cpu')
                    netsD[i].load_state_dict(state_dict['model'])
        # ########################################################## #
#         config = Config()
        cap_model = caption.build_model_v3(config)
        print("Initializing from Checkpoint...")
        cap_model_path = cfg.TRAIN.NET_E.replace('text_encoder', 'cap_model')
        
        if os.path.exists(cap_model_path):
            print('Load C from: {0}'.format(cap_model_path))
            state_dict = \
                torch.load(cap_model_path, map_location=lambda storage, loc: storage)
            cap_model.load_state_dict(state_dict['model'])
        else:
            base_line_path = 'catr/checkpoints/catr_damsm256_proj_coco2014_ep02.pth'
            print('Load C from: {0}'.format(base_line_path))
            checkv3 = torch.load(base_line_path, map_location='cpu')
            cap_model.load_state_dict(checkv3['model'], strict=False)
        
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            cap_model = cap_model.cuda() # caption transformer added
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch, cap_model]

    def define_optimizers(self, image_encoder, text_encoder, netG, netsD, cap_model):
        ### change the learning rate in this function ###
        
        #################################
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
#         print('\n\n CNN Encoder parameters that do not require grad are:')
        paraI = []
        for k,v in image_encoder.named_parameters():
            if v.requires_grad:
                paraI.append(v)
#             else:
#                 print(k)
        optimizerI = torch.optim.Adam(paraI
                                           , lr=1e-5
                                           , weight_decay=config.weight_decay)
        lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI
                                                            , config.lr_drop)
        
        if os.path.exists(img_encoder_path):
            print('Load image encoder optimizer from:', img_encoder_path)
            state_dict = \
                torch.load(img_encoder_path, map_location='cpu')
            optimizerI.load_state_dict(state_dict['optimizer'])
            lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
        #################################
        text_encoder_path = cfg.TRAIN.NET_E
        optimizerT = torch.optim.Adam(text_encoder.parameters()
                                           , lr=1e-5
                                           , weight_decay=config.weight_decay)
        lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT
                                                            , config.lr_drop)
        
        print('Load text encoder optimizer from:', cfg.TRAIN.NET_E)        
        if os.path.exists(cfg.TRAIN.NET_E):
            state_dict = torch.load(cfg.TRAIN.NET_E,map_location='cpu')
            optimizerT.load_state_dict(state_dict['optimizer'])
            lr_schedulerT.load_state_dict(state_dict['lr_scheduler'])
        
            
        
        ##################################################
        optimizersD = []
        num_Ds = len(netsD)
        if cfg.TRAIN.B_NET_D:
            Gname = cfg.TRAIN.NET_G
            for i in range(num_Ds):
                s_tmp = Gname[:Gname.rfind('/')]
                Dname = '%s/netD%d.pth' % (s_tmp, i)
                print('Load Optimizer D from: ', Dname)
                state_dict = torch.load(Dname, map_location='cpu')
                opt = optim.Adam(netsD[i].parameters(),
                                 lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                 betas=(0.5, 0.999))
                opt.load_state_dict(state_dict['optimizer'])
                optimizersD.append(opt)
            
            
        ##################################################
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        if os.path.exists(cfg.TRAIN.NET_G):
            print('Load Generator optimizer from:',cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location='cpu')
            optimizerG.load_state_dict(state_dict['optimizer'])
#             lr_schedulerC.load_state_dict(state_dict['lr_scheduler'])
        # ################## CAPTION model here ################# #
        param_dicts = [
            {"params": [p for n, p in cap_model.named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
#             {
#                 "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#                 "lr": config.lr_backbone,
#             },
        ]
        
        optimizerC = torch.optim.AdamW(param_dicts
                                           , lr=config.lr
                                           , weight_decay=config.weight_decay)
        lr_schedulerC = torch.optim.lr_scheduler.StepLR(optimizerC
                                                        , config.lr_drop)
        cap_model_path = cfg.TRAIN.NET_E.replace('text_encoder', 'cap_model')
        if os.path.exists(cap_model_path):
            print('Load text encoder optimizer from:',cap_model_path)
            state_dict = \
                torch.load(cap_model_path, map_location='cpu')
            optimizerC.load_state_dict(state_dict['optimizer'])
            lr_schedulerC.load_state_dict(state_dict['lr_scheduler'])
        
        ####### change learning rate for each optimizer here ##########
        optimizerI.param_groups[0]['lr'] = 1e-5
        optimizerT.param_groups[0]['lr'] = 1e-6
        optimizerC.param_groups[0]['lr'] = 1e-6
        optimizerG.param_groups[0]['lr'] = 1e-6
        for optim_d in optimizersD:
            optim_d.param_groups[0]['lr'] = 1e-6
        ###############################################################

        return (optimizerI
                , optimizerT
                , optimizerG
                , optimizersD
                , optimizerC
                , lr_schedulerC
                , lr_schedulerI
                , lr_schedulerT)

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, image_encoder, text_encoder, netsD, epoch, cap_model, optimizerC, optimizerI, optimizerT, lr_schedulerC, lr_schedulerI, lr_schedulerT, optimizerG, optimizersD):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        
        
        
        torch.save({
            'model': netG.state_dict(),
            'optimizer': optimizerG.state_dict()
        }, '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            optD = optimizersD[i]           
            
            torch.save({
                'model': netD.state_dict(),
                'optimizer': optD.state_dict()
            }, '%s/netD%d.pth' % (self.model_dir, i))
            
#         print('Save G/Ds models.')
        # save caption model here
        torch.save({
            'model': cap_model.state_dict(),
            'optimizer': optimizerC.state_dict(),
            'lr_scheduler': lr_schedulerC.state_dict(),
        }, '%s/cap_model%d.pth' % (self.model_dir, epoch))

        
        # save image encoder model here
        torch.save({
            'model': image_encoder.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
        }, '%s/image_encoder%d.pth' % (self.model_dir, epoch))

        
        # save text encoder model here
        torch.save({
            'model': text_encoder.state_dict(),
            'optimizer': optimizerT.state_dict(),
            'lr_scheduler': lr_schedulerT.state_dict(),
        }, '%s/text_encoder%d.pth' % (self.model_dir, epoch))

        
    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    @torch.no_grad()
    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        netG.eval()
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img, batch_size = self.batch_size)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze, batch_size = self.batch_size)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        
            
    def train(self):
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME+'-s3-00_00_00_00_01', timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging

        
        ####### init models ########
        text_encoder, image_encoder, netG, netsD, start_epoch, cap_model = self.build_models()
        labels = Variable(torch.LongTensor(range(self.batch_size))) # used for matching loss
        
        text_encoder.train()
        image_encoder.train()
        for k,v in image_encoder.named_children(): # set the input layer1-5 not training and no grads.
            if k in frozen_list_image_encoder:
                v.training = False
                v.requires_grad_(False)
        netG.train()
        for i in range(len(netsD)):
            netsD[i].train()
        cap_model.train()


        
        avg_param_G = copy_G_params(netG)
        
        ###############################################################
        
        ###### init optimizers #####
        optimizerI, optimizerT, optimizerG , optimizersD , optimizerC , lr_schedulerC \
        , lr_schedulerI , lr_schedulerT = self.define_optimizers(image_encoder
                                                                , text_encoder
                                                                , netG
                                                                , netsD
                                                                , cap_model)
        ############################################
        
        ##### init data #############################
        
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        ##################################################################
        
        
        
        ###### init caption model criterion ############
        cap_criterion = torch.nn.CrossEntropyLoss() # add caption criterion here
        if cfg.CUDA:
            labels = labels.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            cap_criterion = cap_criterion.cuda() # add caption criterion here
        cap_criterion.train()
        #################################################
        
        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            text_encoder.train()
            image_encoder.train()
            netG.train()
            cap_model.train()
            for k,v in image_encoder.named_children():
                if k in frozen_list_image_encoder:
                    v.train(False)
            for i in range(len(netsD)):
                netsD[i].train()
            ####################################
            
            ####### init loss variables ############
            fi_w_total_loss0 = 0 
            fi_w_total_loss1 = 0 
            fi_s_total_loss0 = 0 
            fi_s_total_loss1 = 0
            fi_total_damsm_loss = 0
            
            
            ft_w_total_loss0 = 0 
            ft_w_total_loss1 = 0 
            ft_s_total_loss0 = 0 
            ft_s_total_loss1 = 0
            ft_total_damsm_loss = 0
            
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            total_damsm_loss = 0
            
            c_total_loss = 0
            
            total_multimodal_loss = 0
            
            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            print('Learning rates: lr_i %.7f, lr_t %.7f, lr_c %.7f, lr_g %.7f, lr_d0 %.7f, lr_d1 %.7f, lr_d2 %.7f' 
                 % (optimizerI.param_groups[0]['lr'], optimizerT.param_groups[0]['lr'], 
                    optimizerC.param_groups[0]['lr'], optimizerG.param_groups[0]['lr'], 
                    optimizersD[0].param_groups[0]['lr'], optimizersD[1].param_groups[0]['lr'], optimizersD[2].param_groups[0]['lr'])) 
            #########################################################################################
            
            start_t = time.time()

            data_iter = iter(self.data_loader)
#             step = 0
            pbar = tqdm(range(self.num_batches))
            for step in pbar: 
#             while step < self.num_batches:
#                 print('step:{:6d}|{:3d}'.format(step,self.num_batches), end='\r')
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                # add images, image masks, captions, caption masks for catr model
                imgs, captions, cap_lens, class_ids, keys, cap_imgs, cap_img_masks, sentences, sent_masks = prepare_data(data)
                
                ################## feedforward damsm model ##################
                image_encoder.zero_grad() # image/text encoders zero_grad here
                text_encoder.zero_grad()
                
                words_features, sent_code = image_encoder(cap_imgs) # input catr images to image encoder, feedforward, Nx256x17x17
#                 words_features, sent_code = image_encoder(imgs[-1]) # input image_encoder
                nef, att_sze = words_features.size(1), words_features.size(2)
                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions)#, cap_lens, hidden)
                

                
                #### damsm losses
                w_loss0, w_loss1, attn_maps = words_loss(words_features, words_embs, labels, cap_lens, class_ids, batch_size)
                w_total_loss0 += w_loss0.item()
                w_total_loss1 += w_loss1.item()
                damsm_loss = w_loss0 + w_loss1
                
                s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
                s_total_loss0 += s_loss0.item()
                s_total_loss1 += s_loss1.item()
                damsm_loss += s_loss0 + s_loss1
                
                total_damsm_loss += damsm_loss.item()
                ######################################################################################
                
#                 damsm_loss.backward() # do not backward because we will add all
                
#                 words_features = words_features.detach()
                # real image real text matching loss graph cleared here
                # grad accumulated -> text_encoder 
                #                  -> image_encoder 
                #################################################################################
                
                ################## feedforward caption model ##################
                cap_model.zero_grad() # caption model zero_grad here
             
                cap_preds = cap_model(words_features, cap_img_masks, sentences[:, :-1], sent_masks[:, :-1]) # caption model feedforward
                cap_loss = caption_loss(cap_criterion, cap_preds, sentences)
                c_total_loss += cap_loss.item()
#                 cap_loss.backward() # caption loss graph cleared, 
                # grad accumulated -> cap_model -> image_encoder 
                
#                 optimizerC.step() # update cap_model params
                #################################################################################
                
                ############ Prepare the input to Gan from the output of text_encoder ################
                words_embs_detached, sent_emb_detached = words_embs.detach(), sent_emb.detach()
                
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask) ## use the attached version because we want to update the text encoder from gan loss 
                
                
                
                
                
#                 f_img = np.asarray(fake_imgs[-1].permute((0,2,3,1)).detach().cpu())
#                 print('fake_imgs.size():{0},fake_imgs.min():{1},fake_imgs.max():{2}'.format(fake_imgs[-1].size()
#                                   ,fake_imgs[-1].min()
#                                   ,fake_imgs[-1].max()))
                
#                 print('f_img.shape:{0}'.format(f_img.shape))
                
                #######################################################
                # (3) Update D network is a separate training from joint training
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
#                     print(i)
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb_detached, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
#                 step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, fake_imgs, real_labels,
                                   words_embs, sent_emb_detached, match_labels, cap_lens, class_ids) # detached sent_emb_detached
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                

                ####### fake imge real text matching loss #################
                fi_word_features, fi_sent_code = image_encoder(fake_imgs[-1])
#                 words_embs, sent_emb = text_encoder(captions) # to update the text 
                
            
                fi_w_loss0, fi_w_loss1, fi_attn_maps = words_loss(fi_word_features, words_embs, labels,
                                                 cap_lens, class_ids, batch_size)
        
                fi_w_total_loss0 += fi_w_loss0.item()
                fi_w_total_loss1 += fi_w_loss1.item()
                
                fi_damsm_loss = fi_w_loss0 + fi_w_loss1
                
                fi_s_loss0, fi_s_loss1 = sent_loss(fi_sent_code, sent_emb, labels, class_ids, batch_size)
                
                fi_s_total_loss0 += fi_s_loss0.item()
                fi_s_total_loss1 += fi_s_loss1.item()
                
                fi_damsm_loss += fi_s_loss0 + fi_s_loss1
                fi_total_damsm_loss += fi_damsm_loss.item()
                #######################################################################
#                 fi_damsm_loss.backward()
                
                ###### real image fake text matching loss ##############
                
                fake_preds = torch.argmax(cap_preds, axis=-1) # capation predictions
                fake_captions = tokenizer.batch_decode(fake_preds.tolist(), skip_special_tokens=True) # list of strings
                fake_outputs = retokenizer.batch_encode_plus(
                        fake_captions, max_length=64, padding='max_length', add_special_tokens=False,
                        return_attention_mask=True, return_token_type_ids=None, truncation=True)
                fake_tokens = fake_outputs['input_ids']
#                 fake_tkmask = fake_outputs['attention_mask']
                f_tokens = np.zeros((len(fake_tokens), 15), dtype=np.int64)
                f_cap_lens = []
                cnt=0
                for i in fake_tokens: 
                    temp = np.array([x for x in i if x!=27299 and x!=0])
                    num_words = len(temp)
                    if num_words <= 15:
                        f_tokens[cnt][:num_words] = temp
                    else:
                        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                        np.random.shuffle(ix)
                        ix = ix[:15]
                        ix = np.sort(ix)
                        f_tokens[cnt] = temp[ix]
                        num_words = 15
                    f_cap_lens.append(num_words)
                    cnt += 1
                
                f_tokens = Variable(torch.tensor(f_tokens))
                f_cap_lens = Variable(torch.tensor(f_cap_lens))
                if cfg.CUDA:
                    f_tokens = f_tokens.cuda()
                    f_cap_lens = f_cap_lens.cuda()           
                    
                ft_words_emb, ft_sent_emb = text_encoder(f_tokens) # input text_encoder
                
            
                ft_w_loss0, ft_w_loss1, ft_attn_maps = words_loss(words_features, ft_words_emb, labels,
                                                 f_cap_lens, class_ids, batch_size)
        
                ft_w_total_loss0 += ft_w_loss0.item()
                ft_w_total_loss1 += ft_w_loss1.item()
                
                ft_damsm_loss = ft_w_loss0 + ft_w_loss1
                
                ft_s_loss0, ft_s_loss1 = sent_loss(sent_code, ft_sent_emb, labels, class_ids, batch_size)
                
                ft_s_total_loss0 += ft_s_loss0.item()
                ft_s_total_loss1 += ft_s_loss1.item()
                
                ft_damsm_loss += ft_s_loss0 + ft_s_loss1   
                ft_total_damsm_loss += ft_damsm_loss.item()
                ############################################################################
        
#                 ft_damsm_loss.backward()

                ###########################################################################
 
                multimodal_loss = cfg.TRAIN.SMOOTH.LAMBDA_DAMSM*damsm_loss + \
                                    cfg.TRAIN.SMOOTH.LAMBDA_FI*fi_damsm_loss + \
                                    cfg.TRAIN.SMOOTH.LAMBDA_FT*ft_damsm_loss + \
                                    cfg.TRAIN.SMOOTH.LAMBDA_CAP*cap_loss + \
                                    cfg.TRAIN.SMOOTH.LAMBDA_GEN*errG_total
                total_multimodal_loss += multimodal_loss.item()
                ############################################################################
                
                multimodal_loss.backward()
                ## loss = 0.5*loss1 + 0.4*loss2 + ...
                ## loss.backward() -> accumulate grad value in parameters.grad
                
                ## loss1 = 0.5*loss1
                ## loss1.backward() 
                    
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
                    
                optimizerI.step()
                
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
                optimizerT.step()
                
                torch.nn.utils.clip_grad_norm_(cap_model.parameters(),
                                      config.clip_max_norm)
                
                optimizerC.step() # update cap_model params
                
                ####### backward and update GAN parameters ########
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)
    
                ##################### loss values for each step #########################################
                tbw.add_scalar('Train_step/step_loss_D', float(errD_total.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/step_loss_G', float(errG_total.item()), step + epoch * self.num_batches)

                ## damsm ##
#                 tbw.add_scalar('Train_step/train_w_step_loss0', float(w_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_s_step_loss0', float(s_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_w_step_loss1', float(w_loss1.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_s_step_loss1', float(s_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_damsm_step_loss', float(damsm_loss.item()), step + epoch * self.num_batches)

                ## damsm fi rt ##
#                 tbw.add_scalar('Train_step/train_fi_w_step_loss0', float(fi_w_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_fi_s_step_loss0', float(fi_s_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_fi_w_step_loss1', float(fi_w_loss1.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_fi_s_step_loss1', float(fi_s_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_fi_damsm_step_loss', float(fi_damsm_loss.item()), step + epoch * self.num_batches)
                
                ## damsm ri ft ##
#                 tbw.add_scalar('Train_step/train_ft_w_step_loss0', float(ft_w_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_ft_s_step_loss0', float(ft_s_loss0.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_ft_w_step_loss1', float(ft_w_loss1.item()), step + epoch * self.num_batches)
#                 tbw.add_scalar('Train_step/train_ft_s_step_loss1', float(ft_s_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_ft_damsm_step_loss', float(ft_damsm_loss.item()), step + epoch * self.num_batches)

                ## caption loss ###
                tbw.add_scalar('Train_step/train_c_step_loss', float(cap_loss.item()), step + epoch * self.num_batches)

                ## multimodal loss ##
                tbw.add_scalar('Train_step/train_multimodal_step_loss', float(multimodal_loss.item()), step + epoch * self.num_batches)
                ################################################################################################    
                
                ############ tqdm descriptions showing running average loss in terminal ##############################
                pbar.set_description('multimodal %.5f, damsm %.5f, fi %.5f, ft %.5f, cap %.5f' 
                                     % (float(total_multimodal_loss) / (step+1), float(total_damsm_loss) / (step+1), 
                                        float(fi_total_damsm_loss) / (step+1), float(ft_total_damsm_loss) / (step+1), 
                                        float(c_total_loss) / (step+1)))
                ######################################################################################################
                # 14 -- 2800 iterations=steps for 1 epoch    
#                 if gen_iterations % 100 == 0:
#                     print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 5000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    netG.train() ### convert back to train, because in save_img_results we do netG.eval()
                
                ##########################################################
                if step % 1000 == 0 and step != 0:
                    
                    ##################### Average values of each loss every 1000 steps ##########################
                    tbw.add_scalar('Loss_D', float(errD_total), tensorboard_step)
                    tbw.add_scalar('Loss_G', float(errG_total), tensorboard_step)

                    ## damsm ##
                    tbw.add_scalar('train_w_loss0', float(w_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_s_loss0', float(s_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_w_loss1', float(w_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_s_loss1', float(s_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('total_damsm_loss', float(total_damsm_loss)/(step+1), tensorboard_step)

                    ## damsm fi rt ##
                    tbw.add_scalar('train_fi_w_loss0', float(fi_w_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_fi_s_loss0', float(fi_s_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_fi_w_loss1', float(fi_w_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_fi_s_loss1', float(fi_s_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('fi_total_damsm_loss', float(fi_total_damsm_loss)/(step+1), tensorboard_step)

                    ## damsm ri ft ##
                    tbw.add_scalar('train_ft_w_loss0', float(ft_w_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_ft_s_loss0', float(ft_s_total_loss0)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_ft_w_loss1', float(ft_w_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('train_ft_s_loss1', float(ft_s_total_loss1)/(step+1), tensorboard_step)
                    tbw.add_scalar('ft_total_damsm_loss', float(ft_total_damsm_loss)/(step+1), tensorboard_step)

                    ## caption loss ###
                    tbw.add_scalar('train_c_loss', float(c_total_loss)/(step+1), tensorboard_step)

                    ## multimodal loss ##
                    tbw.add_scalar('train_multimodal_loss', float(total_multimodal_loss)/(step+1), tensorboard_step)
                    
                    #### validate ####
                    v_s_cur_loss, v_w_cur_loss, v_c_cur_loss = self.evaluate(image_encoder, text_encoder, 
                                                                        cap_model, self.val_batch_size)
                    ### val losses ###
                    tbw.add_scalar('Val_step/val_w_loss', float(v_w_cur_loss), tensorboard_step)
                    tbw.add_scalar('Val_step/val_s_loss', float(v_s_cur_loss), tensorboard_step)
                    tbw.add_scalar('Val_step/val_c_loss', float(v_c_cur_loss), tensorboard_step)


                    ### back to train ###
                    text_encoder.train()
                    image_encoder.train()
                    netG.train()
                    cap_model.train()
                    for k,v in image_encoder.named_children():
                        if k in frozen_list_image_encoder:
                            v.train(False)
                    for i in range(len(netsD)):
                        netsD[i].train()

                    
                    tensorboard_step+=1
                    
                    #### save model update the files every 1000 iters within epoch 
                    self.save_model(netG, avg_param_G, image_encoder, text_encoder, netsD, epoch, cap_model, optimizerC, optimizerI, optimizerT, lr_schedulerC, lr_schedulerI, lr_schedulerT, optimizerG, optimizersD)
                
                
#             print('step:',step)
            ######### no need to use lr_scheduler, keep lr constant ########
#             lr_schedulerC.step()
#             lr_schedulerI.step()
#             lr_schedulerT.step()
            
            end_t = time.time()
            
            
            ################################################################################################

            
#             print('''[%d/%d][%d]
#                   Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
#                   % (epoch, self.max_epoch, self.num_batches,
#                      errD_total.data, errG_total.data,
#                      end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                
                self.save_model(netG, avg_param_G, image_encoder, text_encoder, netsD, epoch, cap_model, optimizerC, optimizerI, optimizerT, lr_schedulerC, lr_schedulerI, lr_schedulerT, optimizerG, optimizersD)
                
                
            


#             print('v_s_cur_loss:{:.5f}, v_w_cur_loss:{:.5f}, v_c_cur_loss:{:.5f}'.format(v_s_cur_loss, v_w_cur_loss, v_c_cur_loss))


        self.save_model(netG, avg_param_G, image_encoder, text_encoder, netsD, self.max_epoch, cap_model, optimizerC, optimizerI, optimizerT, lr_schedulerC, lr_schedulerI, lr_schedulerT, optimizerG, optimizersD)

    @torch.no_grad()
    def evaluate(self, cnn_model, trx_model, cap_model, batch_size):
        cnn_model.eval()
        trx_model.eval()
        cap_model.eval() ### 
        s_total_loss = 0
        w_total_loss = 0
        c_total_loss = 0 ###
        ### add caption criterion here. #####
        cap_criterion = torch.nn.CrossEntropyLoss() # add caption criterion here
        labels = Variable(torch.LongTensor(range(batch_size))) # used for matching loss
        if cfg.CUDA:
            labels = labels.cuda()
            cap_criterion = cap_criterion.cuda() # add caption criterion here
        cap_criterion.eval()
        #####################################
        
        val_data_iter = iter(self.dataloader_val)
        for step in tqdm(range(len(val_data_iter)),leave=False):
            data = val_data_iter.next()
            
            real_imgs, captions, cap_lens, class_ids, keys, cap_imgs, cap_img_masks, sentences, sent_masks = prepare_data(data)
            
            words_features, sent_code = cnn_model(cap_imgs)

            words_emb, sent_emb = trx_model(captions)
            
            ##### add catr here #####
            cap_preds = cap_model(words_features, cap_img_masks, sentences[:, :-1], sent_masks[:, :-1]) # caption model feedforward

            cap_loss = caption_loss(cap_criterion, cap_preds, sentences)

            c_total_loss += cap_loss.data
            #########################

            w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                                cap_lens, class_ids, batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = \
                sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total_loss += (s_loss0 + s_loss1).data

#             if step == 50:
#                 break

        s_cur_loss = s_total_loss / step
        w_cur_loss = w_total_loss / step
        c_cur_loss = c_total_loss / step

        return s_cur_loss, w_cur_loss, c_cur_loss
        
        
    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir,full_netGpath):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for generator models is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = TEXT_TRANSFORMER_ENCODERv2(emb=cfg.TEXT.EMBEDDING_DIM
                                            ,heads=8
                                            ,depth=1
                                            ,seq_length=cfg.TEXT.WORDS_NUM
                                            ,num_tokens=self.n_words)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            cfg.TRAIN.NET_G = full_netGpath
            
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

#                     hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions)#, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)
                    

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = TEXT_TRANSFORMER_ENCODERv2(emb=cfg.TEXT.EMBEDDING_DIM
                                            ,heads=8
                                            ,depth=1
                                            ,seq_length=cfg.TEXT.WORDS_NUM
                                            ,num_tokens=self.n_words)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = '../gen_images_test'#cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
#                     hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions)#, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
