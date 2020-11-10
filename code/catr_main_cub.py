import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import numpy as np
import time
import sys
import os
import math
import tqdm
import timeit
from datetime import datetime
import dateutil.tz

from catr.models import utils, caption, backbone
from catr.datasets import coco
from catr.cfg_damsm_cub import Config
# from catr.engine import train_one_epoch, evaluate

from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(image_encoder, model, criterion, data_loader,
                    optimizer, device, epoch, max_norm, tbw):
    model.train()
    model.backbone.eval() # keep damsm cnn encoder freeze
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)
    cnt = 0

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            features = image_encoder(images)
#             samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(features, masks, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value
            cnt += 1

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)
            pbar.set_description('Train Loss (run_avg): %.6f, batch loss: %.6f' 
                                 % (epoch_loss / cnt, loss_value))
            tbw.add_scalar('Train/total_loss_iter', loss_value, cnt + total * epoch)
            
#             if cnt == 50:
#                 break

    return epoch_loss / total


@torch.no_grad()
def evaluate(image_encoder, model, criterion, data_loader, device, epoch, tbw):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)
    cnt = 0

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            features = image_encoder(images)
#             samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(features, masks, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()
            cnt += 1

            pbar.update(1)
            pbar.set_description('Validation loss (run_avg): %.6f, batch loss: %.6f' 
                                 % (validation_loss / cnt, loss.item()))
#             tbw.add_scalar('Val/total_loss_iter', loss.item(), cnt + total * epoch)

#             if cnt == 200:
#                 break
        
    return validation_loss / total


def main(config):
    device = torch.device(config.device)
    cudnn.benchmark = True
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ## load cnn encoder ##
    checkpoint_cnn = '/media/MyDataStor1/mmrl/MMRL/output/T_L1_TRX_birds_DAMSM_2020_11_09_15_02_21/Model/image_encoder600.pth'
    checkpt = torch.load(checkpoint_cnn, map_location='cpu') 
    image_encoder = backbone.CNN_ENCODER(nef=256, checkpoint=checkpt['model'])
    image_encoder.to(device)
    image_encoder.requires_grad_(False)
    image_encoder.eval()

    model = caption.build_model_v3(config)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
#     print("Initializing from Checkpoint V3...")
#     checkv3 = torch.load('catr/checkpoint_v3.pth', map_location='cpu')
#     vocab_len = config.vocab_size # 27300
#     old_len = checkv3['model']['mlp.layers.2.bias'].shape[0]
#     if vocab_len < old_len - 1000:
#         print('Checkpoint Vocab: %d, Model Vocab: %d, need adjusting dimension ...' % (old_len, vocab_len))
#         # keep the id mapping of special tokens: [PAD] [CLS] [SEP] [UNK]
#         idx = np.array([0] + list(range(1000, vocab_len+996)) + [101,102,100], dtype=int)
# #         print(len(idx))
#         # output layer from Transformer Decoder
#         checkv3['model']['mlp.layers.2.weight'] = checkv3['model']['mlp.layers.2.weight'][idx,:] 
#         checkv3['model']['mlp.layers.2.bias'] = checkv3['model']['mlp.layers.2.bias'][idx]
#         # embedding layer to Transformer Decoder
#         checkv3['model']['transformer.embeddings.word_embeddings.weight'] = checkv3['model']['transformer.embeddings.word_embeddings.weight'][idx,:]

#     model.load_state_dict(checkv3['model'], strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = coco.build_dataset_cub(config, mode='training')
    dataset_val = coco.build_dataset_cub(config, mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint from %s..." % config.checkpoint)
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1
    
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    tb_dir = 'catr/tensorboard/{0}_{1}_{2}'.format('cub', config.prefix, timestamp)
    tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging
    
    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        
#         print('\n=>Validation on COCO%s valset' % config.data_ver)
#         start_time = timeit.default_timer()
#         # validation
#         validation_loss = evaluate(model, criterion, data_loader_val, device, epoch, tbw)
#         tbw.add_scalar('Val/total_loss_epoch', validation_loss, epoch)
#         stop_time = timeit.default_timer()
#         print('[Epoch: %d, Val Loss: %.6f, Execution time: %.2f]' 
#                 % (epoch, validation_loss, (stop_time - start_time) / 60))
        
#         print('\n=>Validation on COCO%s trainset' % config.data_ver)
#         start_time = timeit.default_timer()
#         # validation
#         validation_loss = evaluate(model, criterion, data_loader_train, device, epoch, tbw)
#         tbw.add_scalar('Val/total_loss_epoch', validation_loss, epoch)
#         stop_time = timeit.default_timer()
#         print('[Epoch: %d, Val Loss: %.6f, Execution time: %.2f]' 
#                 % (epoch, validation_loss, (stop_time - start_time) / 60))
        
#         break
        
        print('\n=>Epoches %i, learning rate = %.7f' % (epoch, optimizer.param_groups[0]['lr']))
        start_time = timeit.default_timer()
        # training
        epoch_loss = train_one_epoch(
            image_encoder, model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm, tbw)
        lr_scheduler.step()
        tbw.add_scalar('Train/total_loss_epoch', epoch_loss, epoch)
        stop_time = timeit.default_timer()
        print('[Epoch: %d, Train Loss: %.6f, Execution time: %.2f]' 
                % (epoch, epoch_loss, (stop_time - start_time) / 60))

        print('\n=>Validation on CUB')
        start_time = timeit.default_timer()
        # validation
        validation_loss = evaluate(image_encoder, model, criterion, data_loader_val, device, epoch, tbw)
        tbw.add_scalar('Val/total_loss_epoch', validation_loss, epoch)
        stop_time = timeit.default_timer()
        print('[Epoch: %d, Val Loss: %.6f, Execution time: %.2f]' 
                % (epoch, validation_loss, (stop_time - start_time) / 60))
        
        # save checkpoint every epoch
#         if epoch % 5 == 4:

        if True:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, 'catr/checkpoints/' + config.prefix + '_cub_ep%02d.pth' % epoch)

        print()


if __name__ == "__main__":
    config = Config()
    main(config)
