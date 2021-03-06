from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer_s3 import condGANTrainer as trainer

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


# import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import pickle
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from catr.cfg_damsm_bert import Config ## add catr config file here
config  = Config()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "rb") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "rb") as f:
                print('Load from:', name)
                sentences = f.read().decode('utf8').split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    
#     filename = '../data/coco/data_dic.pickle'
#     with open(filename , 'wb') as f:
#         val_imgs_stats = pickle.dump(data_dic,f)
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    # change the batch size for scenario 3 #
    cfg.TRAIN.BATCH_SIZE = 6  
    ########################################
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME+'-s3-01_01_01_01_50', timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform,
                          max_length=config.max_position_embeddings, 
                          vocab=config.vocab)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    val_batch_size = 24
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    
    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader,dataloader_val, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            df = pd.read_csv('../IMG_TEXT_TXFMR_COCO_IS_Results.csv')
            dont_run_epochs = df['epoch'].values
#             dont_run_epochs
            paths = []
            counter = 0
            num_models = len(os.listdir('../output/coco_glu-gan2_2020_10_13_23_16_32/Model'))
            for gPath in sorted(os.listdir('../output/coco_glu-gan2_2020_10_13_23_16_32/Model'),reverse=True):
                if 'netG' in gPath and '.pth' in gPath:
                    e = int(gPath.split('_')[-1].replace('/','').split('.')[0])
                    print('---- model counter --- {}/{}'.format(counter,num_models))
                    if e not in dont_run_epochs:
                        full_netGpath = os.path.join('../output/coco_glu-gan2_2020_10_13_23_16_32/Model',gPath)
#                         print(full_netGpath)
                        algo.sampling(split_dir,full_netGpath)  # generate images for the whole valid dataset
                    counter+=1
        else:
            print('gen_example')
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
