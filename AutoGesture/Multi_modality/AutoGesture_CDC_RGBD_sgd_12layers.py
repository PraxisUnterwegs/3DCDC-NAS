import argparse
import time
import os
import numpy as np
from tqdm import tqdm
import random
# import pprint

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torchvision
from collections import namedtuple

Genotype_Net = namedtuple('Genotype', 'normal8 normal_concat8 normal16 normal_concat16 normal32 normal_concat32')

PRIMITIVES = [
    'none',
    'skip_connect',
    'TCDC06_3x3x3',
    'TCDC03avg_3x3x3',
    'conv_1x3x3',
    'TCDC06_3x1x1',
    'TCDC03avg_3x1x1',
]

# AutoGesture_3Branch_CDC_RGB_New
genotype_RGB = Genotype_Net(
    normal8=[('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
             ('skip_connect', 1), ('TCDC06_3x1x1', 2), ('skip_connect', 3)], normal_concat8=range(2, 6),
    normal16=[('TCDC06_3x1x1', 1), ('skip_connect', 0), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 2), ('TCDC06_3x3x3', 3),
              ('TCDC06_3x1x1', 1), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x3x3', 3)], normal_concat16=range(2, 6),
    normal32=[('TCDC03avg_3x3x3', 1), ('skip_connect', 0), ('conv_1x3x3', 1), ('conv_1x3x3', 0), ('TCDC06_3x1x1', 1),
              ('skip_connect', 2), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0)], normal_concat32=range(2, 6))

# Old --> save as '2'
# genotype_RGB = Genotype_Net(normal8=[('skip_connect', 1), ('TCDC03avg_3x3x3', 0), ('skip_connect', 2), ('TCDC06_3x3x3', 0), ('TCDC03avg_3x3x3', 3), ('TCDC03avg_3x1x1', 0), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0)], normal_concat8=range(2, 6), normal16=[('TCDC06_3x1x1', 1), ('TCDC06_3x1x1', 0), ('TCDC03avg_3x3x3', 0), ('conv_1x3x3', 2), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x3x3', 0), ('TCDC06_3x3x3', 4), ('TCDC03avg_3x1x1', 3)], normal_concat16=range(2, 6), normal32=[('TCDC06_3x3x3', 0), ('TCDC06_3x1x1', 1), ('conv_1x3x3', 2), ('TCDC03avg_3x3x3', 0), ('TCDC03avg_3x3x3', 3), ('TCDC06_3x1x1', 0), ('TCDC06_3x1x1', 3), ('TCDC03avg_3x3x3', 4)], normal_concat32=range(2, 6))


# AutoGesture_3Branch_CDC_Depth_New
genotype_Depth = Genotype_Net(
    normal8=[('conv_1x3x3', 1), ('TCDC06_3x1x1', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1),
             ('skip_connect', 0), ('conv_1x3x3', 2), ('skip_connect', 1)], normal_concat8=range(2, 6),
    normal16=[('TCDC06_3x1x1', 1), ('conv_1x3x3', 0), ('TCDC06_3x3x3', 1), ('skip_connect', 2), ('TCDC06_3x1x1', 3),
              ('conv_1x3x3', 0), ('TCDC06_3x1x1', 1), ('conv_1x3x3', 3)], normal_concat16=range(2, 6),
    normal32=[('TCDC03avg_3x3x3', 1), ('TCDC06_3x1x1', 0), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x1x1', 1),
              ('TCDC03avg_3x1x1', 1), ('TCDC06_3x3x3', 3), ('TCDC03avg_3x1x1', 4), ('TCDC03avg_3x3x3', 2)],
    normal_concat32=range(2, 6))

# Old --> save as '2'
# genotype_Depth = Genotype_Net(normal8=[('skip_connect', 1), ('TCDC03avg_3x3x3', 0), ('skip_connect', 0), ('TCDC03avg_3x1x1', 2), ('conv_1x3x3', 3), ('TCDC06_3x3x3', 1), ('skip_connect', 1), ('TCDC03avg_3x1x1', 2)], normal_concat8=range(2, 6), normal16=[('TCDC03avg_3x3x3', 1), ('skip_connect', 0), ('TCDC03avg_3x3x3', 1), ('conv_1x3x3', 0), ('TCDC03avg_3x3x3', 3), ('TCDC03avg_3x1x1', 2), ('TCDC06_3x3x3', 4), ('TCDC06_3x3x3', 2)], normal_concat16=range(2, 6), normal32=[('TCDC03avg_3x1x1', 1), ('conv_1x3x3', 0), ('skip_connect', 1), ('TCDC06_3x3x3', 2), ('TCDC06_3x3x3', 2), ('TCDC06_3x3x3', 3), ('TCDC06_3x3x3', 3), ('TCDC06_3x1x1', 1)], normal_concat32=range(2, 6))


Genotype_Con_Unshared = namedtuple('Genotype', 'Low_Connect Mid_Connect High_Connect')

# For stride = 2 or stride = 1
PRIMITIVES_3x1x1 = [
    'none',
    'TCDC06_3x1x1',
    'TCDC03avg_3x1x1',
    'conv_3x1x1',
]

# For stride = 4
PRIMITIVES_5x1x1 = [
    'none',
    'TCDC06_5x1x1',
    'TCDC03avg_5x1x1',
    'conv_5x1x1',
]

# Connection_unshared
# epoch 15
genotype_con_unshared = Genotype_Con_Unshared(
    Low_Connect=['TCDC03avg_3x1x1', 'none', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC06_3x1x1', 'none', 'none',
                 'conv_3x1x1', 'none', 'conv_5x1x1', 'none', 'none', 'TCDC06_3x1x1', 'TCDC06_3x1x1', 'TCDC06_3x1x1',
                 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1'],
    Mid_Connect=['TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1', 'TCDC06_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1',
                 'conv_3x1x1', 'TCDC03avg_3x1x1', 'conv_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_3x1x1',
                 'TCDC06_3x1x1', 'TCDC06_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1'],
    High_Connect=['TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_3x1x1',
                  'TCDC06_5x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1', 'conv_5x1x1', 'none', 'TCDC03avg_3x1x1',
                  'TCDC03avg_3x1x1', 'none', 'TCDC06_3x1x1', 'TCDC06_5x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_3x1x1'])

# epoch 16
# genotype_con_unshared = Genotype_Con_Unshared(Low_Connect=['TCDC06_3x1x1', 'none', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'none', 'none', 'none', 'conv_3x1x1', 'none', 'none', 'none', 'none', 'none', 'TCDC03avg_3x1x1', 'TCDC06_3x1x1', 'none', 'conv_3x1x1', 'TCDC03avg_3x1x1'], Mid_Connect=['TCDC03avg_3x1x1', 'none', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'none', 'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_3x1x1', 'conv_3x1x1', 'TCDC06_5x1x1', 'TCDC03avg_3x1x1', 'TCDC06_3x1x1'], High_Connect=['TCDC06_3x1x1', 'TCDC06_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'conv_5x1x1', 'TCDC06_3x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'conv_5x1x1', 'conv_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1', 'none', 'none', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1'])


# Connection_shared
# epoch 15
# genotype_con_shared = Genotype_Con_Shared(Shared_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'])

genotype_con_shared = Genotype_Con_Unshared(
    Low_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1',
                 'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1',
                 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'],
    Mid_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1',
                 'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1',
                 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'],
    High_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1',
                  'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1', 'TCDC06_3x1x1',
                  'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'])


# epoch 16
# genotype_con_shared = Genotype_Con_Unshared(Low_Connect=['TCDC03avg_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'conv_5x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'none', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1'], Mid_Connect=['TCDC03avg_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'conv_5x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'none', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1'], High_Connect=['TCDC03avg_3x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'conv_5x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'none', 'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1'])
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Module()的目的实际上是为了方便训练做出选择（有与旋律模型 or 无预训练模型）
# The purpose of Module() is actually to facilitate the choice of training(with pre-trained model or without pre-training model)
def Module(args, device=torch.device('cuda:0')):
    print("load from AutoGesture_RGBD_Con_shared_DiffChannels")
    from models.AutoGesture_RGBD_searched_12layers_DiffChannels import AutoGesture_RGBD_12layers as Auto_RGBD_Diff

    model = Auto_RGBD_Diff(args.init_channels8, args.init_channels16, args.init_channels32, args.num_classes,
                           args.layers, genotype_RGB, genotype_Depth,
                           genotype_con_unshared)
    if args.init_model:
        # model.classifier = torch.nn.Linear(6144, 27)
        params = torch.load(args.init_model, map_location=device)
        try:
            model.load_state_dict(params)
            print('Load state dict...')
        except:
            print('Load state dict...')
            new_state_dict = OrderedDict()
            for k, v in params.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        if args.pretrain:
            print('init.xavier_normal_(model.classifier.weight)')
            from torch.nn import init
            model.classifier = torch.nn.Linear(6144, 249)
            # model.classifier = torch.nn.Linear(6144, 249)
            init.xavier_normal_(model.classifier.weight)
    print('Load module Finished')
    print('=' * 20)
    if torch.cuda.is_available():
        return torch.nn.DataParallel(model).cuda() if len(args.gpu_ids) > 1 else model.cuda()
    return torch.nn.DataParallel(model).cpu()
    # return value is model object
    # torch.nn.DataParallel used to Multi-GPU parallel training
