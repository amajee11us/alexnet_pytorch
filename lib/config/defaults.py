from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from easydict import EasyDict as edict

__C = edict()

cfg = __C
'''
General
'''
__C.NUM_WORKERS = 2

__C.GPU = [0]  # pass a set of GPUS for data-parallel

__C.PRINT_FREQUENCY = 10

__C.ARCH = "alexnet"

__C.EXP_NAME = "basic_224"

__C.DEVICE = "cpu"
'''
Training setting
'''
__C.TRAIN = edict()

__C.TRAIN.LEARNING_RATE = 0.01

__C.TRAIN.NUM_EPOCHS = 90

__C.TRAIN.LR_DECAY_STEP = 30

__C.TRAIN.GAMMA = 0.1

__C.TRAIN.BATCH_SIZE = 12

__C.TRAIN.IMAGE_SIZE = 224

__C.TRAIN.NUM_CLASSES = 1000

__C.TRAIN.DATASET = "imagenet"

__C.TRAIN.DATA_DIR = 'data/imagenet'

__C.TRAIN.DROPOUT_RATE = 0.5
'''
Testing Settings
'''
__C.TEST = edict()

__C.TEST.DATASET = "imagenet"

__C.TEST.DATA_DIR = 'data/imagenet'