import torch
import logging
import numpy as np
import matplotlib
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image

import os, sys
import argparse
import pprint

# import module classes
from dataset import imagenet
# from dataset import mini_imagenet
from dataset import cifar10

from lib import model
from lib.engine import train, validate
from lib.utils import *
from lib.config.conf import cfg_from_file
from lib.config.conf import __C as cfg

OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet/CIFAR10 Training')

    # General parser
    parser.add_argument('-c',
                        '--config_file',
                        dest='config_file',
                        default='configs/alexnet_224x224.yaml',
                        help='model architecture (default: alexnet)')
    parser.add_argument('--dataset',
                        default='cifar',
                        help='Choose from imagenet/cifar')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='alexnet',
                        help='model architecture (default: alexnet)')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--opt_level',
                        default="O1",
                        type=str,
                        help="Choose which accuracy to train. (default: 'O1')")
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Get configuration
    log.info("Reading config from file: {}".format(args.config_file))
    cfg_from_file(args.config_file)
    pprint.PrettyPrinter(indent=4).pprint(cfg)
    # Select appropriate device
    device = get_target_device(cfg)
    log.info(f'Using {device} for execution.')
    
    '''
    Model/Optimizer setup
    '''
    tbwriter = SummaryWriter(log_dir=get_output_tb_dir(cfg))
    seed = torch.initial_seed()
    log.info("Using Seed : {}".format(seed))

    # create model and load to device
    alexnet = model.AlexNet(cfg=cfg)
    alexnet = alexnet.to(device)
    log.info(alexnet)
    if 'cuda' in device:
        alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=cfg.GPU)

    # TODO: Add code to resume from a checkpoint
    # Create optimizer
    optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # Create an LR scheduler, Multiply rate by 0.1 every LR_STEP
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    '''
    Load and prepare datasets
    Supported CIFAR10/Imagenet.
    TODO: Add more
    '''
    if args.dataset == 'cifar':
        train_dataset = cifar10.CIFAR10Dataset(
            data_path=cfg.TRAIN.DATA_DIR,
            split='train',
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32),  # square image transform
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ]),
            download=True)

        val_dataset = cifar10.CIFAR10Dataset(
            data_path=cfg.TEST.DATA_DIR,
            split='val',
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32),  # square image transform
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ]),
            download=True)
    elif args.dataset == 'imagenet':
        train_dataset = imagenet.ImageNetDataset(
            data_path=cfg.TRAIN.DATA_DIR,
            split='train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(cfg.TRAIN.IMAGE_SIZE),
                #transforms.CenterCrop(cfg.TRAIN.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

        val_dataset = cifar10.CIFAR10Dataset(
            data_path=cfg.TEST.DATA_DIR,
            split='val',
            transform=transforms.Compose([
                # transforms.CenterCrop(cfg.TRAIN.IMAGE_SIZE),
                # transforms.Resize(cfg.TRAIN.IMAGE_SIZE, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

    log.info(
        "Dataset created:\n\tTRAIN images : {}\n\tVAL images: {}\n\tNUM_CLASSES: {}"
        .format(len(train_dataset), len(val_dataset), cfg.TRAIN.NUM_CLASSES))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=cfg.NUM_WORKERS,
                                               drop_last=True,
                                               batch_size=cfg.TRAIN.BATCH_SIZE)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True,
        batch_size=cfg.TRAIN.BATCH_SIZE  #TODO: See if this works and update
    )
    '''
    Start Training with specified config.
    '''
    # Set the initial param for best accuracy to beat
    best_acc1 = 0

    # Train over the dataset
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # adjust_learning_rate(optimizer, epoch, cfg)
        # step on the learning-rate
        lr_scheduler.step()

        # train one epoch on the target device
        train(train_loader, alexnet, criterion, optimizer, epoch, cfg,
              tbwriter)

        # Get the top1 accuracy from the validation set
        acc1 = validate(val_loader, alexnet, criterion, cfg, tbwriter)

        check_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': cfg.ARCH,
                'state_dict': alexnet.state_dict(),
                'best_acc1': best_acc1,
                'optimzer': optimizer.state_dict(),
                'lr': get_lr(optimizer)
            },
            is_best=check_best)


if __name__ == "__main__":
    main()