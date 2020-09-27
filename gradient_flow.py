import argparse
import logging
import os
import pprint
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter

from dataset import cifar10, imagenet
from lib.config.conf import __C as cfg
from lib.config.conf import cfg_from_file
from lib.dataset_factory import build_dataset
from lib.engine import resume_from_ckpt
from lib.gradcam import GradCam, GuidedBackpropReLUModel
from lib.models import factory
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import *

# Create logger object
log = Logger(cfg)

# This path will store gradient visualization cache.
DUMP_DIR = os.path.join(cfg.OUTPUT_DIR, 'gradients')
if os.path.exists(DUMP_DIR):
    shutil.rmtree(DUMP_DIR)
os.mkdir(DUMP_DIR)


# TODO: Move ImUtils to some generic module
class ImUtils(object):
    @staticmethod
    def denorm(img_tensor, mean, std):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be denormalized.
        Returns:
            new Tensor: DeNormalized image.
        """
        tensor = img_tensor.clone()
        assert len(mean) == len(std)
        if len(tensor.shape) == 4:
            # Across N
            for sample in tensor:
                # Across channel
                n_c = sample.shape[0]
                if n_c != len(mean):
                    log.error(sample.shape)
                    raise Exception(
                        f'Tensor has {n_c} channels, but mean/std has {len(mean)}'
                    )
                for t, m, s in zip(sample, mean, std):
                    t.mul_(s).add_(m)
        return tensor

    @staticmethod
    def save_as_fig(x_batch,
                    y_batch,
                    features_batch,
                    classes=None,
                    figname=None):
        """Creates overlay with x_batch and mask_batch.
        Plots this into a figure, which is saved as a PNG file.

        Args:
            x_batch (np.array): Array NCHW
            y_batch (np.array): Array [N]
            features_batch (np.array): Array NHW, feature maps of each image in x_batch
            classes (np.array): Tuple of strings corresponding to labels (optional)
        """
        try:
            nrows = 2
            ncols = x_batch.shape[0]
            figsize = (ncols, nrows)  # (width, height)
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
            col = 0
            for x, y, f in zip(x_batch, y_batch, features_batch):
                # Apply colormap if mask is single channel
                if len(f.shape) == 2:
                    # remove the alpha channel from RGBA
                    f = plt.cm.jet(f)[..., :3]
                    # overlay and normalize
                    f = f + np.float32(x)
                    f = np.clip(f / np.max(f), 0, 1)

                elif len(f.shape) == 3:
                    # normalize and clip
                    f -= np.mean(f)
                    f /= (np.std(f) + 1e-6)
                    f *= 0.1
                    f += 0.5
                    f = np.clip(f, 0, 1)

                f = np.uint8(np.clip(f * 255, 0, 255))

                ax[0][col].set_title(classes[y.item(
                )]) if classes is not None else ax[0][col].set_title(y)
                ax[0][col].imshow(x)
                ax[0][col].set_xticks([])
                ax[0][col].set_yticks([])

                ax[1][col].imshow(f)
                ax[1][col].set_xticks([])
                ax[1][col].set_yticks([])

                col += 1
            fig.tight_layout()
            plt.savefig(
                os.path.join(
                    DUMP_DIR,
                    'saved_figure.png' if figname is None else figname),
                bbox_inches='tight',
                pad_inches=0,
            )
            log.info('Saved figure.')
        except Exception as e:
            log.error(f'Exception in save: {e}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet/CIFAR10 Training')

    # General parser
    parser.add_argument('-c',
                        '--config_file',
                        dest='config_file',
                        default='configs/alexnet_32x32.yaml',
                        help='model architecture (default: alexnet)')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Get configuration
    cfg_from_file(args.config_file)
    cfg.OUTPUT_DIR = get_output_ckpt_dir(cfg)

    log.info("Reading config from file: {}".format(args.config_file))

    log.info(pprint.PrettyPrinter(indent=4).pprint(cfg))
    # Select appropriate device
    device = get_target_device(cfg)
    log.info(f'Using {device} for execution.')

    # Model/Optimizer setup
    tbwriter = SummaryWriter(log_dir=get_output_tb_dir(cfg))
    if args.seed is None:
        seed = torch.initial_seed()
    else:
        seed = torch.manual_seed(args.seed)
    log.info("Using Seed : {}".format(seed))

    # create model and load to device
    alexnet = factory.build_model(cfg)
    log.info(alexnet)

    # Create optimizer
    optimizer = build_optimizer(cfg, alexnet)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # Resume from a checkpoint
    if not args.resume == None:
        resume_from_ckpt(args.resume, alexnet, optimizer)

    np.random.seed(0)
    torch.manual_seed(0)

    '''
    Prepare data batch for inference/saving data
    '''
    # Prepare dataloader
    train_loader = build_dataset(cfg, split='train')
    # Load a datapoint
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch[:8], y_batch[:8]
    # Denormalize each image. This is needed to save images ONLY.
    denormed_x_batch = ImUtils.denorm(x_batch,
                                      mean=cfg.TRAIN.DATASET.NORM_MEAN,
                                      std=cfg.TRAIN.DATASET.NORM_STD_DEV)
    x_batch_npy = denormed_x_batch.data.numpy().transpose((0, 2, 3, 1))
    y_batch_npy = y_batch.data.numpy()

    '''
    Part 1. GradCam
    '''
    log.info('GradCam')
    gradcam = GradCam(model=alexnet.module,
                      feature_module=alexnet.module.conv_feat,
                      target_layer_names=["12"])
    # Evaluate CAMs over batch and convert to npy arrays. Mask default is npy array
    gradcam_fmap_npy = gradcam(x_batch, index=None)
    log.info(f'INPUT: {x_batch_npy.shape}, MASK: {gradcam_fmap_npy.shape}')

    # Save plots in a figure.
    ImUtils.save_as_fig(x_batch_npy,
                        y_batch_npy,
                        gradcam_fmap_npy,
                        classes = None,
                        figname='gradcam.png')
    '''
    Part 2. Guided Backpropagation
    '''
    log.info('Guided Backprop')
    guided = GuidedBackpropReLUModel(model=alexnet.module)
    # Compute Guided backprop. Returns (N, H, W, C)
    gb_fmap_npy = guided(x_batch.requires_grad_(True), index=None).transpose((0, 2, 3, 1))
    log.info(f'INPUT: {x_batch_npy.shape}, MASK: {gb_fmap_npy.shape}')
    # Save plots in a figure.
    ImUtils.save_as_fig(x_batch_npy,
                        y_batch_npy,
                        gb_fmap_npy,
                        classes = None,
                        figname='guided_backprop.png')

    '''
    Part 3. Guided Backpropagation + GradCam
    '''
    log.info('Guided Backprop + GradCam')
    # GradCam gives 1 channel, GuidedBackprop gives 3 channels (since equal to input ch)
    cam_plus_gb_npy = np.stack([gradcam_fmap_npy for _ in range(3)], axis=-1) * gb_fmap_npy
    log.info(f'INPUT: {x_batch_npy.shape}, MASK: {cam_plus_gb_npy.shape}')
    # Save plots in a figure.
    ImUtils.save_as_fig(x_batch_npy,
                        y_batch_npy,
                        cam_plus_gb_npy,
                        classes = None,
                        figname='guided_backprop_gradcam.png')

if __name__ == "__main__":
    main()
