import os
import logging

import cv2
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torchvision.transforms as transforms

from lib import model
from lib.engine import resume_from_ckpt
from dataset import cifar10, imagenet
from lib.config.conf import cfg_from_file
from lib.config.conf import __C as cfg
from lib.utils import get_target_device
from lib.gradcam import GradCam, GuidedBackpropReLUModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ImUtils(object):
    @staticmethod
    def denorm(tensor, mean, std):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        Returns:
            Tensor: DeNormalized image.
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    @staticmethod
    def norm(tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    @staticmethod
    def save_as_fig(x_batch, y_batch, mask_batch, classes):
        """Save all as a plot
        TODO: Fill this up
        Args:
            x_batch ([type]): [description]
            y_batch ([type]): [description]
            mask_batch ([type]): [description]
            classes ([type]): [description]
        """
        try:
            nrows = 2
            ncols = x_batch.shape[0]
            figsize = (ncols, nrows)
            f, ax = plt.subplots(nrows, ncols, figsize=figsize)
            col = 0
            for x, y, m in zip(x_batch, y_batch, mask_batch):
                heatmap = cv2.applyColorMap(np.uint8(255 * m),
                                            cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.float32(x)
                cam = np.uint8(np.clip((cam / np.max(cam)) * 255, 0, 255))

                ax[0][col].set_title(classes[y.data.numpy().item()])
                ax[0][col].imshow(x)
                ax[0][col].set_xticks([])
                ax[0][col].set_yticks([])

                ax[1][col].imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
                ax[1][col].set_xticks([])
                ax[1][col].set_yticks([])

                col += 1
            f.tight_layout()
            plt.savefig(
                os.path.join(DUMP_DIR, 'gradcam_output.png'),
                bbox_inches='tight',
                pad_inches=0,
            )
            log.info('Saved figure.')
        except Exception as e:
            log.error(f'Exception in save: {e}')


if __name__ == '__main__':
    '''
    Config/Output setup
    '''
    cfg_from_file('configs/alexnet_32x32.yaml')
    device = get_target_device(cfg=cfg)
    DUMP_DIR = os.path.join(cfg.OUTPUT_DIR, 'gradients')
    if os.path.exists(DUMP_DIR):
        shutil.rmtree(DUMP_DIR)
    os.mkdir(DUMP_DIR)

    np.random.seed(0)
    torch.manual_seed(0)
    '''
    Load a pretrained Model
    '''
    alexnet = model.AlexNet(cfg=cfg)
    alexnet = alexnet.to(device)
    if 'gpu' in cfg.DEVICE:
        alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=cfg.GPU)
    optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=0.0001)
    resume_from_ckpt('output/alexnet_basic_cifar_32/model_best.pth', alexnet,
                     optimizer)
    '''
    Prepare dataloader
    '''
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(cfg.TRAIN.IMAGE_SIZE),  # square image transform
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x for x in [0.491, 0.482, 0.446]],
                             std=[x for x in [0.247, 0.243, 0.261]])
    ])
    cifar10_data = cifar10.CIFAR10Dataset('data/cifar10',
                                          'train',
                                          transform=transformations)
    train_loader = torch.utils.data.DataLoader(dataset=cifar10_data,
                                               batch_size=16,
                                               shuffle=True)
    '''
    Load a datapoint
    '''
    # TODO: Send this to CIFAR class, it is helpful there too
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    x_batch, y_batch = next(iter(train_loader))
    '''
    Part 1. GradCam
    '''
    log.info('GradCam')
    gradcam = GradCam(model=alexnet.module,
                      feature_module=alexnet.module.conv_feat,
                      target_layer_names=["12"])
    # Evaluate CAMs over batch
    mask_batch = gradcam(x_batch, index=None)
    log.info(f'INPUT: {x_batch.shape}, MASK: {mask_batch.shape}')

    # Denormalize each image. This is needed to save images ONLY.
    for each in x_batch:
        ImUtils.denorm(each,
                       mean=[0.491, 0.482, 0.446],
                       std=[0.247, 0.243, 0.261])
    x_batch_npy = x_batch.data.numpy().transpose((0, 2, 3, 1))
    # Save plots in a figure.
    ImUtils.save_as_fig(x_batch_npy, y_batch, mask_batch, classes)
    exit(0)
    '''
    Part 2. Guided Backpropagation + GradCam [NOT IMPLEMENTED IN BATCH]
    '''
    # log.info('Guided Backprop + GradCam')
    # gb_model = GuidedBackpropReLUModel(model=alexnet.module)
    # # TODO: Figure out why explicit require_grad is needed
    # # If index=None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # gb = gb_model(img.requires_grad_(True), index=None)  # This is gb = (3, IMG_SHAPE, IMG_SHAPE)
    # gb = gb.transpose((1, 2, 0))  # Convert gb to  (IMG_SHAPE, IMG_SHAPE, 3)
    # cam_mask = cv2.merge([mask, mask, mask])  # Converts MASK to (IMG_SHAPE, IMG_SHAPE, 3)
    # gb = np.uint8(ImUtils.normalize(gb) * 255)
    # cam_gb = np.uint8(ImUtils.normalize(cam_mask * gb) * 255)
    # ImUtils.save_image(gb, 'gb.png')
    # ImUtils.save_image(cam_gb, 'cam_gb.png')
    # log.info('All done!')