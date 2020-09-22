import os
import logging

import cv2
import torch
import numpy as np
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
    def swap_channels(tensor):
        """Swaps channels of a given 3D/4D tensor

        Args:
            tensor (torch.Tensor): Tensor

        Returns:
            torch.Tensor: Tensor
        """
        if np.argmin(tensor.shape) != len(tensor.shape) - 1:
            if len(tensor.shape) == 3:
                return tensor.permute(1, 2, 0)
            elif len(tensor.shape) == 4:
                return tensor.permute(0, 2, 3, 1)
            else:
                raise Exception(f'Unsupported Tensor shape: {tensor.shape}')
        else:
            log.warning(f'Swap not necessary: {tensor.shape}')
        return tensor

    @staticmethod
    def overlay(img, mask):
        """Overlay the mask on img

        Args:
            img (np.array/torch.Tensor): Values between 0-255.
            mask (np.array/torch.Tensor): Values between 0-1. This will be overlaid as 'JET' colormap
        Returns:
            cam (np.array): Overlaid image, pixels between 0-255
        """
        if isinstance(img, torch.Tensor):
            img = img.data.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.data.numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img / 255)
        cam = np.clip((cam / np.max(cam)) * 255, 0, 255)
        return cam

    @staticmethod
    def save_image(img, filename):
        """Saves image to output/gradients directory

        Args:
            img (np.array): Pixel values 0-255
        """
        filepath = os.path.join('output', 'gradients', filename)
        if cv2.imwrite(filepath, img):
            log.info(f'Image saved at {filepath}.')

    @staticmethod
    def normalize(x):
        """Normalize array

        Args:
            x (np.array): numpy array, typically from any intermediate outputs

        Returns:
            np.array: normalized array, 0 mean, 0.1 std
        """
        x = x - np.mean(x)
        x = x / (np.std(x) + 1e-6)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        # x = np.uint8(x * 255)
        return x


if __name__ == '__main__':
    '''
    Load a pretrained Model
    '''
    cfg_from_file('configs/alexnet_32x32.yaml')
    device = get_target_device(cfg=cfg)
    # create model and load to device
    alexnet = model.AlexNet(cfg=cfg)
    alexnet = alexnet.to(device)
    if 'gpu' in cfg.DEVICE:
        alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=cfg.GPU)
    # Create optimizer
    optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # Load model from checkpoint
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
                                               batch_size=1,
                                               shuffle=True)
    '''
    Load a datapoint
    '''
    # Get a sample from the loader
    img, label = next(iter(train_loader))

    # TODO: Send this to CIFAR class, it is helpful there too
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    log.info(f'{img.shape}, {classes[label.data.numpy().item()]}')
    input_img = ImUtils.denorm(img,
                               mean=(0.491, 0.482, 0.446),
                               std=(0.247, 0.243, 0.261))[0].data.numpy() * 255
    input_img = input_img.transpose((1, 2, 0))
    input_img = cv2.cvtColor(
        input_img, cv2.COLOR_BGR2RGB)  # cv2 uses RGB format, rest use BGR
    ImUtils.save_image(input_img, 'input_image.png')

    '''
    Part 1. GradCam
    '''
    log.info('GradCam')
    grad_cam = GradCam(model=alexnet.module,
                        feature_module=alexnet.module.conv_feat,
                        target_layer_names=["12"])
    # If index=None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    mask = grad_cam(img, index=None)  # Mask is (IMG_SHAPE, IMG_SHAPE)
    overlaid_mask = ImUtils.overlay(input_img, mask)  # Overlays both arrays with colormap
    ImUtils.save_image(overlaid_mask, 'gradcam_overlay.png')

    '''
    Part 2. Guided Backpropagation + GradCam
    '''
    log.info('Guided Backprop + GradCam')
    gb_model = GuidedBackpropReLUModel(model=alexnet.module)
    # TODO: Figure out why explicit require_grad is needed
    # If index=None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    gb = gb_model(img.requires_grad_(True), index=None)  # This is gb = (3, IMG_SHAPE, IMG_SHAPE)
    gb = gb.transpose((1, 2, 0))  # Convert gb to  (IMG_SHAPE, IMG_SHAPE, 3)
    cam_mask = cv2.merge([mask, mask, mask])  # Converts MASK to (IMG_SHAPE, IMG_SHAPE, 3)
    gb = np.uint8(ImUtils.normalize(gb) * 255)
    cam_gb = np.uint8(ImUtils.normalize(cam_mask * gb) * 255)
    ImUtils.save_image(gb, 'gb.png')
    ImUtils.save_image(cam_gb, 'cam_gb.png')
    log.info('All done!')