# Derived from implementation in PyTorch by jacobgil
# Link to source repo: https://github.com/jacobgil/pytorch-grad-cam
# import sys
# sys.path.insert(0, '../dataset')
import logging
import argparse

import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Function

from .model import AlexNet
from .utils import get_target_device
from .engine import resume_from_ckpt
from .config.conf import cfg_from_file
from .config.conf import __C as cfg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
device = get_target_device(cfg=cfg)

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module,
                                                  target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.model = model.to(device)

        self.extractor = ModelOutputs(self.model, self.feature_module,
                                      target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, index=None):

        features, output = self.extractor(input_img.to(device))

        # Evaluate N
        n_samples = output.shape[0]

        if index == None:
            index = torch.argmax(output, dim=1)

        # One_hot of size N x num_classes
        one_hot = torch.zeros(output.size(), dtype=torch.float32)
        for sample_id, pred in enumerate(index):
            one_hot[sample_id][pred] = 1

        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output, dim=1)

        # Compute gradients
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(gradient=torch.ones([n_samples]), retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]

        # features[-1] OR target is (n_samples, C, H, W) of avgpool
        target = features[-1]
        weights = torch.mean(grads_val, dim=(2, 3))

        # needed cam of size (N, H, W) of the avgpool layer
        H, W = target.shape[2:]
        cams = torch.zeros(n_samples, H, W, dtype=torch.float32)
        for n in range(n_samples):
            for i, w in enumerate(weights[n]):
                cams[n] += w * target[n, i, :, :]

        # apply ReLU
        torch.nn.ReLU(inplace=True)(cams)

        # Convert to npy arrays as need to resize
        cams = cams.data.numpy()

        # Resize each cam to input_img shape
        cams_batch = []
        for cam in cams:
            cam = cv2.resize(cam, input_img.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams_batch.append(cam)

        cams_batch = np.array(cams_batch)
        return cams_batch


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(input_img.size()).type_as(input_img), grad_output,
                positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model = model.to(device)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, index=None):
        output = self.forward(input_img.to(device))

        # Evaluate N
        n_samples = output.shape[0]

        if index == None:
            index = torch.argmax(output, dim=1)

        # one_hot of size N x num_classes
        one_hot = torch.zeros(output.size(), dtype=torch.float32)
        for sample_id, pred in enumerate(index):
            one_hot[sample_id][pred] = 1

        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output, dim=1)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(gradient=torch.ones([n_samples]), retain_graph=True)

        output = input_img.grad.data.numpy()

        return output
