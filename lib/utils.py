from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import shutil
import os
import os.path as osp

def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 (say) epochs"""
    lr = cfg.TRAIN.LEARNING_RATE * (0.1 ** (epoch // cfg.TRAIN.LR_DECAY_STEP))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")

def get_output_tb_dir(cfg):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join('output', 'tensorboard',cfg.ARCH +  "_" + cfg.EXP_NAME))
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir

def get_output_dir():
    pass