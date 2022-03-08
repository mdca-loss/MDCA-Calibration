import logging
from .logger import *
from .eval import *
from .argparser import parse_args
from .misc import *
from .earlystopper import EarlyStopping
import torch
import os
import shutil

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f"model_best.pth"))

def create_save_path(args):

    ans_str = f"_{args.model}_{args.loss}"

    if args.loss == "FLSD":
        ans_str += f"_gamma={args.gamma}"
        return ans_str
    
    if "focal_loss" in args.loss or "FL" in args.loss:
        ans_str += f"_gamma={args.gamma}"
    
    if "LS" in args.loss:
        ans_str += f"_alpha={args.alpha}"

    if "MDCA" in args.loss:
        ans_str += f"_beta={args.beta}"
        return ans_str

    if "DCA" in args.loss or "MMCE" in args.loss:
        ans_str += f"_beta={args.beta}"

    return ans_str
