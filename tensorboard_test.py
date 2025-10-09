import argparse
import datetime
import itertools
import more_itertools
import numpy as np
import PIL.Image
import torch
import torch.nn
import torch.autograd
import torch.utils.tensorboard as tb
import tensorboard
import tqdm
from torch.utils.tensorboard import SummaryWriter

import dataset_triplet
import softmax_basic

import random


writer = SummaryWriter()

for i in range(100):
    writer.add_scalar("Loss/train", random.random(), i)
