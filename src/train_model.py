import os
import sys
import glob
import shutil
import random
import pickle
import numpy as np
from PIL import Image
import time
import copy
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.models import resnet34
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler



# Add the src directory for functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
print(src_dir)
sys.path.append(src_dir)

# import my functions:
from functions import*

# Set it to use GPU1
torch.cuda.set_device(1)
print(torch.cuda.is_available())
print(torch.cuda.current_device())