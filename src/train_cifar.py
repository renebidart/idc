import fire
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
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.models import resnet34
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

from models import ResNet50
from densenet import DenseNet


# Add the src directory for functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
sys.path.append(src_dir)

# import my functions:
from functions import*
from models import*

class Trainer(object):
    """ Just use a class to use python fire rather than argparse because its faster.
    """

    def __init__(self):
        pass

    
    def train_resnet50_cifar(self, epochs=200, n_models=8):
        # Based off: https://github.com/kuangliu/pytorch-cifar
        PATH = Path('/home/rene/data')
        save_path = PATH / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs)
        num_workers = 4
        batch_size = 180
        
        for i in range(n_models):
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                               valid_name='valid')
            model_name = 'ResNet50_'+str(i)
            model = ResNet50()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/2), gamma=0.1)

            best_acc, model = train_model(model, criterion, optimizer, scheduler,
                                             epochs, dataloaders, dataset_sizes)
            torch.save(model.state_dict(), str(save_path / model_name))

    def train_densenet_cifar(self, epochs=350, n_models=4, device="cuda:0", start_num=0):
        # Based off: https://github.com/kuangliu/pytorch-cifar
        start_num=int(start_num)
        PATH = Path('/home/rene/data')
        save_path = PATH / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs)
        num_workers = 4
        batch_size = 100
        
        for i in range(start_num, n_models+start_num):
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                               valid_name='valid')
            model_name = 'densenet_'+str(i)

            model = DenseNet(growthRate=12, depth=121, reduction=0.5,
                                    bottleneck=True, nClasses=10)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1)

            best_acc, model = train_model(model, criterion, optimizer, scheduler,
                                             epochs, dataloaders, dataset_sizes, device)
            torch.save(model.state_dict(), str(save_path / model_name))

def main():
    fire.Fire(Trainer)

if __name__ == '__main__':
    main()