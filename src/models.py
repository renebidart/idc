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


class Fusion1(nn.Module):
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list):
        super(Fusion1, self).__init__()
        self.model_list = model_list
        self.num_input = int(len(self.model_list)*2)
        self.fc = nn.Linear(self.num_input, 2)

    def forward(self, x):
        outputs = []
        for model in model_list:
            outputs.append(model(x))
        x = torch.cat(outputs, 1)
        out = self.fc(x)
        return out
    
class Fusion1Hardcode(nn.Module):
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list):
        super(Fusion1Hardcode, self).__init__()
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.fc = nn.Linear(6, 2)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = torch.cat((x1, x2, x3), 1)
        out = self.fc(x4)
        return out

class Fusion2Hardcode(nn.Module):
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list):
        super(Fusion2Hardcode, self).__init__()
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = torch.cat((x1, x2, x3), 1)
        x5 = self.fc1(x4)
        x6 = self.relu(x5)
        out = self.fc2(x5)
        return out

class Fusion3(nn.Module):
    """Throw away final classification layer before the concat is done.
    More parameters are in merged section of model: moving towards to a single model 
    Pop last (classification) layer of models, concat output->256 node fc->output(2 class)
    """
    def __init__(self, model_list):
        super(Fusion1, self).__init__()
        new_models = []
        model_ft.fc.in_features = 0
        for model in new_models:
            model_ft.fc.in_features += model.fc.in_features
            modules = list(model.children())[:-1]      # delete the last fc layer.
            model = nn.Sequential(*modules)
            new_models.append(model)        
        
        self.model_list = new_models
        self.num_input = int(len(self.model_list)*2)
        self.fc1 = nn.Linear(num_input, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        outputs = []
        for model in model_list:
            outputs.append(model(x))
        x1 = torch.cat(outputs, 1)
        x2 = F.relu(self.fc1(x))
        out = self.fc2(x2)
        return out