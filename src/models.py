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


class Fusion2(nn.Module):
    ### only for 4 models
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list, num_input, num_output):
        super(Fusion2, self).__init__()
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.model4 = model_list[3]
        self.fc1 = nn.Linear(int(40), 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, int(10))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)

        x5 = torch.cat((x1, x2, x3, x4), 1)
        x6 = self.fc1(x5)
        x7 = self.relu(x6)
        out = self.fc2(x7)
        return out

class Fusion3(nn.Module):
    ### only for 3 models
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list, num_input, num_output):
        super(Fusion3, self).__init__()
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.fc1 = nn.Linear(int(30), 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, int(10))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)

        x5 = torch.cat((x1, x2, x3), 1)
        x6 = self.fc1(x5)
        x7 = self.relu(x6)
        out = self.fc2(x7)
        return out


class Fusion6(nn.Module):
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list, num_input, num_output):
        super(Fusion6, self).__init__()
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.model4 = model_list[3]
        self.model5 = model_list[4]
        self.model6 = model_list[5]

        self.fc1 = nn.Linear(int(60), 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, int(10))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x6 = self.model6(x)

        x8 = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        x8 = self.fc1(x8)
        x8 = self.relu(x8)
        out = self.fc2(x8)
        return out

class Fusion2More(nn.Module):
    ### only for 4 models
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list):
        super(Fusion2More, self).__init__()
        self.model1 = ResNet50Bottom(model_list[0])
        self.model2 = ResNet50Bottom(model_list[1])
        self.model3 = ResNet50Bottom(model_list[2])
        self.model4 = ResNet50Bottom(model_list[3])
        self.fc1 = nn.Linear(int(5120), 512)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, int(10))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)

        x5 = torch.cat((x1, x2, x3, x4), 1)
        x6 = self.relu(self.fc1_drop(self.fc1(x5)))
        out = self.fc2(x6)
        return out

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.squeeze(x)
        return x



##### USELESS STUFF?


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
        return out2

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





 #########.   OLD ONE WAS $ MODEL. HARDCODE TO #3??? POP The last fc layers howerever they're labelled.
class TriFusionMoreFC(nn.Module):
    ### only for 4 models
    """Take list of models, fuse their output into 2 classes"""
    def __init__(self, model_list, num_input, num_output, num):
        super(Fusion2, self).__init__()
        self.model1 = nn.Sequential(*list(model_list[0].features.children())[:-1])
        print(nn.Sequential(*list(model_list[0].features.children())[:-1]))
        self.model2 = model_list[1]
        self.model3 = model_list[2]
        self.fc1 = nn.Linear(int(40), 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, int(10))

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = torch.cat((x1, x2, x3, x4), 1)
        x6 = self.fc1(x5)
        x7 = self.relu(x6)
        out = self.fc2(x7)
        return out



################
'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())



# ???? 
# https://github.com/kuangliu/pytorch-cifar
# https://github.com/bamos/densenet.pytorch
