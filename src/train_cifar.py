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
    
    def train_resnet50_cifar(self, epochs=200,  device="cuda:1", n_models=8):
        # Based off: https://github.com/kuangliu/pytorch-cifar
        PATH = Path('/media/rene/data/')
        save_path = PATH / 'cifar-10-batches-py/models'
        save_path.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs)
        num_workers = 4
        batch_size = 256

        for i in range(n_models):
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                               valid_name='valid')
            model_name = 'ResNet50_'+str(i)
            model = ResNet50()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=.05, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1)

            best_acc, model = train_model(model, criterion, optimizer, scheduler,
                                             epochs, dataloaders, dataset_sizes, device)
            torch.save(model.state_dict(), str(save_path / model_name))


    def train_densenet_cifar(self, epochs=200, n_models=4, device="cuda:0", start_num=0):
        # Based off: https://github.com/kuangliu/pytorch-cifar
        start_num=int(start_num)
        PATH = Path('/media/rene/data')
        save_path = PATH / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs)
        num_workers = 4
        batch_size = 50
        
        for i in range(start_num, n_models+start_num):
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                               valid_name='valid')
            model_name = 'densenet_'+str(i)
            model = DenseNet(growthRate=24, depth=121, reduction=0.5,
                                    bottleneck=True, nClasses=10)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1)

            best_acc, model = train_model(model, criterion, optimizer, scheduler,
                                             epochs, dataloaders, dataset_sizes, device)
            torch.save(model.state_dict(), str(save_path / model_name))


    def train_fusion(self, epochs1=100, epochs2=200, device="cuda:1"):
        epochs1, epochs2 = int(epochs1), int(epochs2)
        num_workers = 4

        PATH = Path('/media/rene/data/')
        save_path = PATH / 'cifar-10-batches-py/models'
        save_path.mkdir(parents=True, exist_ok=True)
        model_name_list = ['ResNet50_5', 'ResNet50_0', 'ResNet50_7', 'ResNet50_6']
        batch_size = 128

        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        # get all the models
        pretrained_model_list = []
        for i, model_name in enumerate(model_name_list):
            model = ResNet50()
            model = model.to(device)
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
            pretrained_model_list.append(model)

        model = Fusion2(pretrained_model_list, num_input=40, num_output=10)

        ######################  TRAIN LAST FEW LAYERS
        print('training last few layers')

        model_name = 'Fusion2_2s1_r2'
        for p in model.parameters():
            p.requires_grad = True

        for p in model.model1.parameters():
            p.requires_grad = False
        for p in model.model2.parameters():
            p.requires_grad = False
        for p in model.model3.parameters():
            p.requires_grad = False
        for p in model.model4.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
                                   dataloaders, dataset_sizes, device=device)
        torch.save(model.state_dict(), str(save_path / model_name))

        ########################   TRAIN ALL LAYERS
        model_name = 'Fusion2_2s2_r2'
        batch_size = 24
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.001, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                   dataloaders, dataset_sizes, device=device)

        torch.save(model.state_dict(), str(save_path / model_name))


    def train_fusion_more(self, epochs1=100, epochs2=150, device="cuda:1"):
        epochs1, epochs2 = int(epochs1), int(epochs2)
        num_workers = 4

        PATH = Path('/media/rene/data/')
        save_path = PATH / 'cifar-10-batches-py/models'
        save_path.mkdir(parents=True, exist_ok=True)
        model_name_list = ['ResNet50_5', 'ResNet50_0', 'ResNet50_7', 'ResNet50_6']
        batch_size = 128

        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        # get all the models
        pretrained_model_list = []
        for i, model_name in enumerate(model_name_list):
            model = ResNet50()
            model = model.to(device)
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))

            # remove last layers
            res50_conv = ResNet50Bottom(model)
            pretrained_model_list.append(res50_conv)


        model = Fusion2More(pretrained_model_list, num_input=40, num_output=10)

        ######################  TRAIN LAST FEW LAYERS
        print('training last few layers')

        model_name = 'Fusion2_2s1_more'
        for p in model.parameters():
            p.requires_grad = True

        for p in model.model1.parameters():
            p.requires_grad = False
        for p in model.model2.parameters():
            p.requires_grad = False
        for p in model.model3.parameters():
            p.requires_grad = False
        for p in model.model4.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
                                   dataloaders, dataset_sizes, device=device)
        torch.save(model.state_dict(), str(save_path / model_name))

        ########################   TRAIN ALL LAYERS
        model_name = 'Fusion2_2s2_more'
        batch_size = 32
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.001, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                   dataloaders, dataset_sizes, device=device)

        torch.save(model.state_dict(), str(save_path / model_name))






###########################. IDC.   ######################

    def train_fusion_IDC_morefc(self, epochs1=2, epochs2=3, device="cuda:0"):
        epochs1, epochs2 = int(epochs1), int(epochs2)
        num_workers = 4
        device="cuda:0"

        PATH = Path('/home/rene/data/')
        save_path = PATH /models
        save_path.mkdir(parents=True, exist_ok=True)
        model_name_list = ['ResNet50_2', 'ResNet50_7', 'ResNet50_1', 'ResNet50_0']
        batch_size = 128

        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')
       
        # get all the models
        pretrained_model_list = []
        for i, model_name in enumerate(model_name_list):
            model = ResNet50()
            model = model.to(device)
            model.load_state_dict(torch.load(os.path.join(save_path, model_name)))
            pretrained_model_list.append(model)

        model = TriFusionMoreFC(pretrained_model_list, num_input=40, num_output=10)

        ######################  TRAIN LAST FEW LAYERS
        print('training last few layers')

        model_name = 'Fusion2_2s1'
        for p in model.parameters():
            p.requires_grad = True

        for p in model.model1.parameters():
            p.requires_grad = False
        for p in model.model2.parameters():
            p.requires_grad = False
        for p in model.model3.parameters():
            p.requires_grad = False
        for p in model.model4.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.05, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
                                   dataloaders, dataset_sizes, device=device)
        torch.save(model.state_dict(), str(save_path / model_name + '_morefc'))

        ########################   TRAIN ALL LAYERS
        model_name = 'Fusion2_2s2'
        batch_size = 6
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
                                   dataloaders, dataset_sizes, device=device)

        torch.save(model.state_dict(), str(save_path / model_name + '_morefc'))



def main():
    fire.Fire(Trainer)

if __name__ == '__main__':
    main()