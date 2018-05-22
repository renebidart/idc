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
from torchvision import models

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# Add the src directory for functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
sys.path.append(src_dir)

# import my functions:
from functions import*
from models import*




class Trainer(object):
    """Everything is hardcoded. Just faster to copy paste.

    Given the amount of variance between hyperparameters for all the different models, each run will just be a function here. 
    For another run, add new function. Don't delete/clean up to make it reproducable

    Same thing is duplicated all the time for the methods. Can just put it into init with fire?
    !!! Self makes the variables specefic to an instance, otherwise they're class variables for all instances
    """

    def __init__(self, GPU_NUM=0, sample=False, PATH = '/media/rene/Data/data/idc'):
        num_workers = 6

        self.PATH = PATH
        if sample:
            self.PATH = self.PATH+'sample'
        self.save_path = os.path.join(PATH, 'models')
    

    def train_inceptionv3(self, epochs = 10, batch_size=32, size=299):
        dataloaders, dataset_sizes = make_batch_gen(self.PATH, batch_size, num_workers, valid_name='valid', size=size)
        model_name = 'inception_v3'

        model_ft = inception_v3(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        best_acc, model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                 epochs, dataloaders, dataset_sizes)
        torch.save(model_ft.state_dict(), os.path.join(save_path, model_name))


    def train_resnext101_fai(self, batch_size=32, size=224):
        arch = resnext101
        tfms = tfms_from_model(arch, size, aug_tfms=transforms_top_down, max_zoom=1)
        data = ImageClassifierData.from_paths(self.PATH, tfms=tfms, bs=batch_size)
        learn = ConvLearner.pretrained(arch, data, precompute=False)

        lr =.001
        learn.fit(lr, 1, cycle_len=1, cycle_mult=1) # train last few layers
        lrs = np.array([lr/4,lr/2,lr])
        learn.unfreeze()
        learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, best_save_name='resnext101_1_full') # train whole model


    def train_Fusion1(self, GPU_NUM):
        torch.cuda.set_device(GPU_NUM)
        print(torch.cuda.current_device())
        PATH =  '/media/rene/Data/data/idc'
        save_path = self.save_path
        num_workers = 6
        model_list = [resnet34, resnet34, resnet34]
        model_name_list = ['resnet34_0', 'resnet34_5', 'resnet34_2']
        batch_size = 4
        input_size = 6
        epochs = 10


        dataloaders, dataset_sizes = make_batch_gen(PATH, batch_size, num_workers, 
                                                    valid_name='valid', test_name='test', size=197)

        pretrained_model_list = []
        for idx, model_arch in enumerate(model_list):
            model_ft = model_arch(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 2)
            model_ft = model_ft.cuda()
            model_ft.load_state_dict(torch.load(os.path.join(save_path, model_name_list[idx])))
            pretrained_model_list.append(model_ft)

        model_ft = Fusion1Hardcode(pretrained_model_list)
        model_ft = model_ft.cuda()
        model_name = 'Fusion1_1'

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        best_acc, model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         epochs, dataloaders, dataset_sizes)
        torch.save(model_ft.state_dict(), os.path.join(save_path, model_name))


    def train_Fusion2(self, GPU_NUM):
        torch.cuda.set_device(GPU_NUM)
        print(torch.cuda.current_device())
        PATH =  '/media/rene/Data/data/idc'
        save_path = self.save_path
        num_workers = 6
        model_list = [resnet34, resnet34, resnet34]
        model_name_list = ['resnet34_0', 'resnet34_5', 'resnet34_2']
        batch_size = 32
        input_size = 6
        epochs = 4#10

        dataloaders, dataset_sizes = make_batch_gen(PATH, batch_size, num_workers, 
                                                    valid_name='valid', test_name='test', size=197)

        pretrained_model_list = []
        for idx, model_arch in enumerate(model_list):
            model_ft = model_arch(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 2)
            model_ft = model_ft.cuda()
            model_ft.load_state_dict(torch.load(os.path.join(save_path, model_name_list[idx])))
            pretrained_model_list.append(model_ft)

        model_ft = Fusion2Hardcode(pretrained_model_list)
        model_ft = model_ft.cuda()
        model_name = 'Fusion2_1'

        for p in model_ft.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        best_acc, model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         epochs, dataloaders, dataset_sizes, verbose=True)
        torch.save(model_ft.state_dict(), os.path.join(save_path, model_name+'-test-full-data'))


    def train_Fusion2_2stage(self, GPU_NUM):
        torch.cuda.set_device(GPU_NUM)
        print(torch.cuda.current_device())
        PATH =  '/media/rene/Data/data/idc'
        save_path = self.save_path
        num_workers = 6
        model_list = [resnet34, resnet34, resnet34]
        model_name_list = ['resnet34_0', 'resnet34_5', 'resnet34_2']
        batch_size = 64
        input_size = 6

        dataloaders, dataset_sizes = make_batch_gen(PATH, batch_size, num_workers, 
                                                    valid_name='valid', test_name='test', size=197)

        pretrained_model_list = []
        for idx, model_arch in enumerate(model_list):
            model_ft = model_arch(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 2)
            model_ft = model_ft.cuda()
            model_ft.load_state_dict(torch.load(os.path.join(save_path, model_name_list[idx])))
            pretrained_model_list.append(model_ft)

        model_ft = Fusion2Hardcode(pretrained_model_list)
        model_ft = model_ft.cuda()
        model_name = 'Fusion2_2s1-test-full-data'

        #Train last few layers
        epochs=3

        for p in model_ft.parameters():
            p.requires_grad = True

        for p in model_ft.model1.parameters():
            p.requires_grad = False
        for p in model_ft.model2.parameters():
            p.requires_grad = False
        for p in model_ft.model3.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,model_ft.parameters()), lr=0.003)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        best_acc, model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         epochs, dataloaders, dataset_sizes, verbose=True)

        torch.save(model_ft.state_dict(), os.path.join(save_path, model_name))

        # train all layers
        batch_size = 16
        epochs = 10
        dataloaders, dataset_sizes = make_batch_gen(PATH, batch_size, num_workers, 
                                                    valid_name='valid', test_name='test', size=197)

        for p in model_ft.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,model_ft.parameters()), lr=0.0003)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        best_acc, model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         epochs, dataloaders, dataset_sizes, verbose=True)

        model_name = 'Fusion2_2s2-test-full-data'
        torch.save(model_ft.state_dict(), os.path.join(save_path, model_name))


    def train_old_fusion(self, GPU_NUM):
        torch.cuda.set_device(GPU_NUM)
        def make_fusion_model(model_list, model_name_list, save_path, input_size, model=WeightedSum):
            model_list_ft = []    
            for idx, model_arch in enumerate(model_list):
                # get the proper model architecture
                model_ft = model_arch(pretrained=False)
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, 2)
                model_ft = model_ft.cuda()
                # load the saved weights
                model_ft.load_state_dict(torch.load(os.path.join(save_path, model_name_list[idx])))
                model_list_ft.append(model_ft)
                
            fusion_model = model(num_input=6)
            fusion_model = fusion_model.cuda()
            return model_list_ft, fusion_model

        model_list = [resnet34, resnet34, resnet34]
        model_name_list = ['resnet34_0', 'resnet34_5', 'resnet34_2']    

        batch_size = 16
        num_workers = 6
        save_path = '/media/rene/Data/data/idc/models'
        PATH = '/media/rene/Data/data/idc/sample'
        input_size = 6

        dataloaders, dataset_sizes = make_batch_gen(PATH, batch_size, num_workers, valid_name='valid', size=197)
        model_list_ft, fusion_model = make_fusion_model(model_list, model_name_list, save_path, input_size, model=WeightedSum)


        num_epochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(fusion_model.parameters(), lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        model = train_fusion_model(fusion_model, model_list_ft, 
                            criterion, optimizer_ft, exp_lr_scheduler, num_epochs,
                             dataloaders, dataset_sizes)

        torch.save(model.state_dict(), os.path.join(save_path, 'old_fusion'))


def main():
    fire.Fire(Trainer)

if __name__ == '__main__':
    main()




