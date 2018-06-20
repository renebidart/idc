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

# from fastai.imports import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *

from models import PreActResNet50


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




        ###################
    def train_preact50_cifar(self, epochs=2):
        save_path = '/home/rene/data/cifar-10-batches-py/models'
        best_acc = 0  # best test accuracy

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='/home/rene/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/home/rene/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        net = PreActResNet50()
        device = 'cuda'

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1)

        def train(epoch):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                scheduler.step()
                inputs, targets = inputs.to(device), targets.to(device)
        #         inputs = Variable(inputs.cuda())
        #         targets = Variable(targets.cuda())
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print(100.*correct/total, correct, total)

        def test(epoch, best_acc):
            # global best_acc
            print(best_acc)
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            print('test acc:', 100.*correct/total, correct, total)

            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, save_path +'/preact_resnet50')
                best_acc = acc
            return best_acc

        best_acc = 0
        for epoch in tqdm(range(epochs)):
            train(epoch)
            best_acc = test(epoch, best_acc)
            print(epoch, best_acc)


    def train_preact50_cifar_myway(self, epochs=35, n_models=8, batch_size=150):
        PATH = Path('/home/rene/data/cifar')
        # PATH = PATH / 'sample'
        save_path = PATH / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        epochs = int(epochs)
        num_workers = 6
            
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                    valid_name='valid')

        for i in range(n_models):
            model_name = 'PreActResNet50_'+str(i)
            model = PreActResNet50()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.2)

            best_acc, model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                                             epochs, dataloaders, dataset_sizes)
            torch.save(model.state_dict(), str(save_path / model_name))

    def train_resnet50(self, epochs=50, n_models=8):
        PATH = Path('/home/rene/data/cifar')
        save_path = PATH / 'models'
        save_path.mkdir(parents=True, exist_ok=True)

        num_workers = 6
        batch_size=80
        epochs = 40

        models = {}
        performance = {}
        for i in range(8):
            models['resnet50_'+str(i)] = resnet50
            
        dataloaders, dataset_sizes = make_batch_gen(str(PATH), batch_size, num_workers, 
                                                    valid_name='valid', size=197)

        for model_name, model_arch in models.items():
            model = model_arch(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.2)

            best_acc, model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                                             epochs, dataloaders, dataset_sizes)
            torch.save(model.state_dict(), str(save_path / model_name))
    


def main():
    fire.Fire(Trainer)

if __name__ == '__main__':
    main()




