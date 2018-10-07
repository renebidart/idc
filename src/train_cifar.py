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
from collections import OrderedDict

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
import torch.backends.cudnn as cudnn

from models import ResNet50
from densenet import DenseNet


# Add the src directory for functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
sys.path.append(src_dir)

# import my functions:
from functions import*
from models import Fusion2, Fusion6, Fusion2More, Fusion3

sys.path.append('/media/rene/code/wide-resnet.pytorch')
from networks import Wide_ResNet



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



    def train_fusionWRN4(self, epochs1=120, epochs2=80, device="cuda:1"):
        epochs1, epochs2 = int(epochs1), int(epochs2)
        num_workers = 4

        PATH = Path('/media/rene/data/')
        save_path = PATH / 'cifar-10-batches-py/wide-RN-models'
        model_name_list = ['wideRN2.t7', 'wideRN5.t7', 'wideRN0.t7', 'wideRN1.t7']
        batch_size = 12

        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        # get all the models
        pretrained_model_list = []
        for i, model_name in enumerate(model_name_list):
            model = Wide_ResNet(28, 20, 0, 10)
            model = model.to(device)
            
            # load the saved weights
            model_path = os.path.join(save_path, model_name)
            model.load_state_dict(torch.load(model_path)['net'].state_dict())
            pretrained_model_list.append(model)

        model = Fusion2(pretrained_model_list, num_input=40, num_output=10)

        ######################  TRAIN LAST FEW LAYERS
        print('training last few layers')

        model_name = 'Fusion2_WRN_1'
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

        model_name = 'Fusion2_WRN_2'
        batch_size = 1
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.0005, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                   dataloaders, dataset_sizes, device=device)

        torch.save(model.state_dict(), str(save_path / model_name))



    def train_fusionWRN6(self, epochs1=120, epochs2=3, device="cuda:0"): # https://github.com/xternalz/WideResNet-pytorch.git #120 80
        sys.path.append('/media/rene/code/WideResNet-pytorch')
        from wideresnet import WideResNet

        epochs1, epochs2 = int(epochs1), int(epochs2)
        num_workers = 4

        PATH = Path('/media/rene/data/')
        save_path = Path('/media/rene/code/WideResNet-pytorch/runs')
        model_name_list = ['WideResNet-28-10_0/model_best.pth.tar', 'WideResNet-28-10_1/model_best.pth.tar', 'WideResNet-28-10_2/model_best.pth.tar', 
                           'WideResNet-28-10_3/model_best.pth.tar', 'WideResNet-28-10_4/model_best.pth.tar', 'WideResNet-28-10_5/model_best.pth.tar']
        batch_size = 8

        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        pretrained_model_list = []
        # First trained model was with DATA PARALLEL
        model = WideResNet(28, 10, 20)
        model = model.to(device)
        state_dict = torch.load(os.path.join(save_path, 'WideResNet-28-10_0/model_best.pth.tar'))['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        pretrained_model_list.append(model)

        # get all the models
        for i, model_name in enumerate(model_name_list[1:]):
            print('------------loading model: ', model_name)
            model = WideResNet(28, 10, 20)
            model = model.to(device)

            # original saved file with DataParallel
            state_dict = torch.load(os.path.join(save_path, model_name))['state_dict']
            model.load_state_dict(state_dict)
            pretrained_model_list.append(model)

        model = Fusion6(pretrained_model_list, num_input=60, num_output=10)

        ######################  TRAIN LAST FEW LAYERS
        # print('training last few layers')

        model_name = 'Fusion6_WRN_1'
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
        for p in model.model5.parameters():
            p.requires_grad = False
        for p in model.model6.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
                                   dataloaders, dataset_sizes, device=device)
        torch.save(model.state_dict(), str(save_path / model_name))

        ########################   TRAIN ALL LAYERS

        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        # model.load_state_dict(torch.load(save_path / 'Fusion2_WRN_1'))

        model_name = 'Fusion6_WRN_2'
        batch_size = 1
        print('---------', batch_size)
        dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                            valid_name='valid')

        for p in model.parameters():
            p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.0001, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

        best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                   dataloaders, dataset_sizes, device=device)

        torch.save(model.state_dict(), str(save_path / model_name))


    def train_fusionWRN_last4(self, epochs1=80, epochs2=40, device="cuda:1"): # https://github.com/xternalz/WideResNet-pytorch.git #120 80
        with torch.cuda.device(1):
            sys.path.append('/media/rene/code/WideResNet-pytorch')
            from wideresnet import WideResNet

            epochs1, epochs2 = int(epochs1), int(epochs2)
            num_workers = 4

            PATH = Path('/media/rene/data/')
            save_path = Path('/media/rene/code/WideResNet-pytorch/runs')
            model_name_list = ['WideResNet-28-10_0/model_best.pth.tar', 'WideResNet-28-10_1/model_best.pth.tar', 'WideResNet-28-10_2/model_best.pth.tar', 
                               'WideResNet-28-10_3/model_best.pth.tar', 'WideResNet-28-10_4/model_best.pth.tar', 'WideResNet-28-10_5/model_best.pth.tar']
            batch_size = 300

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            ])
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid', transformation=transform_test)

            pretrained_model_list = []
            # First trained model was with DATA PARALLEL
            model = WideResNet(28, 10, 20)
            model = model.to(device)

            state_dict = torch.load(os.path.join(save_path, 'WideResNet-28-10_0/model_best.pth.tar'))['state_dict']

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            pretrained_model_list.append(model)

            # get all the models
            for i, model_name in enumerate(model_name_list[1:4]):
                print('------------loading model: ', model_name)
                model = WideResNet(28, 10, 20)
                model = model.to(device)

                # original saved file with DataParallel
                state_dict = torch.load(os.path.join(save_path, model_name))['state_dict']
                model.load_state_dict(state_dict)
                pretrained_model_list.append(model)

            model = Fusion2(pretrained_model_list, num_input=40, num_output=10)

            ######################  TRAIN LAST FEW LAYERS
            print('training last few layers')

            model_name = 'fusionWRN_last4_1'
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

            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/4), gamma=0.2)

            # best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
            #                            dataloaders, dataset_sizes, device=device)
            # torch.save(model.state_dict(), str(save_path / model_name))

            ########################   TRAIN ALL LAYERS
            model.load_state_dict(torch.load(save_path / 'fusionWRN_last4_1'))
            model = model.to(device)
            model_name = 'fusionWRN_last4_2'

            batch_size = 2
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid')
 
            ### ONLY THE LAST BLOCK:
            for i, child in enumerate(model.model1.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True
            for i, child in enumerate(model.model2.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True
            for i, child in enumerate(model.model3.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True
            for i, child in enumerate(model.model4.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.00005, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.2)

            best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                       dataloaders, dataset_sizes, device=device, multi_gpu=False)

            torch.save(model.state_dict(), str(save_path / model_name))



    def train_fusionWRN_last4_more(self, epochs1=80, epochs2=80, device="cuda:1"): # https://github.com/xternalz/WideResNet-pytorch.git #120 80
        with torch.cuda.device(1):
            sys.path.append('/media/rene/code/WideResNet-pytorch')
            from wideresnet import WideResNet

            epochs1, epochs2 = int(epochs1), int(epochs2)
            num_workers = 4

            PATH = Path('/media/rene/data/')
            save_path = Path('/media/rene/code/WideResNet-pytorch/runs')
            model_name_list = ['WideResNet-28-10_0/model_best.pth.tar', 'WideResNet-28-10_1/model_best.pth.tar', 'WideResNet-28-10_2/model_best.pth.tar', 
                               'WideResNet-28-10_3/model_best.pth.tar', 'WideResNet-28-10_4/model_best.pth.tar', 'WideResNet-28-10_5/model_best.pth.tar']
            batch_size = 128

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            ])
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid', transformation=transform_test)

            pretrained_model_list = []
            # First trained model was with DATA PARALLEL
            model = WideResNet(28, 10, 20)
            model = model.to(device)

            state_dict = torch.load(os.path.join(save_path, 'WideResNet-28-10_0/model_best.pth.tar'))['state_dict']

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            pretrained_model_list.append(model)

            # get all the models
            for i, model_name in enumerate(model_name_list[1:4]):
                print('------------loading model: ', model_name)
                model = WideResNet(28, 10, 20)
                model = model.to(device)

                # original saved file with DataParallel
                state_dict = torch.load(os.path.join(save_path, model_name))['state_dict']
                model.load_state_dict(state_dict)
                pretrained_model_list.append(model)

            model = Fusion2More(pretrained_model_list)

            ######################  TRAIN LAST FEW LAYERS
            print('training last few layers')

            model_name = 'fusionWRN_more_last4_1'
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

            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/4), gamma=0.2)

            # best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
            #                            dataloaders, dataset_sizes, device=device)
            # torch.save(model.state_dict(), str(save_path / model_name))


            ########################   TRAIN ALL LAYERS
            model.load_state_dict(torch.load(save_path / 'fusionWRN_more_last4_1'))
            model_name = 'fusionWRN_more_last4_2'

            batch_size = 16
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid')

            ### ONLY THE LAST BLOCK:
            for i, child in enumerate(model.model1.children()):
                for i, grandchild in enumerate(child.children()):
                    if(i >=3):
                        for p in grandchild.parameters():
                            p.requires_grad = True
            for i, child in enumerate(model.model2.children()):
                for i, grandchild in enumerate(child.children()):
                    if(i >=3):
                        for p in grandchild.parameters():
                            p.requires_grad = True
            for i, child in enumerate(model.model3.children()):
                for i, grandchild in enumerate(child.children()):
                    if(i >=3):
                        for p in grandchild.parameters():
                            p.requires_grad = True
            for i, child in enumerate(model.model4.children()):
                for i, grandchild in enumerate(child.children()):
                    if(i >=3):
                        for p in grandchild.parameters():
                            p.requires_grad = True


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.0001, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/3), gamma=0.1)

            best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                       dataloaders, dataset_sizes, device=device, multi_gpu=True)

            torch.save(model.state_dict(), str(save_path / model_name))



    def train_fusionWRN_last3(self, epochs1=40, epochs2=25, device="cuda:1"): # https://github.com/xternalz/WideResNet-pytorch.git #120 80
        with torch.cuda.device(1):
            sys.path.append('/media/rene/code/WideResNet-pytorch')
            from wideresnet import WideResNet

            epochs1, epochs2 = int(epochs1), int(epochs2)
            num_workers = 4

            PATH = Path('/media/rene/data/')
            save_path = Path('/media/rene/code/WideResNet-pytorch/runs')
            model_name_list = ['WideResNet-28-10_0/model_best.pth.tar', 'WideResNet-28-10_1/model_best.pth.tar', 'WideResNet-28-10_2/model_best.pth.tar', 
                               'WideResNet-28-10_3/model_best.pth.tar', 'WideResNet-28-10_4/model_best.pth.tar', 'WideResNet-28-10_5/model_best.pth.tar']
            batch_size = 300

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            ])
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid', transformation=transform_test)

            pretrained_model_list = []
            # First trained model was with DATA PARALLEL
            model = WideResNet(28, 10, 20)
            model = model.to(device)

            state_dict = torch.load(os.path.join(save_path, 'WideResNet-28-10_0/model_best.pth.tar'))['state_dict']

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            pretrained_model_list.append(model)

            # get all the models
            for i, model_name in enumerate(model_name_list[1:3]):
                print('------------loading model: ', model_name)
                model = WideResNet(28, 10, 20)
                model = model.to(device)

                # original saved file with DataParallel
                state_dict = torch.load(os.path.join(save_path, model_name))['state_dict']
                model.load_state_dict(state_dict)
                pretrained_model_list.append(model)

            model = Fusion3(pretrained_model_list, num_input=30, num_output=10)

            ######################  TRAIN LAST FEW LAYERS
            print('training last few layers')

            model_name = 'fusionWRN_last3_1'
            for p in model.parameters():
                p.requires_grad = True
            for p in model.model1.parameters():
                p.requires_grad = False
            for p in model.model2.parameters():
                p.requires_grad = False
            for p in model.model3.parameters():
                p.requires_grad = False

            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.005, momentum=0.9, weight_decay=5e-4)
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs1/3), gamma=0.3)

            # best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs1, 
            #                            dataloaders, dataset_sizes, device=device)
            # torch.save(model.state_dict(), str(save_path / model_name))

            ########################   TRAIN ALL LAYERS
            model.load_state_dict(torch.load(save_path / 'fusionWRN_last3_1'))
            model = model.to(device)
            model_name = 'fusionWRN_last3_2'

            batch_size = 88
            dataloaders, dataset_sizes = make_batch_gen_cifar(str(PATH), batch_size, num_workers,
                                                                valid_name='valid', transformation=transform_test)
 
            ### ONLY THE LAST BLOCK:
            for i, child in enumerate(model.model1.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True
            for i, child in enumerate(model.model2.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True
            for i, child in enumerate(model.model3.children()):
                if(i >=3):
                    for p in child.parameters():
                        p.requires_grad = True

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.0001, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/2), gamma=0.1)
            best_acc, model = train_model(model, criterion, optimizer, scheduler, 2, 
                                       dataloaders, dataset_sizes, device=device, multi_gpu=False)

            optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=.001, momentum=0.9, weight_decay=5e-4)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs2/2), gamma=0.2)

            best_acc, model = train_model(model, criterion, optimizer, scheduler, epochs2, 
                                       dataloaders, dataset_sizes, device=device, multi_gpu=False)

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