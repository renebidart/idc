import os
import glob
import shutil
import random
import pickle
from pathlib import Path







######### DATASET  UTILS  ##########
"""keras format: -> ttv dir -> classes -> file"""


def ttv_split_list(all_files, save_loc, valid_frac, test_frac):
    random.seed(101)
    random.shuffle(all_files)
    num = len(all_files)
    
    files = {}
    files['train'] = all_files[0:int(num*(1-test_frac-valid_frac))]
    files['valid'] = all_files[int(num*(1-valid_frac-test_frac)):int(num*(1-test_frac))]
    files['test'] = all_files[int(num*(1-test_frac)):]
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    pickle.dump(files, open(os.path.join(save_loc, 'ttv_key.pkl'), "wb" ))
    return files

# def to_folder_format(PATH, file_list, get_file_class):
#     # go through files, move to directory containing only this class      
#     for file in all_files:
#         class_name = get_file_class(file)
#         new_path = Path(train_path / class_name).mkdir(parents=True, exist_ok=True)
#         file_destination = new_path / file.name
#         shutil.move(file, str(file_destination))

def ttv_split_list_move(PATH_OUT, get_file_class, file_ttv_split):
    """Take file_ttv_split and save in keras format in PATH_OUT dir"""
    for ttv_dir, file_list in file_ttv_split.items():
        if len(file_list)==0:
            print('No files in ', ttv_dir)
        else:
            for file in file_list:
                class_name = get_file_class(file)
                new_path = Path(PATH_OUT / ttv_dir / class_name)
                new_path.mkdir(parents=True, exist_ok=True)
                shutil.move(file, str(new_path / file.name))

def make_sample_dataset(DATA_PATH, SAMPLE_PATH, frac, ttv_folders=['train', 'test', 'valid']):
    """ttv_folders are the names of test, train, valid folders. Assume already in fastai folder format"""   
    for ttv_name in ttv_folders:
        classes = [folder for folder in list(Path(DATA_PATH / ttv_name).glob('*')) if folder.is_dir()]
        # go through classes, make output directory, copy a sample of image to this
        for class_loc in classes:    
            files = list(Path(class_loc).glob('*.png'))
            sample_files = random.sample(files, int(len(files)*frac))
            out_path =  Path(SAMPLE_PATH / ttv_name / class_loc.name)
            out_path.mkdir(parents=True, exist_ok=True)
            for file_to_copy in sample_files:
                shutil.copyfile(str(file_to_copy), str(out_path / file_to_copy.name))


def make_cfiar10(PATH, PATH_OUT):
    """Given location of unzipped CIFAR file from http://pjreddie.com/media/files/cifar.tgz, convert to keras format and make sample"""
    def get_file_class(file_name):
        file_class = str(file_name).rsplit('_')[-1].rsplit('.')[-2]
        return file_class

    # Split the training folder into valid and train
    train_file_list = list(Path(PATH / 'train').glob('*.png'))
    file_ttv_split = ttv_split_list(train_file_list, save_loc=PATH_OUT, valid_frac=.1, test_frac=0)
    file_ttv_split['test'] = list(Path(PATH / 'test').glob('*.png'))

    # move to keras format
    ttv_split_list_move(PATH_OUT, get_file_class, file_ttv_split)

    # make a sample dataset from the keras format
    SAMPLE_PATH = PATH_OUT / 'sample'
    make_sample_dataset(DATA_PATH=PATH_OUT, SAMPLE_PATH=SAMPLE_PATH, frac=.1, ttv_folders=['train', 'test', 'valid'])



