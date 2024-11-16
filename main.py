# main file
# preprocessing data script

data_path = '../MASAD/'
# image path example
train_cat_1 = '../MASAD/train/image/negative/apple/10250032023.jpg'

import shutil
import os
import sys

save_path = 'data'
t_options = ['train', 'test']
modalities = ['image', 'text']
sentiments = ['positive', 'negative']
categories = ['apple', 'autumn', 'bady'] 


def init_directories(save_path, t_options, modalities, sentiments, categories):
    # initialize folders to save subset of data to 
    # will need to initialize directories to avoid os errors
    new_directories = []
    for t in t_options:
        for modality in modalities:
            for sentiment in sentiments:
                for category in categories:
                    if category == 'bady': # fix typo
                        category = 'baby'
                    path = f'{save_path}/{t}/{modality}/{sentiment}/{category}'
                    os.makedirs(path)
                    new_directories.append(path)
    # print(f'created the following new directories: ')
    # [print(dir) for dir in new_directories]

def extract_data_subset():

    save_path = 'data'
    t_options = ['train', 'test']
    modalities = ['image', 'text']
    sentiments = ['positive', 'negative']
    categories = ['apple', 'autumn', 'bady']

    # go through existing dataset and save subset of data to new folders
    for t in t_options:
        for modality in modalities:
            for sentiment in sentiments:
                for category in categories:
                    if category == 'autumn':
                        break
                    folder_path = f'../MASAD/{t}/{modality}/{sentiment}/{category}'
                    print(f'extracting the first 5 images at {folder_path}...')
                    files = os.listdir(folder_path)
                    # print(f'  first 5 files: ')
                    # [print(f'    {file}') for file in files[:5]]
                    for i in range(5):
                        file_name = files[i].rsplit('/')[0]
                        src_path = f'{folder_path}/{file_name}'
                        # print(f'file name: {file_name}')
                        print(f'src path: {src_path}')
                        
                        dest_path = f'{save_path}/{t}/{modality}/{sentiment}/{category}/{file_name}'
                        # print(f'dest path: {dest_path}')

                        shutil.copy(src_path, dest_path)



extract_data_subset()


