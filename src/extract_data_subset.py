""" Extracts a subset of the data, used to develop a fast, iterative pipeline 
    for ML development
"""


data_path = '../MASAD/'
# image path example
train_cat_1 = '../MASAD/train/image/negative/apple/10250032023.jpg'

import shutil
import os

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

def fix_data():

    t_options = ['train', 'test']
    modalities = ['image', 'text']
    sentiments = ['positive', 'negative']
    categories = ['bady']

    for t in t_options:
        for modality in modalities:
            for sentiment in sentiments:
                for category in categories:
                    folder_path = f'../MASAD/{t}/{modality}/{sentiment}/{category}'
                    if os.path.exists(folder_path):
                        new_path = f'../MASAD/{t}/{modality}/{sentiment}/baby'
                        os.rename(folder_path, new_path)
                        print(f'renamed {folder_path} to end in baby')





def extract_data_subset():

    save_path = 'data'
    t_options = ['train', 'test']
    modalities = ['image', 'text']
    sentiments = ['positive', 'negative']
    categories = ['baby', 'autumn', 'apple']

    # go through existing dataset and save subset of data to new folders
    for t in t_options:
        for sentiment in sentiments:
            for category in categories:
                print(f'processing {category} {t} data')
                folder_path = f'../MASAD/{t}/image/{sentiment}/{category}'
                print(f'extracting the first 5 images at {folder_path}...')
                files = os.listdir(folder_path)
                for i in range(5):
                    id = files[i].rsplit('/')[0].split('.')[0]
                    print(f'id: {id}')
                    # copy image to new location
                    img_src_path = f'{folder_path}/{id}.jpg'
                    img_dest_path = f'{save_path}/{t}/image/{sentiment}/{category}/{id}.jpg'
                    shutil.copy(img_src_path, img_dest_path)

                    # copy corresponding caption to new location
                    txt_src_path = f'../MASAD/{t}/text/{sentiment}/{category}/{id}.txt'
                    txt_dest_path = f'{save_path}/{t}/text/{sentiment}/{category}/{id}.txt'
                    shutil.copy(txt_src_path, txt_dest_path)



print(f'running init directories')
init_directories(save_path, t_options, modalities, sentiments, categories)
print(f'running fix data')
fix_data()
extract_data_subset()


