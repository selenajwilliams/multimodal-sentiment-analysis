
import os
import pandas as pd
import glob
import config

# from tqdm import tqdm
# from PIL import Image
# from torchvision import transforms
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import CLIPProcessor
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# import re
# import sys
# import sklearn
# import pickle 


""" This file has 3 main functions:
    1. load the data by reading in it's filepaths
    1.5 cleaning the text 
    2. pre-process data by extracting the clip hidden state sequence level embeddings
    3. save all the sequence embeddings to numpy files so they can be easily accessed in  
       the future
    4. create the data loader w/ train test split via a custom dataset class that inherits    
       from pytorch Dataset
"""


""" Section 0: Configuration
"""
use_data_subset = config.use_data_subset
root_dir = config.root_dir
save_path_raw = config.save_path_raw
df_save_path = config.df_save_path

""" Copied / new functions
"""
def validate_path(root_dir):
    if not os.path.exists(root_dir):
        raise Exception(f"The root directory does not exist. \nRoot dir: {root_dir} \nPlease verify the path in config.py and try again.")
    else:
        top_level_dirs = os.listdir(root_dir)
        print("Top-level directories in MASAD:")
        print(top_level_dirs)

def print_directory_structure(root_dir, max_depth=4, max_dirs=2, max_files=2):
    """
    Prints the directory structure up to a specified depth.

    Parameters:
    - root_dir (str): The root directory to start traversal.
    - max_depth (int): Maximum depth to traverse.
    - max_dirs (int): Maximum number of subdirectories to display at each level.
    - max_files (int): Maximum number of files to display in each directory.
    """
    for current_path, dirs, files in os.walk(root_dir):
        # Calculate the current depth
        relative_path = os.path.relpath(current_path, root_dir)
        depth = relative_path.count(os.sep)
        if depth > max_depth:
            continue  # Skip directories beyond the max_depth

        # Create indentation based on depth
        indent = '    ' * depth
        dir_name = os.path.basename(current_path)
        print(f"{indent}{dir_name}/")

        # Display a limited number of subdirectories
        for sub_dir in sorted(dirs)[:max_dirs]:
            print(f"{indent}    {sub_dir}/")

        # Display a limited number of files with their extensions
        for file in sorted(files)[:max_files]:
            print(f"{indent}    {file}")

        print()  # Add an empty line for better readability

def init_data_frame(root_dir, df_save_path):
    """ iterates through all filepaths in the root directory, saving the file paths of every text &
        image data sample to a dataframe
        returns:
            dataframe with columns:
                'image_path'
                'text_path'
                'label'
    """
    # Initialize lists to store paths and labels
    image_paths = []
    text_paths = []
    labels = []

    # Define splits and sentiments
    splits = ['train', 'test']
    sentiments = ['negative', 'positive']

    # Function to check if a file is a valid image or text file
    def is_valid_file(file_path, extension):
        return os.path.isfile(file_path) and not os.path.basename(file_path).startswith('.') and file_path.lower().endswith(extension)

    # Traverse the directories
    for split in splits:
        for sentiment in sentiments:
            # Define paths for image and text modalities
            image_modality_path = os.path.join(root_dir, split, 'image', sentiment)
            text_modality_path = os.path.join(root_dir, split, 'text', sentiment)

            # Check if both image and text modality paths exist
            if not os.path.isdir(image_modality_path):
                print(f"Missing image modality directory: {image_modality_path}")
                continue
            if not os.path.isdir(text_modality_path):
                print(f"Missing text modality directory: {text_modality_path}")
                continue

            # Use glob to find all .jpg files recursively in image_modality_path
            # This handles both nested and non-nested directory structures
            image_pattern = os.path.join(image_modality_path, '**', '*.jpg')
            found_images = glob.glob(image_pattern, recursive=True)

            if not found_images:
                print(f"No images found in {image_modality_path}")
                continue

            print(f"Processing {len(found_images)} images in {split}/{sentiment}...")

            for image_file in found_images:
                # Derive the corresponding text file path
                # Replace 'image' with 'text' in the path and change extension to .txt
                relative_image_path = os.path.relpath(image_file, os.path.join(root_dir, split, 'image', sentiment))
                text_file = os.path.join(root_dir, split, 'text', sentiment, relative_image_path)
                text_file = os.path.splitext(text_file)[0] + '.txt'

                # Check if the text file exists
                if is_valid_file(text_file, '.txt'):
                    image_paths.append(image_file)
                    text_paths.append(text_file)
                    labels.append(sentiment)
                else:
                    print(f"Missing text file for image: {image_file}")

    # Create a DataFrame from the collected data
    data = {
        'image_path': image_paths,
        'text_path': text_paths,
        'label': labels
    }

    df = pd.DataFrame(data)
    print(f"\nTotal samples collected: {len(df)}")
    print(df.head())

    # Handle Empty DataFrame
    if df.empty:
        raise ValueError("No data samples were collected. Please check the directory structure and paths.")

    # Save the DataFrame
    target_dir = '/content/drive/MyDrive/computer_vision_final_project/'
    os.makedirs(target_dir, exist_ok=True)
    if use_data_subset:
        df_save_path = os.path.join(target_dir, 'MASAD_processed_subset.csv')
    else:
        df_save_path = os.path.join(target_dir, 'MASAD_processed.csv')
        df.to_csv(df_save_path, index=False)
        print(f"\nDataFrame saved at {df_save_path}")

def extract_text(text_file):
    # TODO: What does this function do?
    """ opens a text file and returns each line?
        returns corrupted/erroneous lines as empty strings?
    """
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        # Instead of printing, return empty string or handle as needed
        return ""

# Save the DataFrame

def main_load_data():
    print(f'validating the root directory path')
    validate_path(root_dir)
    print_directory_structure(root_dir)

    # create the dataframe by reading in CSV of text data
    print(f'savingthe CSV of text and image filepaths to a dataframe')
    df = pd.read_csv(df_save_path)

    # 3. Apply the extraction function with a progress bar
    print(f'cleaning the text in the dataframe')
    df['raw_text'] = df['text_path'].progress_apply(extract_text)

    # save df to pickle file; at this point, df columns are: 'raw_text' and 'text_path'
    print(f'saving the dataframe to a pickle file')
    df.to_pickle(save_path_raw)
    print(f"DataFrame with raw text saved at {save_path_raw}")

    # Display the first few entries
    print(f'first few entries of dataframe are:')
    print(df[['text_path', 'raw_text']].head())


if __name__ == "__main__":
    main_load_data()