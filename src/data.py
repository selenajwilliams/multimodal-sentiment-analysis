
import os
import pandas as pd
import glob
import config
from tqdm import tqdm
import sys

# from PIL import Image
# from torchvision import transforms
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import CLIPProcessor
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# import re
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
USE_DATA_SUBSET = config.use_data_subset
LOAD_PATHS = config.load_paths
SUBSET_FRACTION = config.subset_fraction
HIDDEN_STATES_PATH = config.hidden_states_path

data_dir = config.data_dir
root_dir = config.root_dir
save_path_raw = config.save_path_raw
df_save_path = config.df_save_path

""" Section 1: Read the file paths into a dataframe
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
    if USE_DATA_SUBSET:
        df_save_path = os.path.join(target_dir, 'MASAD_processed_subset.csv')
    else:
        df_save_path = os.path.join(target_dir, 'MASAD_processed.csv')
        df.to_csv(df_save_path, index=False)
        print(f"\nDataFrame saved at {df_save_path}")

def main_read_data():
    """ Ensures all paths are valid,
        Returns df w/ the following columns: label, text_path, image_path
    """
    print(f'validating the root directory path')
    validate_path(root_dir)
    # print_directory_structure(root_dir)

    # create the dataframe by reading in CSV of text data
    print(f'saving the CSV of text and image filepaths to a dataframe')
    df = pd.read_csv(df_save_path)

    # save df to pickle file; at this point, df columns are: 'raw_text' and 'text_path'
    print(f'saving the dataframe w/ filepaths & labels for each sample to a pickle file')
    df.to_pickle(save_path_raw)
    

    # Display the first few entries
    print(f'first few entries of dataframe are:')
    print(df[['text_path', 'image_path']].head())

    return df


""" Section 2: Pre Process the Data! This includes:
    - clenaing the text
    - getting CLIP sequence level embeddings
    - 
"""

from transformers import CLIPImageProcessor, CLIPProcessor, CLIPImageProcessor, CLIPModel
import numpy as np
from transformers import CLIPImageProcessor, CLIPProcessor, CLIPTokenizerFast
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
set_max_cell_output_height()

from PIL import Image, UnidentifiedImageError

def init_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"model will run on {device}")

def serialize_df(df, path):
    """ Saves to file """
    df.to_pickle(path)

def save_to_pt(tensor, path):
    # saves the tensor to a .pt file
    torch.save(tensor, path)
    print(f'successfully saved tensor to file: {path}')

def extract_text(text_file):
    # TODO: What does this function do?
    """ opens a text file and returns each line?
        returns corrupted/erroneous lines as empty strings?
    """
    path = f'{data_dir}raw/{text_file}'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Text file at path: \n{path} \ncould not be found when reading in the text file for all text files in the dataset. Perhaps data directory is misconfigured in configuration file?")

    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        # Instead of printing, return empty string or handle as needed
        print(f'WARNING: when extracting text, the text file has ')
        return ""

def process_labels(labels):
  """ Goes through all labels and converts from a list of strings to a pytorch tensor, 0 if negative, 1 is positive
  """
  label_map = {"negative": 0, "positive": 1}  # Adjust as per your actual label names
  encoded_labels = [label_map[label] for label in labels]
  return torch.tensor(encoded_labels, dtype=torch.float)

def print_model_inputs_info(model_inputs):
  print(f'model inputs has type: {type(model_inputs)}')

  print(f'model inputs is: {model_inputs}')
  print('\n\n')
  print(f'\nmodel has len: {len(model_inputs)}')
  print(f'model inputs at idx 0: model_inputs[0] {type(model_inputs[0])}')
  print(f'model inputs at idx 1: model_inputs[1] {type(model_inputs[1])}')
  print(f'model_inputs[0]: \n{model_inputs[0]}')
  print(f'model_inputs[1]: \n{model_inputs[1]}')

def sample_subset_of_data(df, subset_frac): 
    """ 
    """
    df_small = df.sample(frac=subset_frac, random_state=42).reset_index(drop=True)

    # Display the sizes of the original and scaled-down DataFrames
    print(f"Original DataFrame size: {df.shape}")
    print(f"Scaled-down DataFrame size: {df_small.shape}")
    # (Optional) Verify the class distribution in the scaled-down DataFrame
    print("\nClass distribution in scaled-down DataFrame:")
    print(df_small['label'].value_counts())

    return df_small

def print_model_inputs_size(model_inputs):
    mytensor = model_inputs['input_ids']
    print(f'mytensor has shape: {mytensor.shape} w/ dtype: {mytensor.dtype}')
    print(f'total bytes: {64 * mytensor.shape[0] * mytensor.shape[1]}')
    print(f'total Mb: {64 * mytensor.shape[0] * mytensor.shape[1] // 1000}')
    print(f'total Gb: {64 * mytensor.shape[0] * mytensor.shape[1] // (1000 * 1000)}')

# define simple model to project the image and text sequence-level embeddings to the same dimensional space
class Projection(nn.Module):
    # taken from https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
    def __init__(self, d_in: int, d_out: int, dropout: float=0.5) -> None:
        super().__init__()
        print(f'shapes into layer 1: {d_in}, {d_out}')
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

def extract_hidden_states(model, model_inputs):
    # model_inputs = model_inputs # removing this b/c it seems redundant
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
      outputs = model(**model_inputs, output_hidden_states=True)
      print(f'model outputs have size: {sys.getsizeof(outputs) / (1000 * 1000)} Gb or {sys.getsizeof(outputs) / 1000} Mb')

    # extract hidden states
    text_hidden_states = outputs.text_model_output.hidden_states[-1]
    tsize = sys.getsizeof(text_hidden_states)
    print(f'text hidden states have size: {tsize / 1000} Mb or {tsize / (1000 * 1000)} Gb')
    image_hidden_states = outputs.vision_model_output.hidden_states[-1]
    isize = sys.getsizeof(image_hidden_states)
    print(f'text hidden states have size: {isize / 1000} Mb or {isize / (1000 * 1000)} Gb')
    # print(f'the text hidden states have shape: {text_hidden_states.shape} and img hidden states: {image_hidden_states.shape}')

    # train a small projection model to project the image embeddings into [batch_sz, x, 512]
    projection_embedding_model = Projection(image_hidden_states.shape[-1], 512, dropout=0.5)
    image_hidden_states = projection_embedding_model(image_hidden_states)

    concatanated_hidden_states = torch.cat((text_hidden_states,image_hidden_states), dim=1)
    print(f'the text hidden states have shape: {text_hidden_states.shape} and img hidden states: {image_hidden_states.shape} concat shape: {concatanated_hidden_states.shape} concatenated hidden states: {concatanated_hidden_states.shape}')

    return concatanated_hidden_states


def load_and_prune_data(df):
  # returns text_list, im_list, removes any corrupted files
  im_list = []
  im_paths = df['image_path'].tolist()
  indices_to_drop = []  # To keep track of corrupted image indices

  # Loop through image paths
  for i, img_path in enumerate(tqdm(im_paths, desc="Loading Images")):
      try:
          # Open and convert the image to RGB
          img = Image.open(img_path).convert("RGB")
          im_list.append(img)
      except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
          print(f"Warning: Skipping corrupted or unreadable image: {img_path} | Error: {e}")
          indices_to_drop.append(i)  # Mark this index for removal

  # Drop the rows with corrupted images
  if indices_to_drop:
      df = df.drop(indices_to_drop).reset_index(drop=True)
      print(f"Dropped {len(indices_to_drop)} corrupted images. New DataFrame size: {df.shape}")

  # Proceed with the remaining valid images
  text_list = df['clean_text'].tolist()

  # Ensure image and text lists have the same length
  assert len(im_list) == len(text_list), "Mismatch between images and texts after processing."
  return text_list, im_list, df


# TODO: factor out processor initialization and pass it in so that the processers are not initialized every time
def preprocess(texts, images):
    ''' Takes in a list of text and images, returns
    '''
    transforms = v2.Compose([
        v2.Resize(size=[224, 224]),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.2),
        v2.RandomRotation(90),
        v2.RandomZoomOut(p=0.3)])

    transformed_images = []
    for img in images:
        img = transforms(img)
        transformed_images.append(img)

    ## start of text token is T_CLS, start of image token is just CLS
    imageProcessor = CLIPImageProcessor(do_rescale=True, rescale_factor=1/255, do_normalize=True)
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16', bos_token='[T_CLS]')

    processor = CLIPProcessor(imageProcessor, tokenizer)
    inputs = processor(text=texts, images=transformed_images, truncation=True, return_tensors="pt", padding=True)
    return inputs


def main_preprocess_data(df):

    # Clean the text & save updated df to pickle file
    print(f'cleaning the text in the dataframe')
    tqdm.pandas()
    df['raw_text'] = df['text_path'].progress_apply(extract_text)
    # write the updatd df to pickle file
    serialize_df(df, save_path_raw)
    print(f"Saved df w/ raw text column to {save_path_raw}")

    df = pd.read_pickle(f'{data_dir}processed/MASAD_processed_clean_text.pkl') # TODO: add this file to local directory, modify paths as needed in the file?

    # load a fraction of the dataset, pruning any corrupted samples
    df_small = sample_subset_of_data(df, SUBSET_FRACTION)
    text, images, df_small = load_and_prune_data(df_small)

    # process the data via clip tokenizer w/ data augmentation for images
    print(f'preprocessing the text w/ CLIP processor')
    clip_processor_outputs = preprocess(text, images) 
    print_model_inputs_size(clip_processor_outputs)

    # initilize clip model
    print(f'getting model inputs from CLIP now')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    clip_model_sz = sys.getsizeof(clip_model) # TODO: fix this -- why does it say clip model is 0 Mb?
    print(f'the clip model takes up {clip_model_sz // 1000} Mb or {clip_model_sz // (1000 * 1000)} Gb')

    # extract concatenated sequence-level embeddings using clip model & processor outputs
    mme = extract_hidden_states(clip_model, clip_processor_outputs) # TODO: refactor mme to sequence_embeds once data loader is migrated to python file
    save_to_pt(mme, HIDDEN_STATES_PATH)

    label_encodings = process_labels(df_small['labels']) # TODO: I think this works?

    return 








    clip_processor_outputs = "empty string"
    return clip_processor_outputs, df

    

if __name__ == "__main__":
    # TODO: add a check that reads the df from .pkl file if config boolean indicates to, otherwise runs main_read_data()
    if LOAD_PATHS:
        df = pd.read_pickle(save_path_raw)
    else: 
        df = main_read_data()

    print(f'calling main preprocess data')
    model_inputs, df = main_preprocess_data(df)

    
""" RAW COLAB COPIES """

