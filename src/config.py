## project parameters
use_data_subset = True
subset_fraction = 1/16 # What fraction of the dataset to use
load_paths = True # if False, will run main_read_data which reads in all the file paths and saves them to a dataframe

## directory paths
data_dir = '../data/'



# todo: change this for local/oscar paths -- set to whatever path you want to save text to
if use_data_subset:
  root_dir = f'{data_dir}raw/MASAD_SUBSET'
  save_path_raw = f'{data_dir}processed/MASAD_processed_raw_text_subset.pkl'
  df_save_path = f'{data_dir}processed/MASAD_processed_subset.csv'
  hidden_states_path = f'{data_dir}processed/hidden_states_subset.pt'
else:
  root_dir = f'{data_dir}raw/MASAD'
  save_path_raw = f'{data_dir}processed/MASAD_processed_raw_text.pkl'
  df_save_path = f'{data_dir}processed/MASAD_processed.csv'
  hidden_states_path = f'{data_dir}processed/hidden_states.pt'
