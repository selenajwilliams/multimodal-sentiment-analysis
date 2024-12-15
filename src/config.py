## project parameters
use_data_subset = True

## directory paths
data_dir = '../data/'
# text_file = 

# todo: change this for local/oscar paths -- set to whatever path you want to save text to
if use_data_subset:
  root_dir = f'{data_dir}raw/MASAD_SUBSET'
  save_path_raw = f'{data_dir}processed/MASAD_processed_raw_text_subset.pkl'
  df_save_path = f'{data_dir}processed/MASAD_processed_subset.csv'
else:
  root_dir = f'{data_dir}raw/MASAD'
  save_path_raw = f'{data_dir}processed/MASAD_processed_raw_text.pkl'
  df_save_path = f'{data_dir}processed/MASAD_processed.csv'