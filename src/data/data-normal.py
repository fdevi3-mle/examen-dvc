import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os


from src.utils import ExtensionMethods, X_TRAIN_FILE_DATA_PATH, X_TEST_FILE_DATA_PATH

def main(X_train_file=X_TRAIN_FILE_DATA_PATH,X_test_file=X_TEST_FILE_DATA_PATH):
    if not(os.path.exists(X_train_file) and os.path.exists(X_test_file)):



def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    ##make it if doesnt exist
    os.makedirs(output_folderpath,exist_ok=True) ##
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if ExtensionMethods.check_existing_folder(output_filepath):
            file.to_csv(output_filepath, index=False)