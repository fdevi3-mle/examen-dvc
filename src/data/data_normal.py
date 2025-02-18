import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

from sklearn.preprocessing import StandardScaler

from src.utils import  X_TRAIN_FILE_DATA_PATH, X_TEST_FILE_DATA_PATH, PROCESSED_DATA_PATH
from src.data.data_splitter import split_data_please


def normalize_data_please(X_train_file=X_TRAIN_FILE_DATA_PATH, X_test_file=X_TEST_FILE_DATA_PATH,output_path=PROCESSED_DATA_PATH):
    if not(os.path.exists(X_train_file) and os.path.exists(X_test_file)):
        dic = split_data_please()
        X_train_file = dic['X_train']
        X_test_file = dic['X_test']
        print(X_test_file)
        print(X_train_file)

    X_train = pd.read_csv(X_train_file,sep=',')
    X_test = pd.read_csv(X_test_file,sep=',')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    a =  save_dataframes(pd.DataFrame(X_train_scaled),pd.DataFrame(X_test_scaled),output_path)


def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    ##make it if doesnt exist
    dic = {}
    os.makedirs(output_folderpath,exist_ok=True) ##
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        dic[filename] = output_filepath
        file.to_csv(output_filepath, index=False)
    return dic


if __name__ == '__main__':
    normalize_data_please()
