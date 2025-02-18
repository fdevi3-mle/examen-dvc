import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.get_data import import_raw_data
from src.utils import PROCESSED_DATA_PATH, RAW_FILE_DATA_PATH


def split_data_please(input_filepath=RAW_FILE_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    ##import the data
    if not os.path.exists(input_filepath):
        input_filepath = import_raw_data()

    df = pd.read_csv(input_filepath,index_col='date')
    ## Split
    X_train, X_test, y_train, y_test = split_data(df)

    #makedir
    os.makedirs(output_path,exist_ok=True) ##makes it if it doesnt exist

    # Save dataframes to their respective output file paths
    a = save_dataframes(X_train, X_test, y_train, y_test, output_path)
    return a ## cant be bothered


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    ##make it if doesnt exist
    dic = {}
    os.makedirs(output_folderpath,exist_ok=True) ##
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        dic[filename]  = output_filepath
        file.to_csv(output_filepath, index=False)
    return dic


def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    split_data_please()