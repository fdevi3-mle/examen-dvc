import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.data.data_normal import normalize_data_please
from src.data.data_splitter import split_data_please
from src.models.grid_search import grid_search_please
from src.utils import X_TRAIN_SCALED_FILE_DATA_PATH, X_TEST_SCALED_FILE_DATA_PATH, Y_TRAIN_FILE_DATA_PATH, \
    Y_TEST_FILE_DATA_PATH, MODEL_PATH, BEST_PARAMS

from src.utils import X_TRAIN_SCALED_FILE_DATA_PATH, X_TEST_SCALED_FILE_DATA_PATH


def train_model_please(X_train_scaled_file=X_TRAIN_SCALED_FILE_DATA_PATH,
                       X_test_scaled_file=X_TEST_SCALED_FILE_DATA_PATH,
                       y_train_file=Y_TRAIN_FILE_DATA_PATH,
                       y_test_file=Y_TEST_FILE_DATA_PATH,best_param_file=BEST_PARAMS):
    if not(os.path.exists(y_test_file) and os.path.exists(y_train_file)):
        dic = split_data_please()
        y_train_file = dic['y_train']
        y_test_file = dic['y_test']


    y_train = import_dataset(y_train_file,sep=',')
    y_test = import_dataset(y_test_file,sep=',')
    y_train = np.ravel(y_train) ### this killed me :(
    y_test = np.ravel(y_test)

    if not(os.path.exists(X_train_scaled_file) and os.path.exists(X_test_scaled_file)):
        dic = normalize_data_please()
        X_train_scaled_file = dic['X_train_scaled']
        X_test_scaled_file = dic['X_test_scaled']


    X_train_scaled = import_dataset(X_train_scaled_file)
    X_test_scaled  = import_dataset(X_test_scaled_file)

    if not os.path.exists(best_param_file):
        best_param_file = grid_search_please()

    best_params = joblib.load(best_param_file)
    print(best_params)

    model = RandomForestRegressor(**best_params)
    model.fit(X_train_scaled, y_train)

    filepath = os.path.join(MODEL_PATH, 'random_forest.pkl')
    joblib.dump(model, filepath)

    return filepath


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    train_model_please()
