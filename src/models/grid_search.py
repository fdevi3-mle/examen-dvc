import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.data.data_normal import normalize_data_please
from src.data.data_splitter import split_data_please
from src.utils import X_TRAIN_SCALED_FILE_DATA_PATH, X_TEST_SCALED_FILE_DATA_PATH, Y_TRAIN_FILE_DATA_PATH, \
    Y_TEST_FILE_DATA_PATH, MODEL_PATH


def grid_search_please(X_train_scaled_file=X_TRAIN_SCALED_FILE_DATA_PATH,
                       X_test_scaled_file=X_TEST_SCALED_FILE_DATA_PATH,
                       y_train_file=Y_TRAIN_FILE_DATA_PATH,
                       y_test_file=Y_TEST_FILE_DATA_PATH):
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

    model = RandomForestRegressor()
    param_grid = {'n_estimators': [5, 100, 200], 'max_depth': [1,2,5,10,20]}

    #Since they ask for metrics for mse and r2 might as wekk use r2
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    #https://stackoverflow.com/questions/54197831/gridsearch-for-best-model-save-and-load-parameters
    filepath = os.path.join(MODEL_PATH, 'best_grid_search_parameters.pkl')
    joblib.dump(grid_search.best_params_, filepath)

    #https://stackoverflow.com/questions/69257882/how-to-load-a-joblib-file-with-custom-class-previously-saved-using-a-notebook
    # loaded_duck = joblib.load(filepath)
    # print(loaded_duck)

    return filepath


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    grid_search_please()
