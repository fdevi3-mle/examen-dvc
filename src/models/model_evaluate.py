import json
import os.path
from idlelib.outwin import file_line_pats

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.data.data_normal import normalize_data_please
from src.data.data_splitter import split_data_please
from src.models.grid_search import grid_search_please
from src.models.model_training import train_model_please
from src.utils import X_TRAIN_SCALED_FILE_DATA_PATH, X_TEST_SCALED_FILE_DATA_PATH, Y_TRAIN_FILE_DATA_PATH, \
    Y_TEST_FILE_DATA_PATH, MODEL_PATH, BEST_PARAMS, RANDOM_FOREST, METRICS_PATH

from src.utils import X_TRAIN_SCALED_FILE_DATA_PATH, X_TEST_SCALED_FILE_DATA_PATH


def evaluate_model_please(X_train_scaled_file=X_TRAIN_SCALED_FILE_DATA_PATH,
                          X_test_scaled_file=X_TEST_SCALED_FILE_DATA_PATH,
                          y_train_file=Y_TRAIN_FILE_DATA_PATH,
                          y_test_file=Y_TEST_FILE_DATA_PATH, model_file=RANDOM_FOREST):
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

    if not os.path.exists(model_file):
        model_file = train_model_please()

    model  = joblib.load(model_file)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(METRICS_PATH,exist_ok=True)
    filepath = os.path.join(METRICS_PATH,'metrics.json')
    with open(filepath, "w") as f:
        json.dump({"MSE": mse, "R2": r2}, f)

    return filepath


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    evaluate_model_please()
