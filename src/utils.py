import os.path

##Constants & FILEPATHS
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) ## src
ROOT_PATH = os.path.dirname(CURRENT_PATH)
DATA_PATH =  os.path.join(ROOT_PATH, 'data')
RAW_DATA_PATH= os.path.join(DATA_PATH, 'raw_data')
RAW_FILE_DATA_PATH = os.path.join(RAW_DATA_PATH,'raw.csv')

PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed_data')
X_TRAIN_FILE_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'X_train.csv')
X_TRAIN_SCALED_FILE_DATA_PATH =os.path.join(PROCESSED_DATA_PATH,'X_train_scaled.csv')

X_TEST_FILE_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'X_test.csv')
X_TEST_SCALED_FILE_DATA_PATH =os.path.join(PROCESSED_DATA_PATH,'X_test_scaled.csv')

Y_TRAIN_FILE_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'y_train.csv')
Y_TEST_FILE_DATA_PATH = os.path.join(PROCESSED_DATA_PATH,'y_test.csv')

MODEL_PATH =  os.path.join(ROOT_PATH,'models')
BEST_PARAMS = os.path.join(MODEL_PATH,'best_grid_search_parameters.pkl')
RANDOM_FOREST = os.path.join(MODEL_PATH,'random_forest.pkl')

NOTEBOOK_PATH = os.path.join(ROOT_PATH, 'notebooks')
SRC_PATH = os.path.join(ROOT_PATH,'src')
SRC_DATA_PATH = os.path.join(SRC_PATH,'data')
SRC_MODEL_PATH = os.path.join(SRC_PATH,'models')

BUCKET_URL = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'

METRICS_PATH =  os.path.join(ROOT_PATH,'metrics')

##PYTHON file list cause I can't be bothered to rem names
PY_DATA_NORMAL = os.path.join(SRC_DATA_PATH,'data_normal.py')
PY_DATA_SPLITTER = os.path.join(SRC_DATA_PATH,'data_splitter.py')

##MODEL
PY_MODEL_GRID = os.path.join(SRC_MODEL_PATH,'grid_search.py')
PY_MODEL_TRAIN = os.path.join(SRC_MODEL_PATH,'model_training.py')
PY_MODEL_EVALUATE = os.path.join(SRC_MODEL_PATH,'model_evaluate.py')


if __name__ == "__main__":
    print(f"ROOT_PATH: {ROOT_PATH}")
    print(f"CURRENT_PATH: {CURRENT_PATH}")
    print(f"DATA_PATH: {RAW_DATA_PATH}")
    print(f"DATA_PATH: {PY_MODEL_TRAIN} and it exists {os.path.exists(PY_MODEL_TRAIN)}")


