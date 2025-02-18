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

MODEL_PATH =  os.path.join(ROOT_PATH,'models')
NOTEBOOK_PATH = os.path.join(ROOT_PATH, 'notebooks')
SRC_PATH = os.path.join(ROOT_PATH,'src')
SRC_DATA_PATH = os.path.join(SRC_PATH,'data')
SRC_MODEL_PATH = os.path.join(SRC_PATH,'models')

BUCKET_URL = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'


##PYTHON file list cause I can't be bothered to rem names
PY_DATA_NORMAL = os.path.join(SRC_DATA_PATH,'data-normal.py')
PY_DATA_SPLITTER = os.path.join(SRC_DATA_PATH,'data-splitter.py')

##MODEL
PY_MODEL_GRID = os.path.join(SRC_MODEL_PATH,'grid-search.py')
PY_MODEL_TRAIN = os.path.join(SRC_MODEL_PATH,'model-training.py')
PY_MODEL_EVALUATE = os.path.join(SRC_MODEL_PATH,'model-evaluate.py')

class ExtensionMethods:
    @staticmethod
    def check_existing_folder(folder_path):
        '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
        if not os.path.exists(folder_path):
            while True:
                response = input(f"{os.path.basename(folder_path)} doesn't exists. Do you want to create it? (y/n): ")
                if response.lower() == 'y':
                    return True
                elif response.lower() == 'n':
                    return False
                else:
                    print("Invalid response. Please enter 'y' or 'n'.")
        else:
            return False

    @staticmethod
    def create_folder_if_necessary(output_folderpath):
        # Create folder if necessary
        if ExtensionMethods.check_existing_folder(output_folderpath):
            os.makedirs(output_folderpath)


if __name__ == "__main__":
    print(f"ROOT_PATH: {ROOT_PATH}")
    print(f"CURRENT_PATH: {CURRENT_PATH}")
    print(f"DATA_PATH: {RAW_DATA_PATH}")
    print(f"DATA_PATH: {PY_MODEL_TRAIN} and it exists {os.path.exists(PY_MODEL_TRAIN)}")


