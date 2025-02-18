import requests
import os


from src.utils import RAW_DATA_PATH, BUCKET_URL



def import_raw_data(raw_data_path=RAW_DATA_PATH,filename= 'raw.csv', bucket_url=BUCKET_URL):
    os.makedirs(raw_data_path,exist_ok=True)
    if filename is None:
        filename = 'raw.csv'
    filepath  = os.path.join(raw_data_path,filename)
    print(f'downloading {bucket_url} as {os.path.basename(filepath)}')
    response = requests.get(bucket_url)
    if response.status_code == 200:
        content = response.text
        text_file = open(filepath, "wb")
        text_file.write(content.encode('utf-8'))
        text_file.close()
    else:
        print(f'Error accessing the object {bucket_url}:', response.status_code)
    return filepath


if __name__ == '__main__':
    import_raw_data()

