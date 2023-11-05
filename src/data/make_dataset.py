import pandas as pd

RAW_DATA_PATH = '../../data/raw/'
DATA_ZIP_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

import requests
import zipfile
import os
def download_dataset():
    if os.path.exists(f"{RAW_DATA_PATH}filtered.tsv"):
        return
    zip_file_path = f"{RAW_DATA_PATH}filtered.zip"
    response = requests.get(DATA_ZIP_URL)
    if response.status_code == 200:
        with open(zip_file_path , 'wb') as file:
            file.write(response.content)
        print(f"File downloaded to {zip_file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extract(file_list[0])
        extracted_tsv_path = f"{RAW_DATA_PATH}{os.path.splitext(file_list[0])[0] + '.tsv'}"
        os.rename(file_list[0], extracted_tsv_path)
        print(f"TSV file extracted to {extracted_tsv_path}")
    os.remove(zip_file_path)

def preprocess_dataset(N):
    '''
    Just swap reference and translation columns,
    if reference toxicity less than translation toxicity
    '''
    download_dataset()
    data = pd.read_csv(f"{RAW_DATA_PATH}filtered.tsv", sep='\t')[:N] # I can't preprocess whole data because it too long
    print("Data preprocessing...")
    for i in range(len(data)):
        if i % 1000==0:
            print(f"{100*i/N}% completed")
        if data.iloc[i, 5] < data.iloc[i, 6]:
            # Do swap
            z = data.iloc[i, 1]
            data.iloc[i, 1] = data.iloc[i, 2]
            data.iloc[i, 2] = z

            z = data.iloc[i, 5]
            data.iloc[i, 5] = data.iloc[i, 6]
            data.iloc[i, 6] = z
    return data


def make_dataset_toxic_text_to_neutral_text(N):
    data = preprocess_dataset(N)
    data = data[["reference", "translation"]]
    data.columns = ['toxic', 'non-toxic']
    return data