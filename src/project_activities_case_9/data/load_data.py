# TODO загрузка файла с данными с kaggle
import pandas as pd
import kagglehub
import os

def load_from_kaggle(handle,output_dir):
    print(output_dir)
    path = kagglehub.dataset_download(handle,output_dir=output_dir,force_download=True)
    return path
def load_data(handle="saurabhshahane/wild-blueberry-yield-prediction",output_dir="../data/raw/"):
    return load_from_kaggle(handle,output_dir+handle)