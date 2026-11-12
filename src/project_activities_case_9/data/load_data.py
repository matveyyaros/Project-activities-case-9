# TODO загрузка файла с данными с kaggle
import pandas as pd
import kagglehub
import os

def load_from_kaggle(handle,path,output_dir):
    dataset_path = kagglehub.dataset_download(handle,output_dir=output_dir,force_download=True)
    return dataset_path
def load_data(handle="saurabhshahane/wild-blueberry-yield-prediction",path="Data in Brief/",output_dir="./data/raw/"):
    return load_from_kaggle(handle,path,output_dir+handle)