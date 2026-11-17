# TODO загрузка файла с данными с kaggle
import pandas as pd
import kagglehub
import os

def load_from_kaggle(handle,path,output_dir):
    dataset_path = kagglehub.dataset_download(handle,output_dir=output_dir)
    return dataset_path
def load_data(
    handle="saurabhshahane/wild-blueberry-yield-prediction",
    path="Data in Brief/",
    output_dir="./data/raw/",
):
    target_dir = os.path.join(output_dir, handle)

    # Проверка: папка уже существует и в ней что-то есть
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[skip] Данные уже загружены: {target_dir}")
        return target_dir

    print(f"[load] Загружаем данные в {target_dir} ...")
    return load_from_kaggle(handle, path, target_dir)