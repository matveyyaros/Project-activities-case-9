"""
Скрипт для агрегации результатов моделей машинного обучения.

Функционал:
- Обходит папки с результатами различных архитектур моделей в PATH_folder
- Извлекает все комбинации гиперпараметров из grid_search.json каждой архитектуры
- Создаёт агрегированный файл со всеми комбинациями всех моделей
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore')

PATH_folder = "reports\\multirun\\2026-11-20\\12-11-23"


def find_architecture_folders(base_path: str) -> List[Path]:
    """
    Находит все папки с архитектурами моделей в указанной директории.
    
    Папка архитектуры должна содержать подпапку 'report' с файлом grid_search.json.
    
    Args:
        base_path: Путь к базовой папке с результатами
        
    Returns:
        Список путей к папкам архитектур
    """
    base_dir = Path(base_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Папка {base_path} не существует")
    
    architecture_folders = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name != 'total_result':
            grid_search_file = item / 'report' / 'grid_search.json'
            if grid_search_file.exists():
                architecture_folders.append(item)
    
    return architecture_folders


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Загружает JSON файл."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке {file_path}: {e}")
        return None


def extract_architecture_name(architecture_folder: Path) -> str:
    """
    Извлекает название архитектуры из имени папки.
    
    Имя папки имеет формат: ModelName_PreprocessingType
    """
    return architecture_folder.name


def extract_scoring_metric(grid_search_data: Dict) -> str:
    """Извлекает название scoring метрики из данных grid_search."""
    return grid_search_data.get('scoring', 'unknown')


def extract_cv_split_info(grid_search_data: Dict) -> Dict[str, Any]:
    """
    Извлекает информацию о параметрах разбиения кросс-валидации.
    
    Args:
        grid_search_data: Данные из grid_search.json
        
    Returns:
        Словарь с информацией о разбиении
    """
    cv_info = {
        'cv_type': None,
        'n_splits': None,
        'shuffle': None,
        'random_state': None,
        'cv_raw': None
    }
    
    # Получаем строковое представление CV
    cv_raw = grid_search_data.get('cv', '')
    cv_info['cv_raw'] = cv_raw
    
    # Парсим строку CV (например, "KFold(n_splits=10, random_state=1911, shuffle=True)")
    if cv_raw:
        # Определяем тип кросс-валидации
        if 'KFold' in cv_raw:
            cv_info['cv_type'] = 'KFold'
        elif 'StratifiedKFold' in cv_raw:
            cv_info['cv_type'] = 'StratifiedKFold'
        elif 'RepeatedKFold' in cv_raw:
            cv_info['cv_type'] = 'RepeatedKFold'
        elif 'LeaveOneOut' in cv_raw:
            cv_info['cv_type'] = 'LeaveOneOut'
        else:
            cv_info['cv_type'] = 'Unknown'
        
        # Извлекаем n_splits из строки или из поля n_splits
        if 'n_splits' in grid_search_data:
            cv_info['n_splits'] = grid_search_data['n_splits']
        elif 'n_splits=' in cv_raw:
            try:
                n_splits_str = cv_raw.split('n_splits=')[1].split(',')[0].split(')')[0]
                cv_info['n_splits'] = int(n_splits_str)
            except:
                pass
        
        # Извлекаем shuffle
        if 'shuffle=True' in cv_raw:
            cv_info['shuffle'] = True
        elif 'shuffle=False' in cv_raw:
            cv_info['shuffle'] = False
        
        # Извлекаем random_state
        if 'random_state=' in cv_raw:
            try:
                random_state_str = cv_raw.split('random_state=')[1].split(',')[0].split(')')[0]
                cv_info['random_state'] = int(random_state_str)
            except:
                pass
    
    return cv_info


def save_cv_split_info(cv_info: Dict[str, Any], base_path: str):
    """
    Сохраняет информацию о параметрах разбиения в отдельный файл.
    
    Args:
        cv_info: Словарь с информацией о разбиении
        base_path: Базовый путь для создания папки total_result
    """
    total_result_dir = Path(base_path) / 'total_result'
    total_result_dir.mkdir(parents=True, exist_ok=True)
    
    cv_info_path = total_result_dir / 'cv_split_info.json'
    
    with open(cv_info_path, 'w', encoding='utf-8') as f:
        json.dump(cv_info, f, ensure_ascii=False, indent=2)
    
    print(f"Информация о разбиении сохранена в {cv_info_path}")


def extract_cv_results(grid_search_data: Dict) -> List[Dict]:
    """
    Извлекает результаты всех комбинаций из cv_results.
    
    Returns:
        Список словарей с данными по каждой комбинации
    """
    cv_results = grid_search_data.get('cv_results', [])
    
    if not cv_results:
        print("  Предупреждение: cv_results пуст")
        return []
    
    combinations = []
    for result in cv_results:
        combination = {
            'hyperparameters': result.get('params', {}),
            'train_score_mean': result.get('mean_train_score'),
            'train_score_std': result.get('std_train_score'),
            'test_score_mean': result.get('mean_test_score'),
            'test_score_std': result.get('std_test_score'),
        }
        combinations.append(combination)
    
    return combinations


def aggregate_results(base_path: str) -> tuple:
    """
    Агрегирует результаты всех архитектур моделей.
    
    Args:
        base_path: Путь к папке с результатами моделей
        
    Returns:
        Кортеж (DataFrame с агрегированными результатами, словарь с информацией о разбиении)
    """
    architecture_folders = find_architecture_folders(base_path)
    
    if not architecture_folders:
        print(f"Не найдено папок с архитектурами моделей в {base_path}")
        return pd.DataFrame(), None
    
    print(f"Найдено {len(architecture_folders)} архитектур моделей")
    
    all_results = []
    cv_info = None
    
    for arch_folder in architecture_folders:
        architecture_name = extract_architecture_name(arch_folder)
        print(f"Обработка: {architecture_name}")
        
        grid_search_file = arch_folder / 'report' / 'grid_search.json'
        grid_search_data = load_json_file(grid_search_file)
        
        if not grid_search_data:
            print(f"  Ошибка: не удалось загрузить grid_search.json")
            continue
        
        scoring_metric = extract_scoring_metric(grid_search_data)
        combinations = extract_cv_results(grid_search_data)
        
        # Извлекаем информацию о разбиении только один раз (все модели используют одинаковое разбиение)
        if cv_info is None:
            cv_info = extract_cv_split_info(grid_search_data)
        
        print(f"  Найдено {len(combinations)} комбинаций гиперпараметров")
        
        for combo in combinations:
            result = {
                'architecture': architecture_name,
                'hyperparameters': combo['hyperparameters'],
                'scoring_metric': scoring_metric,
                'train_score_mean': combo['train_score_mean'],
                'train_score_std': combo['train_score_std'],
                'test_score_mean': combo['test_score_mean'],
                'test_score_std': combo['test_score_std'],
            }
            all_results.append(result)
    
    if not all_results:
        print("Нет данных для агрегации")
        return pd.DataFrame(), cv_info
    
    # Создаём DataFrame
    df = pd.DataFrame(all_results)
    
    # Преобразуем гиперпараметры в строку JSON для читаемости
    df['hyperparameters'] = df['hyperparameters'].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
    )
    
    # Сортируем по test_score_mean в порядке убывания
    df = df.sort_values(by='test_score_mean', ascending=False).reset_index(drop=True)
    
    return df, cv_info


def save_results(df: pd.DataFrame, base_path: str, output_format: str = 'csv'):
    """
    Сохраняет агрегированные результаты в папку total_result.
    
    Args:
        df: DataFrame с результатами
        base_path: Базовый путь для создания папки total_result
        output_format: Формат выходного файла ('csv', 'json', 'excel', 'all')
    """
    if df.empty:
        print("Нет данных для сохранения")
        return
    
    # Создаём папку total_result
    total_result_dir = Path(base_path) / 'total_result'
    total_result_dir.mkdir(parents=True, exist_ok=True)
    
    base_output = total_result_dir / 'aggregated_results'
    
    if output_format in ('csv', 'all'):
        csv_path = str(base_output) + '.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Результаты сохранены в {csv_path}")
    
    if output_format in ('json', 'all'):
        json_path = str(base_output) + '.json'
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"Результаты сохранены в {json_path}")
    
    if output_format in ('excel', 'all'):
        try:
            excel_path = str(base_output) + '.xlsx'
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"Результаты сохранены в {excel_path}")
        except ImportError:
            print("Для сохранения в Excel требуется библиотека openpyxl")


def main():
    """Основная функция скрипта."""
    print(f"Агрегация результатов из: {PATH_folder}")
    print("-" * 60)
    
    # Агрегируем результаты
    df, cv_info = aggregate_results(PATH_folder)
    
    if not df.empty:
        # Сохраняем результаты
        save_results(df, PATH_folder, output_format='csv')
        
        # Сохраняем информацию о разбиении
        if cv_info:
            save_cv_split_info(cv_info, PATH_folder)
        
        print("-" * 60)
        print(f"Всего обработано комбинаций: {len(df)}")
        print(f"Уникальных архитектур: {df['architecture'].nunique()}")
        
        if cv_info:
            print(f"\nПараметры разбиения:")
            print(f"  Тип кросс-валидации: {cv_info.get('cv_type', 'N/A')}")
            print(f"  Количество фолдов: {cv_info.get('n_splits', 'N/A')}")
            print(f"  Shuffle: {cv_info.get('shuffle', 'N/A')}")
            print(f"  Random state: {cv_info.get('random_state', 'N/A')}")
        
        print("\nЛучшие результаты по каждой архитектуре:")
        best_results = df.loc[df.groupby('architecture')['test_score_mean'].idxmax()]
        print(best_results[['architecture', 'test_score_mean', 'train_score_mean']].to_string(index=False))
    else:
        print("Результаты не найдены")


if __name__ == '__main__':
    main()
