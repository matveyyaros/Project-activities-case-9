## Описание

## Быстрый старт

### Установка uv, если его нет

```Shell
pip install uv
```

### Настройка окружения

```Shell
uv sync
```

### Запуск обучения модели и сохранения отчета и модели

```Shell
uv run python  train.py
```

### Выбор параметров

Выбираемые параметры могут быть описаны в conf\config.yaml, модели выбираются из conf\pipeline

**Примеры:**

```
uv run python  train.py seed=57
uv run python train.py pipeline@_global_=BaseLinearRegression
```

## Архитектура

```
Project-activities-case-9/
├── .python-version
├── pyproject.toml          # Зависимости и конфигурация инструментов (uv)
├── uv.lock                 # Единый lockfile для воспроизводимости
├── README.md
├── src/
│   └── project_activities_case_9/         # Основной пакет с кодом
│       ├── __init__.py
│       ├── data/           # Загрузка и предобработка данных
│       ├── features/       # Построение признаков
│       ├── models/         # Определения моделей
│       ├── train/          # Обучение
│       ├── evaluation/     # Метрики и отчеты
│       └── serving/        # Код для инференса 
├── tests/                  # Юнит-тесты (pytest)
├── notebooks/              # Jupyter-ноутбуки (только для исследования)
├── conf/                   # Конфигурации
├── data/                   # Данные
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/                 # Обученные артефакты
└── reports/                # Сгенерированные отчеты и графики
```
