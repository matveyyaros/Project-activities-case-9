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
