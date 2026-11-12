import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class ColumnScaler(BaseEstimator, TransformerMixin):
    """Масштабирует только указанные столбцы, остальные оставляет без изменений."""
    
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Обучаем внутренний scaler только на выбранных столбцах
        self.scaler.fit(X[self.columns], y)
        # Сохраняем порядок исходных колонок
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        # Важно: создаем копию, чтобы не изменять исходный DataFrame
        X_copy = X.copy()
        # Применяем масштабирование к выбранным столбцам
        X_copy[self.columns] = self.scaler.transform(X_copy[self.columns])
        return X_copy