import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from src.feature_creation import add_features

def drop_id(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    X_train=X_train.copy()
    X_test=X_test.copy()

    if id_col in X_train.columns:
        X_train=X_train.drop(id_col, axis=1)
    if id_col in X_test.columns:
        X_test=X_test.drop(id_col, axis=1)
    
    return X_train, X_test

class GroupImputer(BaseEstimator, TransformerMixin):
    '''Заполняет пропуски статистикой по группе.

    Вся статистика вычисляется только на train в fit(),
    что предотвращает утечку данных при кросс-валидации.
    Если группа не встречалась при обучении — используется глобальный fallback.

    Параметры
    ----------
    group_col : str
        Колонка по которой группируем (например 'Initial').
    target_col : str
        Колонка в которой заполняем пропуски (например 'Age').
    strategy : str
        Стратегия агрегации: 'median' или 'mean'.'''
    def __init__(self, group_col: str, target_col: str, strategy: str = "median"):
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.global_value = None
        self.group_values = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.strategy == "median":
            self.global_value = X[self.target_col].median()
            agg_func = "median"
        elif self.strategy == "mean":
            self.global_value = X[self.target_col].mean()
            agg_func = "mean"
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        self.group_values = (
            X.groupby(self.group_col)[self.target_col]
             .agg(agg_func)
             .to_dict()
        )
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        mask = X[self.target_col].isna()
        X.loc[mask, self.target_col] = (
            X.loc[mask, self.group_col]
             .map(self.group_values)
             .fillna(self.global_value)
        )
        return X

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    ''' Кодирует категориальные и числовые признаки.

    - Sex, Embarked, Initial — ручной маппинг в числа
    - Age_band  — возраст разбитый на 5 бинов (0-16, 16-32, ..., 64-100)
    - Is_Child  — бинарный флаг для пассажиров младше 16 лет
    - Fare_cat  — стоимость билета разбитая на 4 категории по перцентилям
    - Deck      — палуба закодированная в числа (U=0, A=1, ..., G=7)

    Функция stateless — не требует обучения, безопасна для применения
    отдельно на train и test.'''
    df = df.copy()

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Initial'] = df['Initial'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4})

    df['Age_band'] = pd.cut(
        df['Age'],
        bins=[0, 16, 32, 48, 64, 100],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)
    df['Is_Child'] = (df['Age'] < 16).astype(int)

    df['Fare_cat'] = pd.cut(
        df['Fare'],
        bins=[0, 7.91, 14.454, 31, 513],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)
    deck_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'U':0}
    df['Deck'] = df['Deck'].map(deck_map).fillna(0).astype(int)
    return df

def drop_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train=X_train.copy()
    X_test=X_test.copy()
    X_train=X_train.drop(["Cabin","Name","Age","Ticket","Fare"],axis=1)
    X_test=X_test.drop(["Cabin","Name","Age","Ticket","Fare"],axis=1)
    return X_train, X_test


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    id_col: str,
):
    '''Полный preprocessing pipeline без утечки данных.

    Порядок шагов:
    1. Удаление ID колонки.
    2. Feature engineering на объединённом train+test (stateless операции).
    3. Заполнение пропусков — только на train:
       - Age  — медиана по группе титула (Initial)
       - Fare — медиана по классу билета (Pclass)
       - Embarked — мода
    4. Кодирование признаков.
    5. Удаление исходных колонок заменённых новыми признаками.
    6. Выравнивание колонок train и test.'''
    X_train, X_test = drop_id(X_train, X_test, id_col)
    train_rows = X_train.shape[0]
    full_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    full_df = add_features(full_df)
    X_train = full_df.iloc[:train_rows].copy()
    X_test  = full_df.iloc[train_rows:].copy()
    
    age_imputer = GroupImputer(group_col="Initial", target_col="Age")
    age_imputer.fit(X_train)
    X_train = age_imputer.transform(X_train)
    X_test  = age_imputer.transform(X_test)
    fare_imputer = GroupImputer(group_col="Pclass", target_col="Fare")
    fare_imputer.fit(X_train)
    X_train = fare_imputer.transform(X_train)
    X_test  = fare_imputer.transform(X_test)

    embarked_imputer = SimpleImputer(strategy="most_frequent")
    X_train[["Embarked"]] = embarked_imputer.fit_transform(X_train[["Embarked"]])
    X_test[["Embarked"]]  = embarked_imputer.transform(X_test[["Embarked"]])
    X_train = encode_features(X_train)
    X_test  = encode_features(X_test)
    X_train, X_test = drop_columns(X_train, X_test)
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )
    return X_train, X_test