from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from src.entities.preprocess_params import PreprocessParams
from src.entities.split_params import SplitParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def preprocess_data(data: pd.DataFrame, params: PreprocessParams) \
        -> pd.DataFrame:
    if params.drop_nan:
        data = data.dropna()
    if params.drop_duplicates:
        data = data.drop_duplicates()
    return data


def feature_label_split(data: pd.DataFrame, target_col: str) \
        -> Tuple[pd.DataFrame, pd.Series]:
    features = list(data.columns)
    features.remove(target_col)
    return [data[features], data[target_col]]


def split_data(data: pd.DataFrame, target_col: str, params: SplitParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = feature_label_split(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.val_size, random_state=params.random_state
    )
    return X_train, X_test, y_train, y_test
