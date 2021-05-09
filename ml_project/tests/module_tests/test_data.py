from typing import List

import numpy as np
import pandas as pd
from src.data.data_preprocess import read_data, preprocess_data, feature_label_split, split_data


def test_read_data(generated_data_path: str, target_col: str, gen_data_size: int,
                   categorical_features: List[str], numerical_features: List[str]):
    data = read_data(generated_data_path)
    expected_columns = set(list(target_col.split()) + categorical_features + numerical_features)
    real_columns = set(data.columns)
    assert len(data) == gen_data_size, "Unexpected data size"
    assert type(data) is pd.DataFrame, "Unexpected data type"
    assert expected_columns == real_columns, "Unexpected columns"


def test_preprocess_data_duplicates(generated_data, drop_duplicates_preprocess_params):
    generated_data = generated_data \
        .append(generated_data.iloc[0], ignore_index=True)
    result = preprocess_data(generated_data, drop_duplicates_preprocess_params)
    assert len(generated_data) - 1 == len(result), "Unexpected data size"


def test_preprocess_data_nan(generated_data, drop_nan_preprocess_params):
    generated_data.iloc[0] = np.nan
    result = preprocess_data(generated_data, drop_nan_preprocess_params)
    assert len(generated_data) - 1 == len(result), "Unexpected data size"


def test_feature_label_split(generated_data, target_col: str, gen_data_size: int,
                             categorical_features: List[str], numerical_features: List[str]):
    features_columns = categorical_features + numerical_features
    features, labels = feature_label_split(generated_data, target_col)
    assert features.shape == (gen_data_size, len(features_columns)), "Unexpected features shape"
    assert labels.shape == (gen_data_size,), "Unexpected labels shape"
    assert set(features.columns) == set(features_columns), "Unexpected features columns"
    assert labels.name == target_col, "Unexpected target column name"


def test_split_data(generated_data, split_params, gen_data_size, target_col):
    X_train, X_test, y_train, y_test = \
        split_data(generated_data, target_col, split_params)
    assert X_test.shape[0] == gen_data_size * split_params.val_size, "Unexpected X_test shape"
    assert X_train.shape[0] == gen_data_size * (1 - split_params.val_size), "Unexpected X_tran shape"
    assert y_test.shape[0] == gen_data_size * split_params.val_size, "Unexpected y_test shape"
    assert y_train.shape[0] == gen_data_size * (1 - split_params.val_size), "Unexpected y_train shape"
