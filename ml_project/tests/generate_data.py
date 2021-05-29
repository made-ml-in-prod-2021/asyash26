import sys
from typing import Dict
from typing import List

import click
import numpy as np
import pandas as pd

sys.path.append("..")

from src import read_generate_data_params, read_data


def generate_numerical_data(numerical_features: List[str], data: pd.DataFrame,
                            size: int) -> Dict[str, np.ndarray]:
    fake_numerical_data = dict()
    for num_feature in numerical_features:
        dist_mean = data[num_feature].mean()
        dist_std = data[num_feature].std()
        feature_type = data[num_feature].dtype
        fake_numerical_data[num_feature] = np.random.normal(dist_mean, dist_std, size).astype(feature_type)
    return fake_numerical_data


def generate_categorical_data(categorical_features: List[str], data: pd.DataFrame,
                              size: int) -> Dict[str, np.ndarray]:
    fake_categorical_data = dict()
    for cat_feature in categorical_features:
        cat_data_dist = data[cat_feature].value_counts(normalize=True)
        fake_categorical_data[cat_feature] = np.random.choice(
            a=list(cat_data_dist.keys()),
            size=size,
            p=list(cat_data_dist)
        )
    return fake_categorical_data


def generate_data(size: int, numerical_features: List[str],
                  categorical_features: List[str], data: pd.DataFrame) -> pd.DataFrame:
    fake_numerical_data = generate_numerical_data(numerical_features, data, size)
    fake_categorical_data = generate_categorical_data(categorical_features, data, size)
    all_fake_data = fake_numerical_data.copy()
    all_fake_data.update(fake_categorical_data)
    df = pd.DataFrame.from_dict(all_fake_data)
    return df


def generate_and_save_fake_data(config_path: str):
    generate_data_params = read_generate_data_params(config_path)
    data = read_data(generate_data_params.input_data_path)
    generated_data = generate_data(generate_data_params.size,
                                   generate_data_params.numerical_features,
                                   generate_data_params.categorical_features,
                                   data)
    generated_data.to_csv(generate_data_params.output_data_path, index=False)


@click.command(name="generate_and_save_fake_data")
@click.argument("config_path")
def main(config_path: str):
    generate_and_save_fake_data(config_path)


if __name__ == "__main__":
    main()
