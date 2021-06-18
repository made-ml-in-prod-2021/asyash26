import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("load_and_split")
@click.option("--input-dir")
@click.option("--output-dir")
def load_and_split(input_dir: str, output_dir: str):
    features = pd.read_csv(os.path.join(input_dir, "features.csv"), index_col=0)
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), index_col=0)
    data = features.merge(target, left_index=True, right_index=True)
    train, validation = train_test_split(data, test_size=0.3)

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"))
    validation.to_csv(os.path.join(output_dir, "validation.csv"))


if __name__ == '__main__':
    load_and_split()
