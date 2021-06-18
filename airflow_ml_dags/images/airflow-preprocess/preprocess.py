import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--preprocessed-output-dir")
@click.option("--scaler-output-dir")
def preprocess(input_dir: str, preprocessed_output_dir: str, scaler_output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train.csv"), index_col=0)
    features, target = data.drop(columns=["target"]), data['target']

    scaler = StandardScaler()
    transformed_features = pd.DataFrame(scaler.fit_transform(features))

    os.makedirs(scaler_output_dir, exist_ok=True)
    with open(os.path.join(scaler_output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    os.makedirs(preprocessed_output_dir, exist_ok=True)
    transformed_features.to_csv(os.path.join(preprocessed_output_dir, "preprocessed_features.csv"))
    target.to_csv(os.path.join(preprocessed_output_dir, "target.csv"))


if __name__ == '__main__':
    preprocess()
