import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-data-dir")
@click.option("--model-output-dir")
def train(input_data_dir: str, model_output_dir: str):
    X_train = pd.read_csv(os.path.join(input_data_dir, "preprocessed_features.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(input_data_dir, "target.csv"), index_col=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
