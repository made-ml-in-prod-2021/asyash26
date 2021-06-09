import os
import pickle

import click
import pandas as pd


@click.command("predict")
@click.option("--input-data-dir")
@click.option("--input-model-dir")
@click.option("--input-scaler-dir")
@click.option("--output-preds-dir")
def predict(input_data_dir: str, input_model_dir: str, input_scaler_dir: str, output_preds_dir: str):
    val_data = pd.read_csv(os.path.join(input_data_dir, "validation.csv"), index_col=0)
    x_val = val_data.drop(columns=["target"])

    with open(os.path.join(input_scaler_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    transformed_x_val = scaler.transform(x_val)

    with open(os.path.join(input_model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    predictions = model.predict(transformed_x_val)

    os.makedirs(output_preds_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_preds_dir, "predictions.csv"))


if __name__ == '__main__':
    predict()
