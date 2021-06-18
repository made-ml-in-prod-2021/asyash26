import json
import os

import click
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


@click.command("validate")
@click.option("--input-preds-dir")
@click.option("--input-target-dir")
def validate(input_preds_dir: str, input_target_dir: str):
    preds = pd.read_csv(os.path.join(input_preds_dir, "predictions.csv"), index_col=0)
    val_data = pd.read_csv(os.path.join(input_target_dir, "validation.csv"), index_col=0)
    target = val_data['target']
    accuracy = accuracy_score(target, preds)
    roc_auc = roc_auc_score(target, preds)
    precision = precision_score(target, preds)
    recall = recall_score(target, preds)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall

    }
    with open(os.path.join(input_preds_dir, "metrics.json"), 'w') as fp:
        json.dump(metrics, fp)


if __name__ == '__main__':
    validate()
