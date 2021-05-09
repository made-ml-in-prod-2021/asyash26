import json
from typing import Dict
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def check_metrics(pipline: Pipeline, preds: np.ndarray,
                  true_labels: pd.Series) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(true_labels, preds),
        "roc_auc_score": roc_auc_score(true_labels, preds),
        "f1_score": f1_score(true_labels, preds),
    }


def save_preds(preds: np.ndarray, save_preds_path: str) -> NoReturn:
    preds_df = pd.DataFrame({'target': preds})
    preds_df.to_csv(save_preds_path, index=False)


def save_metrics(metrics: Dict[str, float], save_metrics_path: str) -> NoReturn:
    with open(save_metrics_path, "w") as metric_file:
        json.dump(metrics, metric_file)
