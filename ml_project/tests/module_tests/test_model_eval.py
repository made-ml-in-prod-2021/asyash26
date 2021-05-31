import os

import numpy as np
from src.models.model_eval import check_metrics, save_preds, save_metrics


def test_check_metrics(preds_and_labels):
    metrics = check_metrics(preds_and_labels[0], preds_and_labels[1])
    assert list(metrics.keys()) == ['accuracy_score', 'roc_auc_score', 'f1_score'], "Unexpected metrics keys"
    assert set(list(map(type, list(metrics.values())))) == {np.float64}, "Unexpected metrics values type"


def test_save_preds(preds_and_labels, preds_out_path):
    save_preds(preds_and_labels[0], preds_out_path)
    assert os.path.isfile(preds_out_path), "File was not created"


def test_save_metrics(metrics, metrics_out_path):
    save_metrics(metrics, metrics_out_path)
    assert os.path.isfile(metrics_out_path), "File was not created"
