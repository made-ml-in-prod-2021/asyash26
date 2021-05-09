import logging
import os

import click
from sklearn.pipeline import Pipeline
from src.data.data_preprocess import feature_label_split, read_data
from src.entities.predict_params import read_predict_params
from src.models.model_eval import check_metrics, save_preds, save_metrics
from src.pipelines.build_pipelines import deserialize_pipeline


def test_make_prediction(predcit_config_path):
    predict_params = read_predict_params(predcit_config_path)
    data = read_data(predict_params.input_data_path)
    pipline = deserialize_pipeline(predict_params.pipline_path)
    X_test, y_test = feature_label_split(data, predict_params.target_col)
    preds = pipline.predict(X_test)
    save_preds(preds, predict_params.preds_path)

    assert os.path.exists(predict_params.preds_path), "Preds file was not created"
