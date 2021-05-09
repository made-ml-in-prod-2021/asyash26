import os

from sklearn.pipeline import Pipeline
from src.data.data_preprocess import split_data, preprocess_data, read_data
from src.entities.train_pipeline_params import read_training_pipeline_params
from src.models.model_eval import check_metrics, save_preds, save_metrics
from src.pipelines.build_pipelines import build_full_pipeline, serialize_pipeline


def test_train_pipeline(train_config_path):
    training_pipeline_params = read_training_pipeline_params(train_config_path)

    data = read_data(training_pipeline_params.input_data_path)
    data = preprocess_data(data, training_pipeline_params.preprocess_params)
    X_train, X_test, y_train, y_test = split_data(
        data, training_pipeline_params.features_params.target_col,
        training_pipeline_params.split_params
    )

    pipline = build_full_pipeline(training_pipeline_params.train_params, training_pipeline_params.features_params)
    pipline.fit(X_train, y_train)
    serialize_pipeline(pipline, training_pipeline_params.output_model_path)

    preds = pipline.predict(X_test)
    save_preds(preds, training_pipeline_params.preds_path)

    metrics = check_metrics(pipline, preds, y_test)
    save_metrics(metrics, training_pipeline_params.metrics_path)

    assert metrics['accuracy_score'] > 0.7
    assert os.path.exists(training_pipeline_params.output_model_path), "Model file was not created"
    assert os.path.exists(training_pipeline_params.preds_path), "Preds file was not created"
    assert os.path.exists(training_pipeline_params.metrics_path), "Metrics file was not created"
