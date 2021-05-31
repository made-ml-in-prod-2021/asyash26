import os

from src.data.data_preprocess import split_data, preprocess_data, read_data
from src.entities.train_pipeline_params import read_model_params, read_training_params
from src.models.model_eval import check_metrics, save_preds, save_metrics
from src.pipelines.build_pipelines import build_full_pipeline, serialize_pipeline


def test_train_pipeline(model_config_path, train_config_path):
    
    model_params = read_model_params(model_config_path)
    training_params = read_training_params(train_config_path)

    data = read_data(training_params.input_data_path)
    data = preprocess_data(data, training_params.preprocess_params)
    X_train, X_test, y_train, y_test = split_data(
        data, training_params.features_params.target_col,
        training_params.split_params
    )

    pipline = build_full_pipeline(model_params.train_params, training_params.features_params)
    pipline.fit(X_train, y_train)
    serialize_pipeline(pipline, training_params.output_model_path)

    preds = pipline.predict(X_test)
    save_preds(preds, training_params.preds_path)

    metrics = check_metrics(preds, y_test)
    save_metrics(metrics, training_params.metrics_path)

    assert metrics['accuracy_score'] > 0.7
    assert os.path.exists(training_params.output_model_path), "Model file was not created"
    assert os.path.exists(training_params.preds_path), "Preds file was not created"
    assert os.path.exists(training_params.metrics_path), "Metrics file was not created"
