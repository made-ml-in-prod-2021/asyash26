import logging

import click

from src.data.data_preprocess import split_data, preprocess_data, read_data
from src.entities.train_pipeline_params import read_model_params, read_training_params
from src.models.model_eval import check_metrics, save_preds, save_metrics
from src.pipelines.build_pipelines import build_full_pipeline, serialize_pipeline

logger = logging.getLogger("ml_in_prod")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("src/logs/log_file.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train_pipeline(model_config_path: str, train_parameters_config_path: str):
    logger.info("START TRAIN")
    model_params = read_model_params(model_config_path)
    training_params = read_training_params(train_parameters_config_path)

    logger.info("Start reading and preproces data")
    data = read_data(training_params.input_data_path)
    data = preprocess_data(data, training_params.preprocess_params)
    X_train, X_test, y_train, y_test = split_data(
        data, training_params.features_params.target_col,
        training_params.split_params
    )

    logger.info("Start building a pipeline")
    pipline = build_full_pipeline(model_params.train_params, training_params.features_params)

    logger.info("Start fitting the pipeline")
    pipline.fit(X_train, y_train)

    serialize_pipeline(pipline, training_params.output_model_path)

    logger.info("Start making predictions")
    preds = pipline.predict(X_test)
    save_preds(preds, training_params.preds_path)

    logger.info("Start calculating metrics")
    metrics = check_metrics(preds, y_test)
    save_metrics(metrics, training_params.metrics_path)

    logger.info("END TRAIN")


@click.command(name="train_pipeline")
@click.argument("model_config_path")
@click.argument("train_parameters_config_path")
def train_pipeline_command(model_config_path: str, train_parameters_config_path: str):
    train_pipeline(model_config_path, train_parameters_config_path)


if __name__ == "__main__":
    train_pipeline_command()
