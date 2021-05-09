import logging

import click
from sklearn.pipeline import Pipeline
from src.data.data_preprocess import split_data, preprocess_data, read_data
from src.entities.train_pipeline_params import read_training_pipeline_params
from src.models.model_eval import check_metrics, save_preds, save_metrics
from src.pipelines.build_pipelines import build_full_pipeline, serialize_pipeline

logger = logging.getLogger("ml_in_prod")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/log_file.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train_pipeline(config_path: str):
    logger.info("START TRAIN")
    training_pipeline_params = read_training_pipeline_params(config_path)

    logger.info("Start reading and preproces data")
    data = read_data(training_pipeline_params.input_data_path)
    data = preprocess_data(data, training_pipeline_params.preprocess_params)
    X_train, X_test, y_train, y_test = split_data(
        data, training_pipeline_params.features_params.target_col,
        training_pipeline_params.split_params
    )

    logger.info("Start building a pipeline")
    pipline = build_full_pipeline(training_pipeline_params.train_params, training_pipeline_params.features_params)

    logger.info("Start fitting the pipeline")
    pipline.fit(X_train, y_train)

    serialize_pipeline(pipline, training_pipeline_params.output_model_path)

    logger.info("Start making predictions")
    preds = pipline.predict(X_test)
    save_preds(preds, training_pipeline_params.preds_path)

    logger.info("Start calculating metrics")
    metrics = check_metrics(pipline, preds, y_test)
    save_metrics(metrics, training_pipeline_params.metrics_path)

    logger.info("END TRAIN")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
