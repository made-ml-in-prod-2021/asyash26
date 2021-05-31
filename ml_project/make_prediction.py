import logging

import click

from src.data.data_preprocess import feature_label_split, read_data
from src.entities.predict_params import read_predict_params
from src.models.model_eval import save_preds
from src.pipelines.build_pipelines import deserialize_pipeline

logger = logging.getLogger("ml_in_prod")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("src/logs/log_file.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def predict(config_path: str):
    logger.info("START PREDICT")
    logger.info("Start reading predict data and parameters")
    predict_params = read_predict_params(config_path)
    data = read_data(predict_params.input_data_path)

    logger.info("Start loading the pipeline")
    pipline = deserialize_pipeline(predict_params.pipline_path)

    logger.info("Start data preprocess")
    X_test, y_test = feature_label_split(data, predict_params.target_col)

    logger.info("Start making predictions")
    preds = pipline.predict(X_test)

    logger.info("Start saving predictions")
    save_preds(preds, predict_params.preds_path)

    logger.info("END PREDICT")


@click.command(name="predict")
@click.argument("config_path")
def predict_command(config_path: str):
    predict(config_path)


if __name__ == "__main__":
    predict_command()
