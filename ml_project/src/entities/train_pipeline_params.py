from dataclasses import dataclass
from typing import Union

import yaml
from marshmallow_dataclass import class_schema

from src.entities.features_params import FeaturesParams
from src.entities.preprocess_params import PreprocessParams
from src.entities.split_params import SplitParams
from src.entities.train_params import RFTrainParams, LRTrainParams

TrainingParams = Union[RFTrainParams, LRTrainParams]


@dataclass()
class ModelParams:
    train_params: TrainingParams


@dataclass()
class TrainingParams:
    input_data_path: str
    output_model_path: str
    metrics_path: str
    preds_path: str
    split_params: SplitParams
    preprocess_params: PreprocessParams
    features_params: FeaturesParams


ModelParamsSchema = class_schema(ModelParams)
TrainingParamsSchema = class_schema(TrainingParams)


def read_model_params(path: str) -> ModelParams:
    with open(path, "r") as input_stream:
        schema = ModelParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
