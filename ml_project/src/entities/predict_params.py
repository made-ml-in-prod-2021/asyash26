from dataclasses import dataclass
from typing import Union

import yaml
from marshmallow_dataclass import class_schema

from src.entities.train_params import RFTrainParams, LRTrainParams

TrainingParams = Union[RFTrainParams, LRTrainParams]


@dataclass()
class PredictParams:
    input_data_path: str
    pipline_path: str
    preds_path: str
    target_col: str


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str) -> PredictParamsSchema:
    with open(path, "r") as input_stream:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
