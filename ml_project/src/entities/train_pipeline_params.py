from typing import List
from typing import Union

import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from src.entities.features_params import FeaturesParams
from src.entities.preprocess_params import PreprocessParams
from src.entities.split_params import SplitParams
from src.entities.train_params import RFTrainParams, LRTrainParams

TrainingParams = Union[RFTrainParams, LRTrainParams]


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metrics_path: str
    preds_path: str
    split_params: SplitParams
    preprocess_params: PreprocessParams
    features_params: FeaturesParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
