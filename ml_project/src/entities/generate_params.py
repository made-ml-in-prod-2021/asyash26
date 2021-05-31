from dataclasses import dataclass
from typing import List

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class GenerateDataParams:
    input_data_path: str
    output_data_path: str
    categorical_features: List[str]
    numerical_features: List[str]
    size: int


GenerateDataParamsSchema = class_schema(GenerateDataParams)


def read_generate_data_params(path: str) -> GenerateDataParams:
    with open(path, "r") as input_stream:
        schema = GenerateDataParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
