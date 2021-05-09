from typing import List

from dataclasses import dataclass, field


@dataclass()
class FeaturesParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str
