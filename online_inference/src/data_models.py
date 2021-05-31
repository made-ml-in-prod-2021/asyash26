from pydantic import BaseModel
from typing import List, Union
import pandas as pd


class InputData(BaseModel):
    columns: List[str]
    values: List[List[Union[float, int, None]]]

    def convert_to_pandas(self) -> pd.DataFrame:
        data = pd.DataFrame(self.values, columns=self.columns)
        return data


class OutputData(BaseModel):
    target: int
