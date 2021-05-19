import os
from typing import List, Union, Tuple, Dict

import click
import pandas as pd
import requests

HOST = os.environ.get("HOST", default="0.0.0.0")
PORT = os.environ.get("PORT", default=8000)


def get_data_sample(data_file_path: str, sample_size: int):
    data = pd.read_csv(data_file_path, index_col=0)
    sample_data = data.sample(sample_size)
    return sample_data.columns.tolist(), sample_data.values.tolist()


def make_prediction(columns: List[str],
                    values: List[List[Union[float, int, None]]]) \
        -> Tuple[Dict, requests.Response]:
    url = f"http://{HOST}:{PORT}/predict"
    request = {
        'columns': columns,
        'values': values,
    }
    response = requests.get(
        url=url,
        json=request,
    )
    return request, response


@click.command()
@click.option("--data_file_path", default="data/for_preds.csv")
@click.option("--sample_size", default=2)
def make_requests(data_file_path: str, sample_size: int):
    columns, values = get_data_sample(data_file_path, sample_size)
    request, response = make_prediction(columns, values)
    click.echo(f"Request data:\t {request}")
    click.echo(f"Response status code:\t {response.status_code}")
    click.echo(f"Response data:\t {response.json()}")


if __name__ == "__main__":
    make_requests()
