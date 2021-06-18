import os

import click
from sklearn.datasets import load_breast_cancer


@click.command("generate")
@click.option("--output-dir")
def generate(output_dir: str):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_sample = X.sample(100)
    y_sample = y[X_sample.index]

    os.makedirs(output_dir, exist_ok=True)
    X_sample.to_csv(os.path.join(output_dir, "features.csv"))
    y_sample.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    generate()
