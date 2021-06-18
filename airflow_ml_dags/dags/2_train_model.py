from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import DEFAULT_ARGS, VOLUMES

with DAG(
        "2_train_model",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(1),
) as dag:
    load_and_split = DockerOperator(
        image="airflow-load-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-load-split",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/split/{{ ds }} "
                "--preprocessed-output-dir /data/preprocessed/{{ ds }} "
                "--scaler-output-dir /data/artefacts/scalers/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    train = DockerOperator(
        image="airflow-train",
        command="--input-data-dir /data/preprocessed/{{ ds }} "
                "--model-output-dir /data/artefacts/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    predict = DockerOperator(
        image="airflow-predict",
        command="--input-data-dir /data/split/{{ ds }} "
                "--input-model-dir /data/artefacts/models/{{ ds }} "
                "--input-scaler-dir /data/artefacts/scalers/{{ ds }} "
                "--output-preds-dir /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    validate = DockerOperator(
        image="airflow-validate",
        command="--input-preds-dir /data/predictions/{{ ds }} "
                "--input-target-dir /data/split/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    load_and_split >> preprocess >> train >> predict >> validate
