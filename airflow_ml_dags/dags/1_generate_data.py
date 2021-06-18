from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from utils import DEFAULT_ARGS, VOLUMES

with DAG(
        "1_generate_data",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command="--output-dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        volumes=VOLUMES
    )
