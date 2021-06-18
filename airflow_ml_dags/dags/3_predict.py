from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from utils import DEFAULT_ARGS, VOLUMES

with DAG(
        "3_predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    date = Variable.set("TARGET_DATE", "2021-06-09")
    input_data_sensor = FileSensor(
        task_id="input_data_exists",
        poke_interval=10,
        retries=10,
        filepath="data/split/{{ ds }}/validation.csv",
    )
    model_sensor = FileSensor(
        task_id="model_exists",
        poke_interval=10,
        retries=10,
        filepath="data/artefacts/models/{{ var.value.TARGET_DATE }}/model.pkl",
    )
    scaler_sensor = FileSensor(
        task_id="scaler_exists",
        poke_interval=10,
        retries=10,
        filepath="data/artefacts/scalers/{{ var.value.TARGET_DATE }}/scaler.pkl",
    )
    predictions_sensor = FileSensor(
        task_id="predictions_exist",
        poke_interval=10,
        retries=10,
        filepath="data/predictions/{{ ds }}/predictions.csv",
    )
    predict = DockerOperator(
        image="airflow-predict",
        command="--input-data-dir /data/split/{{ ds }} "
                "--input-model-dir data/artefacts/models/{{ var.value.TARGET_DATE }} "
                "--input-scaler-dir data/artefacts/scalers/{{ var.value.TARGET_DATE }} "
                "--output-preds-dir /data/predictions/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=VOLUMES
    )
    [input_data_sensor, model_sensor, scaler_sensor, predictions_sensor] >> predict
