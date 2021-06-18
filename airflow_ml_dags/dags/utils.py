from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "Anastasiia Shevchuk",
    "email": ["some_email@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=0.5),
}

VOLUMES = ["D:/MADE/2/MLinPROD/airflow_ml_dags/data:/data"]