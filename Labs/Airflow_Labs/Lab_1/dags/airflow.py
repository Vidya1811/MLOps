# dags/airflow.py

import os
import sys

# Ensure /opt/airflow/dags/src is on PYTHONPATH so lab.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from airflow import DAG
from airflow.operators.python import PythonOperator   
from datetime import datetime, timedelta

from lab import load_data, data_preprocessing, build_save_model, load_model_elbow

default_args = {
    "owner": "Vidya",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="Airflow_Lab1",
    default_args=default_args,
    description="Dag example for Lab 1 of Airflow series (modified dataset + artifacts)",
    schedule_interval=None,  # manual trigger
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id="build_save_model_task",
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task


if __name__ == "__main__":
    dag.test()