#!/bin/bash

rm ~/tfx/metadata/mnist/metadata.db
rm ~/airflow/airflow.db

airflow initdb

echo "running - mnist_pipeline_simple.py"
python ~/Documents/github/tfx/tfx/examples/mnist/mnist_pipeline_simple.py

echo "list_dags"
airflow list_dags
