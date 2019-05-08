#!/bin/bash

# this script refreshes the airflow/tfx local databases by deleting
# and recreating them
#
# when developing, having the same DBs but changing the Airflow Pipeline
# steps was leading to failures, that then also affected the TFX metadata.db

# delete TFX artifacts
rm -rf ~/tfx/pipelines/mnist

# delete tf-metadata db
rm ~/tfx/metadata/mnist/metadata.db

# delete airflow db
rm ~/airflow/airflow.db

airflow initdb

echo "running - mnist_pipeline_simple.py"
python ~/Documents/github/tfx/tfx/examples/mnist/mnist_pipeline_simple.py

echo "list_dags"
airflow list_dags
