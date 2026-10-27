"""M5 forecasting pipeline DAG. Each task shells out to an existing, unmodified-logic
script in the project root, so the modelling code path is identical to running the
scripts by hand. `cutoff_date` is an orchestration-level run parameter (for tagging/
logging); the scripts themselves already process every cutoff in `preds_baseline` and
use CREATE OR REPLACE TABLE, so rerunning with the same `cutoff_date` overwrites the same
tables deterministically -- that's what makes the DAG idempotent."""
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Native run (airflow installed on host, as used for local DAG testing): PROJECT_DIR is
# the repo path and PYTHON is its .venv interpreter. In the docker-compose Airflow stack,
# override both via env vars -- PROJECT_DIR=/app, PYTHON_BIN=python -- since the project's
# dependencies are installed directly into that container's image (see Dockerfile).
PROJECT_DIR = os.environ.get("M5_PROJECT_DIR", "/home/gaurav/Desktop/forecasting")
PYTHON = os.environ.get("M5_PYTHON_BIN", f"{PROJECT_DIR}/.venv/bin/python")

default_args = {"owner": "m5-forecasting", "retries": 0}

with DAG(
    dag_id="m5_pipeline",
    description="Weekly M5 demand forecasting: build -> features -> train -> backtest -> register -> forecast",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={"cutoff_date": "latest"},
    tags=["m5", "forecasting"],
) as dag:

    def run(script):
        return BashOperator(
            task_id=script.replace(".py", ""),
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"echo 'cutoff_date param: {{{{ params.cutoff_date }}}}' && "
                f"{PYTHON} {script}"
            ),
        )

    build_weekly = run("build.py")
    check_gaps = run("check_gaps.py")
    build_features = run("build_features_h.py")
    backtest_baselines = run("backtest_baselines.py")
    train_lgbm = run("train_lgbm.py")           # also registers model in MLflow
    train_ets = run("train_ets.py")
    train_quantile = run("train_quantile.py")   # produces q50/q70/q90 forecasts
    evaluate = run("evaluate.py")

    build_weekly >> check_gaps >> build_features >> backtest_baselines
    backtest_baselines >> [train_lgbm, train_ets]
    train_lgbm >> train_quantile
    [train_lgbm, train_ets, train_quantile] >> evaluate
