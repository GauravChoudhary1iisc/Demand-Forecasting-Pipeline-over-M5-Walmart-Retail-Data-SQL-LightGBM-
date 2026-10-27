import mlflow

TRACKING_URI = "sqlite:///mlflow.db"


def setup(experiment):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
