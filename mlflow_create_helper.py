import mlflow

from config import mlflow_tracking_uri

mlflow.set_tracking_uri(mlflow_tracking_uri)


for i in range(100):
    experiment_name = f"find_figure_{i + 1:03d}"
    mlflow.set_experiment(experiment_name)
