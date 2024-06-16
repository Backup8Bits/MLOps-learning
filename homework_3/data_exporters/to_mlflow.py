import mlflow

from mlflow.tracking import MlflowClient

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    EXPERIMENT_NAME = "yellow_trips_experiments"

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.sklearn.log_model(data[1], "lineal_model")
        mlflow.log_artifact(data[0], "dv")