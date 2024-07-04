# neuralforge/track/experiment.py
import mlflow
import mlflow.sklearn
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_model(self, model, model_name):
        mlflow.sklearn.log_model(model, model_name)

    def start_run(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

# Usage example:
tracker = ExperimentTracker("MyExperiment")
tracker.start_run()
tracker.log_params({"learning_rate": 0.01, "batch_size": 32})
tracker.log_metric("accuracy", 0.95)
tracker.log_model(my_model, "my_model")
tracker.end_run()