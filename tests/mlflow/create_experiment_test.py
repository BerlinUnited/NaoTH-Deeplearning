"""
    Creates an experiment if it does not already exist
"""
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="https://mlflow.berlin-united.com/")

all_experiments = client.search_experiments()

experiment_name = "Tests"
experiment_description = (
    "Dummy Experiment that should be used for all tests"
)
experiment_tags = {
    "project_name": "tests",
    "mlflow.note.content": experiment_description,
}

has_match = any(obj.name == "experiment_name" for obj in all_experiments)
print("has_match", has_match)

if not has_match:
    # Create the Experiment, providing a unique name
    test_experiment = client.create_experiment(
        name=experiment_name, tags=experiment_tags
    )

all_experiments = client.search_experiments()
print(all_experiments)
