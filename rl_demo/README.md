# RL Demo with MLFlow
We want to use mlflow to track trainings progress. This code shows two ways of doing this. `main.py` shows the basic example. `robust.py` can handle network or mlflow outages and makes sure training can still work.

You need to set a few environment variables:
```
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=
export MLFLOW_USER= 
```

password is the same as the labs wifi. Ask in slack if you don't know it
Please set MLFLOW_USER to your name. This is for tracking which trainings run belongs to whom.