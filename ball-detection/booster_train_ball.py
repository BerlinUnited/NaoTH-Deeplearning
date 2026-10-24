from ultralytics import YOLO
import mlflow
import os

model = YOLO("yolov8n.pt")


mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
mlflow_experiment_name = "booster_ball_model"
os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

mlflow.set_experiment(mlflow_experiment_name)

with mlflow.start_run():
    results = model.train(
        data="/lustre/informatik/schlotts/yolo_balls_dataset/dataset.yaml",
        epochs=1000,
        imgsz=[544, 448], # super important to get this right
        optimizer="MuSGD",
        batch=-1,  # Auto-determines best batch size for your GPU
        exist_ok=True,
        scale=0.2, degrees=0.0,
        box=10,
    )

    run_dir = model.trainer.save_dir
    best_model_path = os.path.join(run_dir, "weights", "best.pt")

    print(f"Training finished. Best model located at: {best_model_path}")

    # Log the best model file as an MLflow artifact
    if os.path.exists(best_model_path):
        mlflow.log_artifact(best_model_path, artifact_path="model_weights")
        
        # Optional: Log the entire weights directory (contains last.pt, etc.)
        # mlflow.log_artifacts(os.path.join(run_dir, "weights"), artifact_path="weights_dir")
    else:
        print(f"Warning: Could not find best model at {best_model_path}")
