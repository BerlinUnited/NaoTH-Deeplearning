from ultralytics import YOLO
import mlflow
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"GO26-Autolabeling Model-{args.camera}"
    
    # Load the YOLO26 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    model = YOLO("yolo26n.pt") 

    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    mlflow.set_experiment(f"GO26-Autolabeling Model-{args.camera}")

    results = model.train(
        data=f"data_{args.camera.lower()}.yaml", 
        epochs=500, 
        imgsz=640, 
        optimizer="MuSGD",
        batch=-1           # Auto-determines best batch size for your GPU
    )

    model.export(format="onnx")