from ultralytics import YOLO
import mlflow
import argparse
import os
import datetime


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


    ziel_projekt = os.path.abspath(f"data/{args.camera}/autolabel_model")
    ziel_name=f"yolo_{args.camera}_run_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    results = model.train(
        data=f"autolabeling/data_{args.camera.lower()}.yaml", 
        epochs=1, 
        imgsz=640, 
        optimizer="MuSGD",
        batch=-1, # Auto-determines best batch size for your GPU
        project=ziel_projekt,
        name=ziel_name, 
        exist_ok=True         
    )

    model.export(format="onnx")