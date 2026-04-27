from pathlib import Path
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
import requests
from tools import predict_on_image
import argparse
import sys
import os 
import mlflow 
from mlflow.tracking import MlflowClient
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    camera = str(config["camera"]).upper()
    target_class = config.get("target_class", "Ball")
    project = config["project"]
    run_name = config.get("run_name")
    model = config.get("model")
    num_images = config.get("num_images")
 
    if camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if model is None:
        try:
            mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
            experiment_name = f"{target_class}-{camera}-classifier-model"
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            METRIC_TO_OPTIMIZE = "metrics/mAP50-95B"   
            
            mlflow_client = MlflowClient()
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found.")

            if run_name:
                print(f"\n\nLooking for specific run '{run_name}' in '{experiment_name}'...")
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )
                if not runs:
                    raise ValueError(f"No run found with name '{run_name}'.")
                target_run = runs[0]
                run_name = target_run.data.tags.get("mlflow.runName", "Unnamed Run")
                print(f"Found specific run: '{run_name}'")
                
            else:
                print(f"\n\nGetting best run from '{experiment_name}'...")
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=[f"metrics.`{METRIC_TO_OPTIMIZE}` DESC"],
                    max_results=1
                )
                if not runs:
                    raise ValueError(f"No run found in '{experiment_name}'.")
                target_run = runs[0]
                run_name = target_run.data.tags.get("mlflow.runName", "Unnamed Run")
                best_metric_val = target_run.data.metrics.get(METRIC_TO_OPTIMIZE, "N/A")
                print(f"Found best run: {run_name} (score: {best_metric_val})")

            artifact_path = "weights/best.pt" 
            download_dir = f"./data/{camera}/mlflow_cache"
            
            print("\nDownloading model weights from MLflow...")
            local_model_path = mlflow.artifacts.download_artifacts(
                run_id=target_run.info.run_id,
                artifact_path=artifact_path,
                dst_path=download_dir
            )
            model = local_model_path
            print(f"Model weights downloaded to {local_model_path}\n")
            
        except Exception as e:
            print(f"\nMLflow Error - {e}")
            sys.exit(1)
    else:
        print(f"\nUsing model from path {model}\n")

    client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )

    print(f"Fetching unlabeled tasks from project {project}...")
    all_tasks = list(client.tasks.list(project=project))
    unlabeled_tasks = [t for t in all_tasks if not t.annotations]
    print(f"Found {len(unlabeled_tasks)} unlabeled tasks.")

    model = YOLO(model)
    
    pushed, skipped = 0, 0
    
    image_limit_exists = False
    if num_images is not None:
        image_limit_exists = True
    
    for task in unlabeled_tasks:
        if image_limit_exists:
            if pushed >= num_images:
                break

        image_url = task.data.get("image")
        
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            print(f"Warning: Could not download image for task {task.id}")
            skipped += 1
            continue

        tmp_path = Path(f"/tmp/{task.id}.jpg")
        tmp_path.write_bytes(response.content)

        results =  model.predict(source=str(tmp_path))
        boxes = results[0].boxes

        inspect_dir = Path(f"inspection/")
        inspect_dir.mkdir(parents=True, exist_ok=True)
        results[0].save(filename=str(inspect_dir / f"{task.id}.jpg"))

        tmp_path.unlink()

        boxes = results[0].boxes

        # if boxes is None or len(boxes) == 0:
        #     # Push empty prediction so task is marked as processed in Label Studio
        #     #predict_on_image(client, task_id=task.id, predictions=[], score=0.0)
        #     #pushed += 1
        #     continue
        
        existing_predictions_results = []
        if hasattr(task, 'predictions') and task.predictions:
            for pred in task.predictions:
                try:
                    if pred.result:
                        existing_predictions_results.extend(pred.result)
                    client.predictions.delete(id=pred.id)
                except Exception as e:
                    print(e)

        new_predictions = []
        confidences = [] 

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x_center, y_center, w, h = box.xywhn[0].tolist()
                confidence = float(box.conf[0])
                label_name = target_class
                confidences.append(confidence) 

            new_predictions.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": confidence,
                "value": {
                    "x": (x_center - w / 2) * 100,
                    "y": (y_center - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                    "rotation": 0,
                    "rectanglelabels": [label_name]
                }
            })

        filtered_predictions = []
        for old_box in existing_predictions_results:
            existing_labels = old_box.get('value', {}).get('rectanglelabels', [])
            if target_class not in existing_labels:
                filtered_predictions.append(old_box)

        final_predictions = filtered_predictions + new_predictions

        if len(final_predictions) > 0:
            mean_score = sum(confidences) / len(confidences) if confidences else 1.0
            predict_on_image(client, task_id=task.id, predictions=final_predictions, score=mean_score)
            pushed += 1

    print(f"\nDone. {pushed} predictions pushed, {skipped} empty or failed.")