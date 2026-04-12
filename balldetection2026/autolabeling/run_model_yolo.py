from pathlib import Path
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
import requests
from tools import predict_on_image, invert_class_map, CLASS_MAP
import argparse
import sys
import os 
import mlflow 
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, required=True, help="Set BOTTOM or TOP")
    parser.add_argument("-p", "--project", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("-r", "--run_name", type=str, help="Specific MLflow run name to fetch")
    parser.add_argument("-m", "--model", type=str, help="Path to model weights (.pt file)")
    parser.add_argument("-n", "--num_images", type=int, help="Maximum number of images to predict (default: all)")
    
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if args.model is None:
        try:
            mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
            experiment_name = f"GO26-Autolabeling Model-{args.camera}"
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            METRIC_TO_OPTIMIZE = "metrics/mAP50-95B"   
            
            mlflow_client = MlflowClient()
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found.")

            if args.run_name:
                print(f"\n\nLooking for specific run '{args.run_name}' in '{experiment_name}'...")
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{args.run_name}'"
                )
                if not runs:
                    raise ValueError(f"No run found with name '{args.run_name}'.")
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
            download_dir = f"./data/{args.camera}/mlflow_cache"
            
            print("\nDownloading model weights from MLflow...")
            local_model_path = mlflow.artifacts.download_artifacts(
                run_id=target_run.info.run_id,
                artifact_path=artifact_path,
                dst_path=download_dir
            )
            args.model = local_model_path
            print(f"Model weights downloaded to {local_model_path}\n")
            
        except Exception as e:
            print(f"\nMLflow Error - {e}")
            sys.exit(1)
    else:
        print(f"\nUsing model from path {args.model}\n")

    client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )

    print(f"Fetching unlabeled tasks from project {args.project}...")
    all_tasks = list(client.tasks.list(project=args.project))
    unlabeled_tasks = [t for t in all_tasks if not t.annotations]
    print(f"Found {len(unlabeled_tasks)} unlabeled tasks.")

    model = YOLO(args.model)
    CLASS_MAP_INV = invert_class_map(CLASS_MAP)

    classes_this_model_handles = set()
    for idx in model.names.keys():
        label_name = CLASS_MAP_INV.get(int(idx), "Ball") 
        classes_this_model_handles.add(label_name)
    
    pushed, skipped = 0, 0
    
    image_limit_exists = False
    if args.num_images is not None:
        image_limit_exists = True
    
    for task in unlabeled_tasks:
        if image_limit_exists:
            if pushed >= args.num_images:
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
                label_name = CLASS_MAP_INV.get(int(box.cls[0]), "Ball")
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
            if not any(label in classes_this_model_handles for label in existing_labels):
                filtered_predictions.append(old_box)

        final_predictions = filtered_predictions + new_predictions

        if len(final_predictions) > 0:
            mean_score = sum(confidences) / len(confidences) if confidences else 1.0
            predict_on_image(client, task_id=task.id, predictions=final_predictions, score=mean_score)
            pushed += 1

    print(f"\nDone. {pushed} predictions pushed, {skipped} empty or failed.")