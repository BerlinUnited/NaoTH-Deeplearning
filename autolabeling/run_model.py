from pathlib import Path
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
import requests
from tools import predict_on_image, get_log_ids_per_game, get_log_ids_per_event
import argparse
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    target_class = config["target_class"]
    if not target_class:
        raise ValueError("You must provide a target class in your config.")
    camera = str(config["camera"]).upper()
    if not camera:
        raise ValueError("You must provide a camera in your config.")

    event_ids = config.get("event_ids")
    game_ids = config.get("game_ids")

    log_ids = config.get("log_ids")
    ls_project_ids = config.get("ls_project_ids")
    """
    if log_ids and ls_project_ids:
        print(
            "Both 'log_ids' and 'ls_project_id' are present in the config. Defaulting to 'log_ids'."
        )
        ls_project_ids = None
    elif not log_ids and not ls_project_ids and not event_ids and not game_ids:
        raise ValueError(
            "You must provide either 'log_ids', 'game_ids', 'event_ids' or 'ls_project_ids' in your config."
        )

    if game_ids:
        log_ids = []
        for game_id in game_ids:
            log_ids += get_log_ids_per_game(game_id)

    if event_ids:
        log_ids = []
        for event_id in event_ids:
            log_ids += get_log_ids_per_event(event_id)

    """
    model = "/home/stella/robocup/naoth-deeplearning/autolabeling/runs/Nao/TOP/2026-05-25_20-37-34_n/autolabel_model/weights/best.pt"
    num_images = 50

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
                print(
                    f"\n\nLooking for specific run '{run_name}' in '{experiment_name}'..."
                )
                runs = mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'",
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
                    max_results=1,
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
                dst_path=download_dir,
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

    project_ids_to_process = []

    if log_ids:
        from vaapi.client import Vaapi

        v_client = Vaapi(
            base_url=os.environ.get("VAT_API_URL"),
            api_key=os.environ.get("VAT_API_TOKEN"),
        )
        for log_id in log_ids:
            image_obj_list = v_client.image.list(
                log=log_id, camera=camera, validated=True
            )
            for img_obj in image_obj_list:
                proj_id = int(
                    img_obj.labelstudio_url.split("/projects/")[1].split("/")[0]
                )
                if proj_id not in project_ids_to_process:
                    project_ids_to_process.append(proj_id)
    else:
        # FIXME we need to differentiate between data for training and what we use for labeling
        project_ids_to_process = (
            [ls_project_ids] if isinstance(ls_project_ids, int) else ls_project_ids
        )
    
    #FIXME improve logic, auto annotations should only ever be run on either videos or projects
    # FIXME its wrong to query unlabeled tasks first, we first need to filter image, then we can have a blur filter
    # we also need to filter for classes
    # TODO: what should we do if there images where one robot is labeled but others are not. We could not filter that out
    unlabeled_tasks = []
    for proj_id in project_ids_to_process:
        print(f"Fetching unlabeled tasks from project {proj_id}...")
        all_tasks = list(client.tasks.list(project=proj_id))
        # FIXME only works if no annotation of other classes exist already
        unlabeled_tasks.extend([t for t in all_tasks if not t.annotations])

    print(
        f"Found {len(unlabeled_tasks)} unlabeled tasks in total across {len(project_ids_to_process)} projects."
    )

    model = YOLO(model)

    pushed, skipped = 0, 0

    image_limit_exists = False
    if num_images is not None:
        image_limit_exists = True

    for task in unlabeled_tasks:
        if image_limit_exists:
            if pushed >= num_images:
                break

        image_url = task.data.get("image") or task.data.get("img")

        if not image_url:
            skipped += 1
            continue

        if "logs.berlin-united.com" not in image_url:
            image_url = "https://logs.berlin-united.com/" + image_url.lstrip("/")

        # FIXME use download function from tools
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            print(f"Warning: Could not download image for task {task.id}")
            skipped += 1
            continue

        tmp_path = Path(f"/tmp/{task.id}.jpg")
        tmp_path.write_bytes(response.content)

        results = model.predict(source=str(tmp_path))
        boxes = results[0].boxes

        inspect_dir = Path("inspection/")
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
        if hasattr(task, "predictions") and task.predictions:
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

            new_predictions.append(
                {
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
                        "rectanglelabels": [label_name],
                    },
                }
            )

        filtered_predictions = []
        for old_box in existing_predictions_results:
            existing_labels = old_box.get("value", {}).get("rectanglelabels", [])
            if target_class not in existing_labels:
                filtered_predictions.append(old_box)

        final_predictions = filtered_predictions + new_predictions

        if len(final_predictions) > 0:
            mean_score = sum(confidences) / len(confidences) if confidences else 1.0
            predict_on_image(
                client, task_id=task.id, predictions=final_predictions, score=mean_score
            )
            pushed += 1

    print(f"\nDone. {pushed} predictions pushed, {skipped} empty or failed.")
