from pathlib import Path
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
import requests
from tools import predict_on_image, invert_class_map, CLASS_MAP
import argparse
import sys
import os 
import json
import shutil 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, required=True, help="Set BOTTOM or TOP")
    parser.add_argument("-m", "--model", type=str, help="Path to model weights (.pt file)")
    parser.add_argument("-p", "--project", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("-n", "--num_images", type=int, help="Maximum number of images to predict (default: all)")
    
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if args.model is None:
        print("The model is not set. Pick a model interactive from the list of available models in the autolabel_model folder.")
        model_dir = f"./data/{args.camera}/autolabel_model"
        
        if not os.path.exists(model_dir):
            print(f"Error: No folder named {model_dir} existing.")
            sys.exit(1)
        available_models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        if not available_models:
            print(f"Error: No modelle in folder {model_dir}.")
            sys.exit(1)
            
        print("\nChoose a model:")
        for i, model_name in enumerate(available_models):
            print(f"[{i + 1}] {model_name}")
        
        model = ""
        while True:
            try:
                selection = int(input("\nInput number of model: "))
                if 1 <= selection <= len(available_models):
                    model = available_models[selection - 1]
                    print(f"--> model '{model}' choosen!\n")
                    break 
                else:
                    print("No valid number. Please choose a number from the list.")
            except ValueError:
                print("Type-Erro, only number input.")

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
        # break after maximum number of images are annotated
        if image_limit_exists:
            if pushed >= args.num_images:
                break

        image_url = task.data.get("image")
        
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            print(f"Warning: Could not download image for task {task.id}")
            skipped += 1
            continue

        # Write to temp file for YOLO
        tmp_path = Path(f"/tmp/{task.id}.jpg")
        tmp_path.write_bytes(response.content)

        results =  model.predict(source=str(tmp_path))
        boxes = results[0].boxes

        inspect_dir = Path(f"inspection/")
        inspect_dir.mkdir(parents=True, exist_ok=True)
        # save with YOLO's own visualization (draws boxes on image)
        results[0].save(filename=str(inspect_dir / f"{task.id}.jpg"))

        tmp_path.unlink()  # cleanup

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