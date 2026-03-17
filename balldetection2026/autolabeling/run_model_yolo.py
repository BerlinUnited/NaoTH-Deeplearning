from pathlib import Path
from ultralytics import YOLO
import argparse
import sys
import os 
import json

def yolo_to_labelstudio(yolo_lines, class_map):
    """
    Converts YOLO bbox lines into Label Studio JSON format.
    """
    predictions = []

    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue

        class_id, x_center, y_center, w_yolo, h_yolo, confidence = parts
        class_id = int(class_id)
        label_name = class_map.get(class_id, "Ball")  # default label if not found

        # Convert normalized YOLO coords back to percentages for Label Studio
        x_center = float(x_center) * 100
        y_center = float(y_center) * 100
        w_percent = float(w_yolo) * 100
        h_percent = float(h_yolo) * 100

        # Convert center → top-left
        x_tl = x_center - w_percent / 2
        y_tl = y_center - h_percent / 2

        predictions.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x_tl,
                "y": y_tl,
                "width": w_percent,
                "height": h_percent,
                "confidence": confidence,
                "rotation": 0,
                "rectanglelabels": [label_name]
            }
        })

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    parser.add_argument("-m", "--modelpath", type=str, default="../../runs/detect/yolo_runs/train/weights/best.pt", help="Set the model path")
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    new_model = YOLO(args.modelpath)
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    target_folder = Path(f"{current_folder}/data/yolo")
    target_folder.mkdir(exist_ok=True, parents=True)

    results = new_model.predict(
        source=f"data/{args.camera}/images_not_annotated/", 
        save=True, 
        save_txt=True,
        save_conf=True,
        project=target_folder,        
        name=f"{args.camera}" 
    )


    labels_dir = Path(f"{target_folder}/{args.camera}/labels")
    json_dir = Path(f"{target_folder}/{args.camera}/annotations")
    json_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        print(f"Fehler: Labels-Verzeichnis {labels_dir} existiert nicht.")
        exit()

    mapping = {0: "Ball"}

    converted_count = 0

    for txt_file in labels_dir.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            yolo_lines = f.readlines()

        if not yolo_lines:
            continue

        json_data = yolo_to_labelstudio(yolo_lines, mapping)

        out_path = json_dir / f"{txt_file.stem}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        converted_count += 1

    print(f"Fertig! {converted_count} YOLO-Dateien in JSON umgewandelt und gespeichert unter {json_dir}")