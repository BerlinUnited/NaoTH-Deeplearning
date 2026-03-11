import os
import json
import argparse
from pathlib import Path
import os
import json
import argparse
from pathlib import Path
from PIL import Image  # pip install pillow

def yolo_to_labelstudio(yolo_lines, image_width, image_height, class_map):
    """
    Converts YOLO bbox lines into Label Studio JSON format.
    """
    predictions = []

    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, w_yolo, h_yolo = parts
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
                "rotation": 0,
                "rectanglelabels": [label_name]
            }
        })

    return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Konvertiert YOLO TXT Labels in Label Studio JSON")
    parser.add_argument("-c", "--camera", type=str, required=True, help="BOTTOM oder TOP")
    parser.add_argument("--images_dir", type=str, default=None, help="Pfad zu den Bildern (optional, automatisch)")
    args = parser.parse_args()

    camera = args.camera.upper()

    labels_dir = Path(f"data/{camera}/labels")
    json_dir = Path(f"data/{camera}/annotations_from_yolo")
    json_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        print(f"Fehler: Labels-Verzeichnis {labels_dir} existiert nicht.")
        exit()

    # Map class IDs to names
    mapping = {0: "Ball"}

    converted_count = 0

    for txt_file in labels_dir.glob("*.txt"):
        # get corresponding image file
        if args.images_dir:
            images_dir = Path(args.images_dir)
        else:
            images_dir = labels_dir.parent / "images"

        img_candidates = list(images_dir.glob(f"{txt_file.stem}.*"))
        if not img_candidates:
            print(f"Warnung: Keine Bilddatei gefunden für {txt_file.name}. Überspringe.")
            continue
        img_path = img_candidates[0]

        # read image size
        with Image.open(img_path) as im:
            img_width, img_height = im.size

        # read YOLO labels
        with open(txt_file, 'r', encoding='utf-8') as f:
            yolo_lines = f.readlines()

        if not yolo_lines:
            continue

        # convert YOLO → Label Studio JSON
        json_data = yolo_to_labelstudio(yolo_lines, img_width, img_height, mapping)

        # save as JSON
        out_path = json_dir / f"{txt_file.stem}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        converted_count += 1

    print(f"Fertig! {converted_count} YOLO-Dateien in JSON umgewandelt und gespeichert unter {json_dir}")