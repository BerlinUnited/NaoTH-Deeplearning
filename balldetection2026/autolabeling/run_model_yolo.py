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
    parser.add_argument("-m", "--model", type=str, help="Set the model")
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if args.model is None:
        print("Das Modell wurde nicht festgelegt. Der Inspektor schaut in seinen Spind...")
    
        model_dir = f"./data/{args.camera}/autolabel_model"
        
        if not os.path.exists(model_dir):
            print(f"Fehler: Der Ordner {model_dir} existiert noch nicht.")
            sys.exit(1)
        available_models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        if not available_models:
            print(f"Fehler: Keine Modelle im Ordner {model_dir} gefunden.")
            sys.exit(1)
            
        print("\nBitte wähle ein Modell aus:")
        for i, model_name in enumerate(available_models):
            print(f"[{i + 1}] {model_name}")
            
        while True:
            try:
                auswahl = int(input("\nGib die Nummer des gewünschten Modells ein: "))
                if 1 <= auswahl <= len(available_models):
                    args.model = available_models[auswahl - 1]
                    print(f"--> Modell '{args.model}' wurde erfolgreich ausgewählt!\n")
                    break 
                else:
                    print("Ungültige Nummer. Bitte wähle eine Zahl aus der Liste oben.")
            except ValueError:
                print("Das war keine Zahl. Bitte gib eine gültige Ziffer ein.")

    modell_pfad = f"./data/{args.camera}/autolabel_model/{args.model}/weights/best.pt"
    new_model = YOLO(modell_pfad)

    ziel_projekt = os.path.abspath(f"data/{args.camera}")
    ziel_name = "not_human_proofed"

    results = new_model.predict(
        source=f"data/{args.camera}/not_human_proofed/images", 
        # save=True, 
        save_txt=True,
        save_conf=True,
        project=ziel_projekt,
        name=ziel_name,       
        exist_ok=True
    )


    labels_dir = Path(f"data/{args.camera}/not_human_proofed/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_dir = Path(f"data/{args.camera}/not_human_proofed/annotations")
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