import os
import json
import argparse
from pathlib import Path

def convert_to_yolo(bbox_data, class_map):
    """
    Converts Label Studio rectanglelabels to YOLO format strings.
    """
    yolo_lines = []
    
    for item in bbox_data:
        val = item['value']
        
        label_name = val['rectanglelabels'][0]
        class_id = class_map.get(label_name, 0) 
        
        x_tl = val['x']
        y_tl = val['y']
        w_ls = val['width']
        h_ls = val['height']
        
        x_center = (x_tl + (w_ls / 2)) / 100
        y_center = (y_tl + (h_ls / 2)) / 100
        w_yolo = w_ls / 100
        h_yolo = h_ls / 100
        
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(line)
        
    return yolo_lines


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Konvertiert Label Studio JSON Annotationen in YOLO TXT Dateien.")
    parser.add_argument("-c", "--camera", type=str, required=True, help="Setze BOTTOM oder TOP")
    args = parser.parse_args()

    camera = args.camera.upper()

    anno_dir = Path(f"data/{camera}/annotations")
    labels_dir = Path(f"data/{camera}/labels")    
    labels_dir.mkdir(exist_ok=True, parents=True)

    if not anno_dir.exists():
        print(f"Fehler: Das Verzeichnis {anno_dir} existiert nicht. Bitte lade zuerst die JSON-Daten herunter.")
        exit()
    
    mapping = {"Ball": 0}
    
    converted_count = 0

    for json_file in anno_dir.glob("*.json"):
        
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                bbox_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Fehler beim Lesen von {json_file.name}. Überspringe.")
                continue
            
        if not bbox_data:
            continue
            
        try:
            yolo_results = convert_to_yolo(bbox_data, mapping)
        except KeyError as e:
            print(f"Warnung: Unerwartetes Format in {json_file.name}. Fehlt ein Schlüssel? Fehler: {e}")
            continue
            
        txt_filename = f"{json_file.stem}.txt"
        txt_path = labels_dir / txt_filename
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_results))
            
        converted_count += 1

    print(f"Fertig! Es wurden {converted_count} Dateien erfolgreich in das YOLO-Format übersetzt.")