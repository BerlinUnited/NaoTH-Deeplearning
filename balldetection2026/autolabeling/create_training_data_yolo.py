from pathlib import Path
import random
import json
import argparse
import shutil

def convert_to_yolo(bbox_data, class_map):
    """Konvertiert Label Studio Daten in das YOLO-Format."""
    yolo_lines = []
    for item in bbox_data:
        val = item['value']
        label = val.get("rectanglelabels", [])
        if "Ball" not in label:
            continue
        
        label_name = label[0]
        class_id = class_map.get(label_name, 0) 
        
        x_tl, y_tl = val['x'], val['y']
        w_ls, h_ls = val['width'], val['height']
        
        x_center = (x_tl + (w_ls / 2)) / 100
        y_center = (y_tl + (h_ls / 2)) / 100
        w_yolo, h_yolo = w_ls / 100, h_ls / 100
        
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(line)
    return yolo_lines

def process_and_split(image_list, target_img_dir, target_lbl_dir, label_all_dir, anno_dir, mapping):
    processed_count = 0
    for img_path in image_list:
        json_file = anno_dir / (img_path.stem + ".json")
        if not json_file.exists():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                bbox_data = json.load(f)
                print(bbox_data)
                if not bbox_data: continue
                yolo_results = convert_to_yolo(bbox_data, mapping)
                print(yolo_results)
            except (json.JSONDecodeError, KeyError):
                continue

        shutil.copy2(img_path, target_img_dir / img_path.name)
        
        content = "\n".join(yolo_results)
        txt_name = img_path.stem + ".txt"
        
        with open(target_lbl_dir / txt_name, 'w', encoding='utf-8') as f:
            f.write(content)
            
        with open(label_all_dir / txt_name, 'w', encoding='utf-8') as f:
            f.write(content)
            
        processed_count += 1
    return processed_count

def clean_dir(directory):
    """Löscht den Inhalt eines Ordners, damit beim Reshuffle nichts Altes bleibt."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, required=True)
    args = parser.parse_args()
    camera = args.camera.upper()

    base_data = Path(f"data/{args.camera}")
    image_all_dir = base_data / "images/all"
    anno_all_dir = base_data / "annotations/all"
    
    train_img, val_img = base_data / "images/train", base_data / "images/val"
    train_lbl, val_lbl = base_data / "labels/train", base_data / "labels/val"
    label_all_dir = base_data / "labels/all"

    if not anno_all_dir.exists() or not image_all_dir.exists():
        print("Fehler: Quelldaten fehlen!")
        exit()

    for d in [train_img, val_img, train_lbl, val_lbl, label_all_dir]:
        clean_dir(d)

    mapping = {"Ball": 0}
    images = list(image_all_dir.glob("*.png"))
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_files, val_files = images[:split_idx], images[split_idx:]
    print(train_files)

    c_train = process_and_split(train_files, train_img, train_lbl, label_all_dir, anno_all_dir, mapping)
    c_val = process_and_split(val_files, val_img, val_lbl, label_all_dir, anno_all_dir, mapping)

    print(f"Fertig! Alles neu gemischt. Train: {c_train}, Val: {c_val}. Alle Labels unter {label_all_dir}")