import os
import random
import shutil
import argparse
from pathlib import Path

def reshuffle_and_copy(camera, val_split=0.2, seed=42):
    random.seed(seed)
    
    base_dir = Path(f"data/{camera}/patches")
    classes = ["noball", "ball"]
    
    print(f"Starte den Reshuffle-Roboter in: {base_dir} (Seed: {seed})")
    
    for cls in classes:
        source_cls_dir = base_dir / cls
        source_cls_dir.mkdir(parents=True, exist_ok=True)
        
        if not any(source_cls_dir.iterdir()):
            print(f"Master-Ordner für '{cls}' ist leer. Rette Bilder aus train/val zurück...")
            for split_name in ["train", "val"]:
                old_split_dir = base_dir / split_name / cls
                if old_split_dir.exists():
                    for img in old_split_dir.glob("*.png"):
                        shutil.move(str(img), str(source_cls_dir / img.name))
                        
    for split_name in ["train", "val"]:
        split_dir = base_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
            
    for cls in classes:
        source_cls_dir = base_dir / cls
        
        images = [f for f in source_cls_dir.iterdir() if f.is_file() and f.suffix == '.png']
        if not images:
            print(f"WARNUNG: Keine Bilder in {source_cls_dir} gefunden!")
            continue
            
        random.shuffle(images)
        
        split_index = int(len(images) * (1 - val_split))
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        train_cls_dir = base_dir / "train" / cls
        val_cls_dir = base_dir / "val" / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Kopiere Klasse '{cls}': {len(train_images)} -> train, {len(val_images)} -> val.")
        for img in train_images:
            shutil.copy(str(img), str(train_cls_dir / img.name))
        for img in val_images:
            shutil.copy(str(img), str(val_cls_dir / img.name))

    print("\nReshuffle und Kopieren erfolgreich abgeschlossen!")
    print("Deine Master-Bilder liegen sicher in 'ball' und 'noball'.")
    print("Der Prüf- und Lernstoff liegt in 'train' und 'val'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, required=True, help="Set BOTTOM or TOP")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed für den Shuffle (Standard: 42)")
    args = parser.parse_args()
    
    reshuffle_and_copy(args.camera, seed=args.seed)