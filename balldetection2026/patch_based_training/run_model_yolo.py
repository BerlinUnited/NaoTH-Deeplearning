from pathlib import Path
from ultralytics import YOLO
import argparse
import sys
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    parser.add_argument("-t", "--training", type=str, help="Set the train name")
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if args.training is None:
        print("The training is not set.\nSet with option -t, --training e.g. train6")
        sys.exit()

    new_model = YOLO(f"../../runs/detect/yolo_runs/{args.training}/weights/best.pt")
    
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