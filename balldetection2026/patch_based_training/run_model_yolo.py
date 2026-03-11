from pathlib import Path
from ultralytics import YOLO
import argparse
import sys
import os 

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