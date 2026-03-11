from ultralytics import YOLO
import argparse
import sys
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()
 
    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    new_model = YOLO("./runs/detect/train2/weights/best.pt")
    
    aktueller_ordner = os.path.dirname(os.path.abspath(__file__))
    
    ziel_ordner = os.path.join(aktueller_ordner, "results")

    results = new_model.predict(
        source=f"data/{args.camera}/images/all", 
        save=True, 
        save_txt=True,
        project=ziel_ordner,        
        name=f"{args.camera}" 
    )