import json
import sys
from pathlib import Path
import argparse
import cv2
import re


def check_collision(box1, box2):
    """Check, if two rectangles overlap (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

def overlap_coverage(box1, candidate):
    min_x1, min_y1, max_x1, max_y1 = box1
    candidate_min_x, candidate_min_y, candidate_max_x, candidate_max_y = candidate
    x_left = max(min_x1, candidate_min_x)
    y_top = max(min_y1, candidate_min_y)
    x_right = min(max_x1, candidate_max_x)
    y_bottom = min(max_y1, candidate_max_y)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area2 = (candidate_max_x - candidate_min_x) * (candidate_max_y - candidate_min_y)
    return intersection / min(area1, area2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    parser.add_argument("-p", "--predictor", type=str, help="Set human (h) or Yolo Predicted (y)")
    parser.add_argument("-o", "--overlap", type=int, default=0.1, help="Set overlap")
    parser.add_argument("-l", "--log_ids", required=True, type=lambda s: s.split(","), help="log id")
    args = parser.parse_args()

    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    if args.predictor is None:
        print("The predictor is not set.\nSet with option -p, --predictor h / y (Human/Yolo Predicted)")
        sys.exit()

    if args.log_ids is None:
        print("The log ids are not set.")
        sys.exit()

    if args.predictor == "h":
        image_save_dir = Path(f"data/{args.camera}/human_proofed/images/all")
        anno_save_dir = Path(f"data/{args.camera}/human_proofed/annotations/all")
    elif args.predictor == "y":
        image_save_dir = Path(f"data/{args.camera}/not_human_proofed/images")
        anno_save_dir = Path(f"data/{args.camera}/not_human_proofed/annotations")

    patch_dir = Path(f"data/{args.camera}/patches")
    patch_dir.mkdir(exist_ok=True, parents=True)

    ball_dir = patch_dir / "ball"
    noball_dir = patch_dir / "noball"
    ball_dir.mkdir(exist_ok=True, parents=True)
    noball_dir.mkdir(exist_ok=True, parents=True)

    for log_id in args.log_ids:
        jsonl_path = f"data/logs/{log_id}_ball_candidates.jsonl"

        pattern = re.compile(rf"^{log_id}_.*\.png$")

        try:
            with open(jsonl_path, "r") as f:
                for zeile in f:
                    frame_daten = json.loads(zeile)
                    frame_name = frame_daten["frame_id"]

                    anno_path = anno_save_dir / f"{frame_name}.json"
                    img_path = image_save_dir / f"{frame_name}.png"

                    if not anno_path.exists() or not img_path.exists():
                        continue

                    img_color = cv2.imread(str(img_path))
                    if img_color is None:
                        print(f"Konnte Bild nicht öffnen: {img_path.name}")
                        continue
                    
                    # greyscale
                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                    h_img, w_img = img_gray.shape[:2]

                    debug_img = img_color.copy() 

                    # mark balls with green rectangles, candidates with red rectangles
                    with open(anno_path, "r", encoding="utf-8") as anno_file:
                        bbox_data = json.load(anno_file)

                    ball_boxes = []
                    for item in bbox_data:
                        val = item.get("value", {})
                        if "rectanglelabels" not in val: continue
                        if args.predictor == "y" and float(val["confidence"]) < 0.5: continue
                        if "Ball" not in val["rectanglelabels"]: continue

                        # transfer coordinates
                        x = val["x"] * w_img / 100
                        y = val["y"] * h_img / 100
                        w = val["width"] * w_img / 100
                        h = val["height"] * h_img / 100

                        final_x_min = max(0, int(round(x))) 
                        final_y_min = max(0, int(round(y)))
                        final_x_max = min(w_img, int(round(x + w))) 
                        final_y_max = min(h_img, int(round(y + h)))

                        box = (final_x_min, final_y_min, final_x_max, final_y_max)
                        ball_boxes.append(box)

                        cv2.rectangle(debug_img, (final_x_min, final_y_min), (final_x_max, final_y_max), (0, 255, 0), 2)


                    patches_name = f"{str(args.camera).lower()}_patches"
                    
                    for patch in frame_daten.get(patches_name, []):
                        candidate = (patch["min_x"], patch["min_y"], patch["max_x"], patch["max_y"])
                        c_min_x, c_min_y, c_max_x, c_max_y = candidate

                        cv2.rectangle(debug_img, (c_min_x, c_min_y), (c_max_x, c_max_y), (0, 0, 255), 2)

                        # proof collision and overlap with ground truth 
                        for box_id, box in enumerate(ball_boxes):
                            overlap_wert = overlap_coverage(box, candidate)
                            
                            if overlap_wert > 0:  
                                cv2.putText(debug_img, f"Ov: {overlap_wert:.2f}", (c_min_x, max(0, c_min_y - 5)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                            if overlap_wert >= 0.8: 
                                ball_patch = img_gray[c_min_y:c_max_y, c_min_x:c_max_x]
                                
                                if ball_patch.size > 0:
                                    resized_ball = cv2.resize(ball_patch, (16, 16), interpolation=cv2.INTER_AREA)
                                    ball_save_path = ball_dir / f"{frame_name}_ball_{box_id}.png"
                                    cv2.imwrite(str(ball_save_path), resized_ball)

                    debug_save_path = patch_dir / f"DEBUG_{frame_name}.png"
                    cv2.imwrite(str(debug_save_path), debug_img)

        except FileNotFoundError:
            print(f"Achtung: Konnte die JSONL-Datei für {log_id} nicht finden.")

