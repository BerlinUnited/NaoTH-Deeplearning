import cv2
import json
import random
from pathlib import Path

camera = "BOTTOM"

IMAGE_DIR = Path(f"data/{camera}/images")
ANNO_DIR = Path(f"data/{camera}/annotations")

PATCH_DIR = Path(f"data/{camera}/patches")
BALL_DIR = PATCH_DIR / "ball"
NOBALL_DIR = PATCH_DIR / "noball"

BALL_DIR.mkdir(exist_ok=True, parents=True)
NOBALL_DIR.mkdir(exist_ok=True, parents=True)

def check_collision(box1, box2):
    """Prüft, ob sich zwei Rechtecke (x1, y1, x2, y2) überschneiden."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

for img_path in IMAGE_DIR.glob("*.png"): 
    
    anno_path = ANNO_DIR / f"{img_path.stem}.json"

    if not anno_path.exists():
        continue
 
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Konnte Bild nicht öffnen: {img_path.name}")
        continue
        
    h_img, w_img = img.shape[:2]
    
    with open(anno_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
        
    
    ball_boxes = []

    for item in bbox_data:
        val = item.get('value', {})
        if 'rectanglelabels' not in val:
            continue
             
        x = val['x'] * w_img / 100
        y = val['y'] * h_img / 100
        w = val['width'] * w_img / 100
        h = val['height'] * h_img / 100

        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        ball_boxes.append((x1, y1, x2, y2))
        
    
    for idx, (bx1, by1, bx2, by2) in enumerate(ball_boxes):
        
        ball_patch = img[by1:by2, bx1:bx2]
        if ball_patch.size > 0:
            resized_ball = cv2.resize(ball_patch, (16, 16), interpolation=cv2.INTER_AREA)
            ball_save_path = BALL_DIR / f"{img_path.stem}_ball_{idx}.png"
            cv2.imwrite(str(ball_save_path), resized_ball)
            
        box_width = bx2 - bx1
        box_height = by2 - by1
        
        for _ in range(10):
            
            rand_x = random.randint(0, max(0, w_img - box_width))
            rand_y = random.randint(0, max(0, h_img - box_height))
            
            rand_box = (rand_x, rand_y, rand_x + box_width, rand_y + box_height)
            
            hit_a_ball = False
            for real_ball in ball_boxes:
                if check_collision(rand_box, real_ball):
                    hit_a_ball = True
                    break
                    
            if not hit_a_ball:
                rx1, ry1, rx2, ry2 = rand_box
                noball_patch = img[ry1:ry2, rx1:rx2]
                if noball_patch.size > 0:
                    resized_noball = cv2.resize(noball_patch, (16, 16), interpolation=cv2.INTER_AREA)
                    noball_save_path = NOBALL_DIR / f"{img_path.stem}_noball_{idx}.png"
                    cv2.imwrite(str(noball_save_path), resized_noball)
                break 