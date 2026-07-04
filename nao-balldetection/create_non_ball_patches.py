import json
import random
from pathlib import Path
from PIL import Image

PATCH_SIZE = 16
MAX_SIDE = 64
PATCHES_PER_IMAGE = 3
MAX_ATTEMPTS = 10


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def get_ball_boxes(annotations, img_w, img_h):
    boxes = []
    for ann in annotations:
        value = ann["value"]
        if "Ball" not in value.get("rectanglelabels", []):
            continue
        orig_w = value.get("original_width", img_w)
        orig_h = value.get("original_height", img_h)
        x1 = (value["x"] / 100.0) * orig_w
        y1 = (value["y"] / 100.0) * orig_h
        x2 = x1 + (value["width"] / 100.0) * orig_w
        y2 = y1 + (value["height"] / 100.0) * orig_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def sample_negative_patches(image_path: Path, json_path: Path, output_dir: Path):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    img = Image.open(image_path)
    ball_boxes = get_ball_boxes(annotations, img.width, img.height)
    base_name = image_path.stem

    saved = 0
    for i in range(PATCHES_PER_IMAGE):
        for _ in range(MAX_ATTEMPTS):
            side = random.randint(PATCH_SIZE, min(MAX_SIDE, img.width, img.height))
            x1 = random.randint(0, img.width - side)
            y1 = random.randint(0, img.height - side)
            x2, y2 = x1 + side, y1 + side
            candidate = (x1, y1, x2, y2)

            if any(boxes_overlap(candidate, b) for b in ball_boxes):
                continue

            patch = img.crop(candidate).resize(
                (PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS
            )
            out_name = f"{base_name}_neg{i}.jpg"
            patch.save(output_dir / out_name)
            saved += 1
            break
    return saved


def create_non_ball_patches(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        print(f"Skip missing folder: '{input_dir}'")

    output_dir.mkdir(exist_ok=True, parents=True)
    json_files = sorted(input_dir.glob("*.json"))

    for json_path in json_files:
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            print(f"No image for json: {json_path.name}")
            continue

        try:
            sample_negative_patches(image_path, json_path, output_dir)
        except Exception as e:
            print(f"Error in {json_path.name}: {e}")
