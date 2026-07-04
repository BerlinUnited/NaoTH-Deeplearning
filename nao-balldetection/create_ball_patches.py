import json
from pathlib import Path
from PIL import Image


def crop_balls_from_image(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    min_ball_px_width=10,
    min_ball_px_height=10,
):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    img = Image.open(image_path)
    base_name = image_path.stem

    for i, ann in enumerate(annotations):
        value = ann["value"]

        # Skip anything that isn't a "Ball" label, just in case
        labels = value.get("rectanglelabels", [])
        if "Ball" not in labels:
            continue

        # Values are given in percent of the original image dimensions
        orig_w = value.get("original_width", img.width)
        orig_h = value.get("original_height", img.height)

        x_pct = value["x"]
        y_pct = value["y"]
        w_pct = value["width"]
        h_pct = value["height"]

        # Convert percentages to pixel coordinates
        x1 = (x_pct / 100.0) * orig_w
        y1 = (y_pct / 100.0) * orig_h
        box_w = (w_pct / 100.0) * orig_w
        box_h = (h_pct / 100.0) * orig_h

        x2 = x1 + box_w
        y2 = y1 + box_h

        # Clamp to image bounds and round to ints
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img.width, int(round(x2)))
        y2 = min(img.height, int(round(y2)))

        if x2 <= x1 or y2 <= y1:
            print(f"  Skipping invalid box in {json_path.name} (index {i})")
            continue

            # Adjust patch size to be square, centered on the original box
        center_x = x1 + (box_w / 2)
        center_y = y1 + (box_h / 2)
        max_side = max(box_w, box_h)

        # If the ball itself is bigger than the image in some dimension, clamp side length
        max_side = min(max_side, orig_w, orig_h)

        sq_left = center_x - (max_side / 2)
        sq_top = center_y - (max_side / 2)
        sq_right = center_x + (max_side / 2)
        sq_bottom = center_y + (max_side / 2)

        # Shift the square to stay within image bounds instead of dropping it
        if sq_left < 0:
            shift = -sq_left
            sq_left += shift
            sq_right += shift
        elif sq_right > orig_w:
            shift = sq_right - orig_w
            sq_left -= shift
            sq_right -= shift

        if sq_top < 0:
            shift = -sq_top
            sq_top += shift
            sq_bottom += shift
        elif sq_bottom > orig_h:
            shift = sq_bottom - orig_h
            sq_top -= shift
            sq_bottom -= shift

        sq_left_i = max(0, int(round(sq_left)))
        sq_top_i = max(0, int(round(sq_top)))
        sq_right_i = min(img.width, int(round(sq_right)))
        sq_bottom_i = min(img.height, int(round(sq_bottom)))

        if sq_right_i <= sq_left_i or sq_bottom_i <= sq_top_i:
            continue

        if box_w < min_ball_px_width or box_h < min_ball_px_height:
            print(
                f"  Skipping too-small ball in {json_path.name} (index {i}): "
                f"{box_w:.1f}x{box_h:.1f}"
            )
            continue

        patch = img.crop((sq_left_i, sq_top_i, sq_right_i, sq_bottom_i))
        patch = patch.resize((32, 32), Image.Resampling.LANCZOS)

        if len(annotations) > 1:
            out_name = f"{base_name}_{i}.jpg"
        else:
            out_name = f"{base_name}.jpg"

        patch.save(output_dir / out_name)


def create_ball_patches(input_dir: Path, output_dir: Path):
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
            crop_balls_from_image(image_path, json_path, output_dir)
        except Exception as e:
            print(f"Error in {json_path.name}: {e}")
