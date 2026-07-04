import json
from pathlib import Path
from PIL import Image

import requests
import os
import json
import os
from pathlib import Path
import requests
from vaapi.client import Vaapi


def normalize_label(label: str) -> str:
    return label.strip().lower()


def extract_labels(annotations: list[dict], annotation_type: str) -> set[str]:
    """
    Extract normalized labels of a specific annotation type.
    """
    found = set()

    for ann in annotations:
        if ann.get("type") != annotation_type:
            continue

        labels = ann.get("value", {}).get(annotation_type, [])

        if not isinstance(labels, list):
            continue

        found.update(l.strip().lower() for l in labels)

    return found


def download_image(url: str, output_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"Failed download: {url}")
            return False

        with open(output_path, "wb") as f:
            f.write(response.content)

        return True

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False


def extract_project_id(labelstudio_url: str) -> str:
    """
    Example: https://.../projects/7881/data... -> "7881"
    """
    if not labelstudio_url or "/projects/" not in labelstudio_url:
        return "unknown"

    return labelstudio_url.split("/projects/")[1].split("/")[0]


def download_annotated_ball_images_labelstudio(output_dir, event_id, camera):
    """
    Download all images that have been annotated with a ball in Label Studio and save them in the configured folder.
    Save the corresponding annotations in JSON format alongside the images.
    """
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    logs = v_client.logs.list(event=event_id)

    output_dir.mkdir(parents=True, exist_ok=True)

    for log in logs:
        numeric_log_id = str(log).split(" ")[0]
        print(f"\nProcessing log {numeric_log_id} for camera {camera}...")

        image_obj_list = v_client.image.list(
            log=numeric_log_id,
            camera=camera,
            has_annotations=True,
        )

        for img_obj in image_obj_list:
            frame_id = img_obj.frame.id
            frame_number = img_obj.frame.frame_number
            annotations = getattr(img_obj, "annotation", None) or []

            ls_url = getattr(img_obj, "labelstudio_url", "")

            rectangle_labels = extract_labels(
                annotations,
                annotation_type="rectanglelabels",
            )

            if "ball" in rectangle_labels:
                img_url = "https://logs.berlin-united.com/" + img_obj.image_url

                project_id = extract_project_id(ls_url)

                filename_base = (
                    f"img-{numeric_log_id}-{project_id}-{frame_id}-{frame_number}"
                )

                img_path = output_dir / Path(f"{filename_base}.jpg")
                ann_path = output_dir / Path(f"{filename_base}.json")

                success = download_image(img_url, img_path)

                if not success:
                    continue

                with open(ann_path, "w") as f:
                    json.dump(annotations, f, indent=2)



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
    PATCH_SIZE = 16
    MAX_SIDE = 64
    PATCHES_PER_IMAGE = 3
    MAX_ATTEMPTS = 10
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
