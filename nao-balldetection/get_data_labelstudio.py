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


def download_annotated_ball_images_labelstudio(base_dir, folder_prefix, event_id):
    """
    Download all images that have been annotated with a ball in Label Studio and save them in the configured folder.
    Save the corresponding annotations in JSON format alongside the images.
    """
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    logs = v_client.logs.list(event=event_id)

    cameras = ["TOP", "BOTTOM"]

    for camera in cameras:
        output_dir = base_dir / f"{folder_prefix}_{camera}"
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

                    img_path = output_dir / f"{filename_base}.jpg"
                    ann_path = output_dir / f"{filename_base}.json"

                    success = download_image(img_url, img_path)

                    if not success:
                        continue

                    with open(ann_path, "w") as f:
                        json.dump(annotations, f, indent=2)
