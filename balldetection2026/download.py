import os
import requests
import argparse
import json

from pathlib import Path
from vaapi.client import Vaapi
from label_studio_sdk import LabelStudio

def download_image(img_url, img_path):
    response = requests.get(img_url)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Download failed: {img_url}")

def download_annotation(client, img_obj, anno_path):
    task_id = img_obj.labelstudio_url.split("=")[-1]

    try:
        task = client.tasks.get(id=task_id)
        annotations = task.annotations

        if annotations:
            bbox_data = annotations[0].get("result", [])
            print(bbox_data)
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(bbox_data, f, ensure_ascii=False, indent=4)
        else:
            print(f"No annotations for task {task_id}")

    except Exception as e:
        print(f"Annotation error {task_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--camera",
        required=True,
        choices=["TOP", "BOTTOM"],
        help="Camera to download"
    )

    parser.add_argument(
        "-l", "--logs",
        nargs="+",
        type=int,
        required=True,
        help="List of log IDs"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["annotated", "not_annotated", "both"],
        default="both",
        help="images to download"
    )

    args = parser.parse_args()

    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    client = LabelStudio(
        base_url="https://labelstudio-api.berlin-united.com/",
        api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )

    base_dir = Path(f"data/{args.camera}")
    base_dir.mkdir(parents=True, exist_ok=True)

    human_proofed_dir = base_dir / "human_proofed"
    human_proofed_dir.mkdir(parents=True, exist_ok=True)

    human_proofed_images_all_dir = human_proofed_dir / "images/all"
    human_proofed_images_all_dir.mkdir(parents=True, exist_ok=True)
    human_proofed_anno_all_dir = human_proofed_dir / "annotations/all"
    human_proofed_anno_all_dir.mkdir(parents=True, exist_ok=True)

    not_human_proofed_dir = base_dir / "not_human_proofed"
    not_human_proofed_dir.mkdir(parents=True, exist_ok=True)
    not_human_proofed_images = not_human_proofed_dir / "images"
    not_human_proofed_images.mkdir(parents=True, exist_ok=True)


    for log_id in args.logs:

        if args.mode in ["annotated", "both"]:

            image_obj_list = v_client.image.list(
                log=log_id,
                camera=args.camera,
                validated=True

            )

            for img_obj in image_obj_list:

                img_url = "https://logs.berlin-united.com/" + img_obj.image_url
                img_filename = f"{log_id}_{Path(img_obj.image_url).name}"

                img_path = human_proofed_images_all_dir / img_filename
                anno_path = human_proofed_anno_all_dir / f"{Path(img_filename).stem}.json"

                if img_path.exists() and anno_path.exists():
                    print(f"Already downloaded: {img_filename}")
                    continue

                if not img_path.exists():
                    download_image(img_url, img_path)

                if not anno_path.exists():
                    download_annotation(client, img_obj, anno_path)

                print(f"Downloaded human proofed image: {img_filename}")

        if args.mode in ["not_annotated", "both"]:

            image_obj_list = v_client.image.list(
                log=log_id,
                camera=args.camera)
                                        

            for img_obj in image_obj_list:

                img_url = "https://logs.berlin-united.com/" + img_obj.image_url
                img_filename = f"{log_id}_{Path(img_obj.image_url).name}"

                img_path = not_human_proofed_images / img_filename

                if img_path.exists():
                    print(f"Already downloaded: {img_filename}")
                    continue

                download_image(img_url, img_path)

                print(f"Downloaded not human proofed image: {img_filename}")