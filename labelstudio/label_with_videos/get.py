# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "label_studio_sdk",
# ]
# ///

import requests
import cv2
import os
import json
import numpy as np
from label_studio_sdk import LabelStudio

LS1_URL = "https://labelstudio-api.berlin-united.com"
LS1_TOKEN = os.getenv("LABELSTUDIO_API_KEY", "DEIN_LS1_TOKEN")
PROJECT_IDS = [
    7694,
    7695,
    7696,
    7697,
    7698,
    7699,
    7700,
    7701,
    7702,
    7703,
    7704,
    7705,
    7706,
    7707,
    7708,
    7709,
    7710,
    7711,
    7712,
    7713,
    7714,
    7715,
    7716,
    7717,
    7686,
    7687,
    7688,
    7689,
    7690,
    7691,
    7692,
    7693,
    7676,
    7677,
    7678,
    7679,
    7680,
    7681,
    7682,
    7683,
    7684,
    7685,
]
LS2_URL = "http://localhost:8080"
LS2_TOKEN = "34db6a8a93ad9446a946adc234577a3f2587dc96"
LS2_TARGET_PROJECT_ID = 4

ls1 = LabelStudio(base_url=LS1_URL, api_key=LS1_TOKEN)
HEADERS_LS1 = {"Authorization": f"Token {LS1_TOKEN}"}
HEADERS_LS2 = {"Authorization": f"Token {LS2_TOKEN}"}

DOWNLOAD_DIR = "temp_videos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_image_from_ram(url):
    if url.startswith("/"):
        url = LS1_URL + url
    try:
        response = requests.get(url, headers=HEADERS_LS1, timeout=15)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except:
        pass
    return None


def is_video_in_ls2(filename):
    url = f"{LS2_URL}/api/projects/{LS2_TARGET_PROJECT_ID}/tasks?page_size=1000"
    try:
        response = requests.get(url, headers=HEADERS_LS2, timeout=10)
        return filename in response.text
    except:
        return False


def main():
    for pid in PROJECT_IDS:
        print(f"\n--- Project {pid} ---")
        web_video_filename = f"video_project_{pid}_web.mp4"
        web_video_path = os.path.join(DOWNLOAD_DIR, web_video_filename)
        json_mapping_filename = f"mapping_project_{pid}.json"
        json_mapping_path = os.path.join(DOWNLOAD_DIR, json_mapping_filename)

        video_already_in_ls2 = is_video_in_ls2(web_video_filename)
        json_exists = os.path.exists(json_mapping_path)

        if video_already_in_ls2 and json_exists:
            print(f"Everything done for project {pid}")
            continue

        mapping_data = {}
        if json_exists:
            print("Loading existing mapping file to preserve old annotations...")
            try:
                with open(json_mapping_path, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
            except Exception as e:
                print(
                    f"Couldn't load mapping file, creating new one... The error was: {e}"
                )

        mapping_data["source_project_id"] = pid
        mapping_data["target_video_name"] = web_video_filename
        mapping_data["ls1_task_ids"] = []

        print("Getting tasks with LabelStudioSDK...")
        raw_tasks = list(ls1.tasks.list(project=pid))
        raw_tasks.sort(key=lambda t: t.id if hasattr(t, "id") else t.get("id", 0))

        video_writer = None

        if video_already_in_ls2:
            print("Video is already in LS2.")
            for task in raw_tasks:
                task_data = task.data if hasattr(task, "data") else task.get("data", {})
                if not (task_data.get("image") or task_data.get("img")):
                    continue

                t_id = task.id if hasattr(task, "id") else task.get("id")
                mapping_data["ls1_task_ids"].append(t_id)
        else:
            print("Video is missing. Loading images and render video...")
            for task in raw_tasks:
                task_data = task.data if hasattr(task, "data") else task.get("data", {})
                img_url = task_data.get("image") or task_data.get("img")
                if not img_url:
                    continue

                frame = get_image_from_ram(img_url)
                if frame is None:
                    continue

                t_id = task.id if hasattr(task, "id") else task.get("id")
                mapping_data["ls1_task_ids"].append(t_id)

                if video_writer is None:
                    h, w, _ = frame.shape
                    video_writer = cv2.VideoWriter(
                        web_video_path, cv2.VideoWriter_fourcc(*"avc1"), 25.0, (w, h)
                    )

                video_writer.write(frame)

            if video_writer:
                video_writer.release()

        with open(json_mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f)
        print(f"JSON was saved: {len(mapping_data['ls1_task_ids'])} frames mapped.")

        if not video_already_in_ls2 and os.path.exists(web_video_path):
            print(f"Uploading video...")
            with open(web_video_path, "rb") as f:
                res = requests.post(
                    f"{LS2_URL}/api/projects/{LS2_TARGET_PROJECT_ID}/import",
                    headers=HEADERS_LS2,
                    files={"file": (web_video_filename, f, "video/mp4")},
                )
            if res.status_code in [200, 201]:
                os.remove(web_video_path)
                print("Deleted locally")


if __name__ == "__main__":
    main()
