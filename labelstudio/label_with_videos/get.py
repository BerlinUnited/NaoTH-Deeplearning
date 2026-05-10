import requests
import cv2
import os
import json
import numpy as np
from config import (
    LS_TARGET_PROJECT_ID,
    PROJECT_IDS,
    DOWNLOAD_DIR,
    get_ls1_client,
    get_video_url,
    get_video_headers,
)
from utils import get_image_from_url

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def is_video_in_ls(filename):
    url = f"{get_video_url()}/api/projects/{LS_TARGET_PROJECT_ID}/tasks?page_size=1000"
    try:
        response = requests.get(url, headers=get_video_headers(), timeout=10)
        return filename in response.text
    except Exception:
        return False


def main():
    ls1 = get_ls1_client()
    video_headers = get_video_headers()
    video_url = get_video_url()

    for pid in PROJECT_IDS:
        print(f"\n--- Project {pid} ---")
        web_video_filename = f"video_project_{pid}_web.mp4"
        web_video_path = os.path.join(DOWNLOAD_DIR, web_video_filename)
        json_mapping_filename = f"mapping_project_{pid}.json"
        json_mapping_path = os.path.join(DOWNLOAD_DIR, json_mapping_filename)

        json_exists = os.path.exists(json_mapping_path)

        existing_task_ids = []
        if json_exists:
            try:
                with open(json_mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_task_ids = data.get(
                        "task_ids", data.get("ls1_task_ids", [])
                    )
            except Exception as e:
                print(f"Couldn't load mapping file: {e}")

        video_already_in_ls = is_video_in_ls(web_video_filename)
        if video_already_in_ls:
            if existing_task_ids:
                print(
                    f"Everything done for project {pid} ({len(existing_task_ids)} frames)"
                )
            else:
                print(
                    "WARNING: Video exists but mapping is broken/missing. Delete video from LabelStudio and re-run."
                )
            continue

        print("Getting tasks with LabelStudioSDK...")
        raw_tasks = list(ls1.tasks.list(project=pid))
        raw_tasks.sort(key=lambda t: t.id if hasattr(t, "id") else t.get("id", 0))

        video_writer = None
        task_ids = []

        print("Video is missing. Loading images and rendering video...")

        for task in raw_tasks:
            task_data = task.data if hasattr(task, "data") else task.get("data", {})
            img_url = task_data.get("image") or task_data.get("img")
            if not img_url:
                continue

            frame = get_image_from_url(img_url)
            if frame is None:
                continue

            t_id = task.id if hasattr(task, "id") else task.get("id")
            task_ids.append(t_id)

            if video_writer is None:
                h, w, _ = frame.shape
                video_writer = cv2.VideoWriter(
                    web_video_path, cv2.VideoWriter_fourcc(*"avc1"), 25.0, (w, h)
                )

            video_writer.write(frame)

        if video_writer:
            video_writer.release()

        mapping_data = {
            "source_project_id": pid,
            "target_video_name": web_video_filename,
            "task_ids": task_ids,
            "frame_count": len(task_ids),
        }
        with open(json_mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=4)
        print(f"JSON was saved: {len(task_ids)} frames mapped.")

        if not video_already_in_ls and os.path.exists(web_video_path):
            print("Uploading video...")
            try:
                with open(web_video_path, "rb") as f:
                    res = requests.post(
                        f"{video_url}/api/projects/{LS_TARGET_PROJECT_ID}/import",
                        headers=video_headers,
                        files={"file": (web_video_filename, f, "video/mp4")},
                        timeout=300,
                    )
                if res.status_code in [200, 201]:
                    os.remove(web_video_path)
                    print("Deleted locally")
                else:
                    print(f"Upload error {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
