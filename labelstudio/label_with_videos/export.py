import os
import re
import cv2
from label_studio_sdk import LabelStudio
import json
import requests
import numpy as np

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

DOWNLOAD_DIR = "temp_videos"

ls1 = LabelStudio(base_url=LS1_URL, api_key=LS1_TOKEN)
ls2 = LabelStudio(base_url=LS2_URL, api_key=LS2_TOKEN)
HEADERS_LS1 = {"Authorization": f"Token {LS1_TOKEN}"}


def get_value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_image_dimensions_from_ram(url):
    if url.startswith("/"):
        url = LS1_URL + url
    try:
        res = requests.get(url, headers=HEADERS_LS1, timeout=10)
        if res.status_code == 200:
            img = cv2.imdecode(
                np.asarray(bytearray(res.content), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is not None:
                return img.shape[1], img.shape[0]
    except:
        pass
    return None, None


def main():
    ls2_tasks = ls2.tasks.list(project=LS2_TARGET_PROJECT_ID)

    for pid in PROJECT_IDS:
        print(f"\n--- READING PROJECT {pid} ---")
        mapping_path = os.path.join(DOWNLOAD_DIR, f"mapping_project_{pid}.json")
        if not os.path.exists(mapping_path):
            continue

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        target_video_name = mapping_data.get(
            "target_video_name", f"video_project_{pid}_web.mp4"
        )
        ls1_task_ids = mapping_data.get("ls1_task_ids", [])
        total_images = len(ls1_task_ids)
        if total_images == 0:
            continue

        ls2_task_id = None
        for task in ls2_tasks:
            file_url = (
                get_value(get_value(task, "data", {}), "video")
                or get_value(get_value(task, "data", {}), "file_upload")
                or ""
            )
            if target_video_name in str(file_url):
                ls2_task_id = get_value(task, "id")
                break

        if not ls2_task_id:
            print(f"Video was not found in LS2. Skipping it...")
            continue

        full_task = ls2.tasks.get(id=ls2_task_id)
        annotations = get_value(full_task, "annotations", [])
        if not annotations:
            print(f"No boxes are drawn yet")
            continue

        first_ls1_task_id = ls1_task_ids[0]
        first_task_info = ls1.tasks.get(id=first_ls1_task_id)
        img_url = get_value(
            (
                first_task_info.data
                if hasattr(first_task_info, "data")
                else first_task_info.get("data", {})
            ),
            "image",
        )
        orig_width, orig_height = get_image_dimensions_from_ram(img_url)

        raw_ls2_results = get_value(annotations[0], "result", [])
        frame_annotations = {i: [] for i in range(total_images)}

        for result in raw_ls2_results:
            if get_value(result, "type") == "videorectangle":
                val = get_value(result, "value", {})
                sequence = sorted(
                    get_value(val, "sequence", []),
                    key=lambda k: get_value(k, "frame", 0),
                )
                label_name = (
                    get_value(val, "labels", [""])[0]
                    if get_value(val, "labels", [""])
                    else ""
                )

                for i in range(len(sequence)):
                    kf_start = sequence[i]
                    f_start = max(0, int(get_value(kf_start, "frame")) - 1)
                    if not get_value(kf_start, "enabled"):
                        continue

                    if i + 1 < len(sequence):
                        kf_end = sequence[i + 1]
                        f_end = max(0, int(get_value(kf_end, "frame")) - 1)
                    else:
                        kf_end = kf_start
                        f_end = total_images

                    for f in range(f_start, f_end):
                        if f >= total_images:
                            break
                        ratio = (
                            0
                            if (f_end == f_start or f_end == f_start + 1)
                            else (f - f_start) / (f_end - f_start)
                        )
                        startX, startY = float(get_value(kf_start, "x", 0)), float(
                            get_value(kf_start, "y", 0)
                        )
                        startW, startH = float(get_value(kf_start, "width", 0)), float(
                            get_value(kf_start, "height", 0)
                        )
                        endX, endY = float(get_value(kf_end, "x", startX)), float(
                            get_value(kf_end, "y", startY)
                        )
                        endW, endH = float(get_value(kf_end, "width", startW)), float(
                            get_value(kf_end, "height", startH)
                        )

                        frame_annotations[f].append(
                            {
                                "label": label_name,
                                "rotation": get_value(kf_start, "rotation", 0),
                                "x": startX + (endX - startX) * ratio,
                                "y": startY + (endY - startY) * ratio,
                                "width": startW + (endW - startW) * ratio,
                                "height": startH + (endH - startH) * ratio,
                            }
                        )

        mapping_data["original_width"] = orig_width
        mapping_data["original_height"] = orig_height
        mapping_data["raw_ls2_results"] = raw_ls2_results
        mapping_data["interpolated_frames"] = frame_annotations

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=4)
        print(f"Mapping updated - {mapping_path}")


if __name__ == "__main__":
    main()
