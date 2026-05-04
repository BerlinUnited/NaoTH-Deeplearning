import os
import json
from config import LS_URL, LS_TARGET_PROJECT_ID, PROJECT_IDS, DOWNLOAD_DIR, get_ls_client, get_headers
from utils import get_value, get_image_dimensions_from_url


def main():
    ls = get_ls_client()
    ls_tasks = ls.tasks.list(project=LS_TARGET_PROJECT_ID)

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
        task_ids = mapping_data.get("task_ids", mapping_data.get("ls1_task_ids", []))
        total_images = len(task_ids)
        if total_images == 0:
            continue

        video_task_id = None
        for task in ls_tasks:
            file_url = (
                get_value(get_value(task, "data", {}), "video")
                or get_value(get_value(task, "data", {}), "file_upload")
                or ""
            )
            if target_video_name in str(file_url):
                video_task_id = get_value(task, "id")
                break

        if not video_task_id:
            print(f"Video was not found in LS. Skipping it...")
            continue

        full_task = ls.tasks.get(id=video_task_id)
        annotations = get_value(full_task, "annotations", [])
        if not annotations:
            print(f"No boxes are drawn yet")
            continue

        first_task_id = task_ids[0]
        first_task_info = ls.tasks.get(id=first_task_id)
        img_url = get_value(
            (
                first_task_info.data
                if hasattr(first_task_info, "data")
                else first_task_info.get("data", {})
            ),
            "image",
        )
        orig_width, orig_height = get_image_dimensions_from_url(img_url)

        raw_results = get_value(annotations[0], "result", [])
        frame_annotations = {i: [] for i in range(total_images)}

        for result in raw_results:
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
        mapping_data["raw_results"] = raw_results
        mapping_data["interpolated_frames"] = frame_annotations

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=4)
        print(f"Mapping updated - {mapping_path}")


if __name__ == "__main__":
    main()
