import os
import json
from config import (
    LS_TARGET_PROJECT_ID,
    PROJECT_IDS,
    DOWNLOAD_DIR,
    LS_FROM_NAME,
    LS_TO_NAME,
    get_ls1_client,
)
from utils import round_floats


def main():
    ls = get_ls1_client()

    for pid in PROJECT_IDS:
        print(f"\n--- Project {pid} ---")

        mapping_path = os.path.join(DOWNLOAD_DIR, f"mapping_project_{pid}.json")

        if not os.path.exists(mapping_path):
            continue

        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)

        task_ids = mapping_data.get("task_ids", mapping_data.get("ls1_task_ids", []))
        orig_width = mapping_data.get("original_width")
        orig_height = mapping_data.get("original_height")
        interpolated_frames = mapping_data.get("interpolated_frames", {})

        history_file = f"annotations_project_{pid}.json"
        local_history = {}
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    local_history = json.load(f)
                except Exception:
                    pass

        upload_queue = []
        for f_idx, t_id in enumerate(task_ids):
            annots = interpolated_frames.get(str(f_idx), [])
            t_id = str(t_id)

            raw_new_results = [
                {
                    "from_name": LS_FROM_NAME,
                    "to_name": LS_TO_NAME,
                    "type": "rectanglelabels",
                    "original_width": orig_width,
                    "original_height": orig_height,
                    "image_rotation": 0,
                    "value": {
                        "x": a["x"],
                        "y": a["y"],
                        "width": a["width"],
                        "height": a["height"],
                        "rotation": a["rotation"],
                        "rectanglelabels": [a["label"]],
                    },
                }
                for a in annots
            ]

            new_results = round_floats(raw_new_results)
            history_entry = local_history.get(t_id)

            if history_entry:
                old_results = history_entry.get("result", [])
                if json.dumps(new_results, sort_keys=True) == json.dumps(
                    old_results, sort_keys=True
                ):
                    continue

            upload_queue.append((t_id, new_results))

        if not upload_queue:
            print(f"Everything done for project {pid}")
            continue

        print(f"Uploading {len(upload_queue)} tasks for project {pid}")
        success = 0
        for t_id, res in upload_queue:
            try:
                task_info = ls.tasks.get(task_id=int(t_id))
                existing_annots = (
                    getattr(task_info, "annotations", [])
                    if not isinstance(task_info, dict)
                    else task_info.get("annotations", [])
                )

                for ann in existing_annots:
                    ann_id = (
                        getattr(ann, "id", None)
                        if not isinstance(ann, dict)
                        else ann.get("id")
                    )
                    if ann_id:
                        ls.annotations.delete(task_id=ann_id)

                created_annot = ls.annotations.create(task_id=int(t_id), result=res)
                annot_id = (
                    getattr(created_annot, "id", None)
                    if not isinstance(created_annot, dict)
                    else created_annot.get("id")
                )

                local_history[t_id] = {"annotation_id": annot_id, "result": res}

                success += 1
                if success % 10 == 0:
                    with open(history_file, "w") as f:
                        json.dump(local_history, f)
                print(f"   ... {success} / {len(upload_queue)}", end="\r")
            except Exception as e:
                print(f"\nError with task {t_id}: {e}")

        with open(history_file, "w") as f:
            json.dump(local_history, f)
        print(f"\nProject {pid} finished")


if __name__ == "__main__":
    main()
