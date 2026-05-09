from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from vaapi.client import Vaapi
from pathlib import Path
from typing import List
import requests
import random
import yaml
import json
import os


def get_secure_session():
    session = requests.Session()

    # Definiere die Retry-Strategie
    retry_strategy = Retry(
        total=5,  # Maximal 5 Versuche
        backoff_factor=1,  # Warte 1s, dann 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],  # Bei diesen Fehlern neu versuchen
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def create_dataset_json(
    log_ids: list,
    ls_project_ids: list,
    camera: str,
    target_class: str,
    v_client,
    l_client,
    output_path: str,
    split_ratio: float,
    seed: int,
):
    """
    Retrieves images and their corresponding bounding box annotations
    from Label Studio, converts them to YOLO format, and saves the dataset as a JSON file.
    The dataset is shuffled and split into training and validation sets.
    Args:
        log_ids (list): List of log IDs to retrieve images from.
        ls_project_id (int): A specific Labelstudio project.
        camera (str): Camera identifier to filter images by.
        v_client: client for retrieving images.
        l_client: Label Studio client for retrieving annotations.
        output_path (str): File path where the output JSON dataset will be saved.
        split_ratio (float): Ratio (0.0-1.0) for train/val split. Images with index < split_idx
                             are assigned to 'train', others to 'val'.
        seed (int): Random seed for reproducibility of dataset shuffling and split.
    Returns:
        None. Writes the dataset to a JSON file at output_path.
    Notes:
        - Only processes images with validated=True status. Validation is true if a human once clicked submit in labelstudio, even if there are no annotations.
    """

    dataset = {
        "metadata": {
            "seed": seed,
            "split_ratio": split_ratio,
            "camera": camera,
            "log_ids": log_ids,
            "ls_project_ids": ls_project_ids,
        },
        "images": [],
    }

    def sort_key_fn(image):
        return image.frame.frame_number

    if log_ids:
        task_dict = {}
        project_id = 0
        for log_id in log_ids:
            image_obj_list = v_client.image.list(
                log=log_id, camera=camera, validated=True
            )
            for img_obj in sorted(image_obj_list, key=sort_key_fn):
                new_project_id = int(
                    img_obj.labelstudio_url.split("/projects/")[1].split("/")[0]
                )
                # only fetch the list of tasks the first time its needed
                if not project_id == new_project_id:
                    project_id = new_project_id
                    all_tasks = l_client.tasks.list(project=project_id)
                    task_dict = {str(task.id): task for task in all_tasks}

                img_url = "https://logs.berlin-united.com/" + img_obj.image_url
                task_id = img_obj.labelstudio_url.split("=")[-1]

                try:
                    task = task_dict.get(task_id)
                    if not task:
                        continue
                    annotations = task.annotations
                    if annotations:
                        bbox_data = annotations[0].get("result", [])
                        yolo_results = convert_to_yolo(bbox_data, {target_class: 0})
                        dataset["images"].append(
                            {
                                "frame_number": f"{int(img_obj.frame.frame_number):07d}",
                                "url": img_url,
                                "labelstudio_url": img_obj.labelstudio_url,
                                "annotations": yolo_results,
                            }
                        )
                except Exception as e:
                    print(f"Failed to fetch Task {task_id}: {e}")

    elif ls_project_ids:
        print(f"Fetching tasks directly from Label Studio Projects: {ls_project_ids}")

        for proj_id in ls_project_ids:
            all_tasks = l_client.tasks.list(project=proj_id)

            for task in all_tasks:
                try:
                    task_id = (
                        str(task.id) if hasattr(task, "id") else str(task.get("id"))
                    )
                    annotations = (
                        task.annotations
                        if hasattr(task, "annotations")
                        else task.get("annotations", [])
                    )
                    task_data = (
                        task.data if hasattr(task, "data") else task.get("data", {})
                    )

                    if annotations:
                        anno_obj = annotations[0]
                        if not isinstance(anno_obj, dict):
                            anno_obj = (
                                anno_obj.model_dump()
                                if hasattr(anno_obj, "model_dump")
                                else vars(anno_obj)
                            )

                        bbox_data = anno_obj.get("result", [])
                        yolo_results = convert_to_yolo(bbox_data, {target_class: 0})

                        img_url = task_data.get("image") or task_data.get("img")
                        if not img_url:
                            continue

                        if "logs.berlin-united.com" not in img_url:
                            img_url = (
                                "https://logs.berlin-united.com/" + img_url.lstrip("/")
                            )

                        dataset["images"].append(
                            {
                                "frame_number": f"{int(task_id):07d}",
                                "url": img_url,
                                "labelstudio_url": f"https://labelstudio-api.berlin-united.com/projects/{proj_id}/data?task={task_id}",
                                "annotations": yolo_results,
                            }
                        )
                except Exception as e:
                    print(f"Failed to fetch Task {task_id} from Project {proj_id}: {e}")

    random.seed(seed)
    random.shuffle(dataset["images"])
    split_idx = int(len(dataset["images"]) * split_ratio)
    for i, entry in enumerate(dataset["images"]):
        entry["split"] = "train" if i < split_idx else "val"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)


def create_local_yolo_ds(dataset_file: str, run_path: str, target_class: str) -> int:
    """
    Converts JSON metadata with image URLs and annotations into a local YOLO dataset structure.
    Downloads images from URLs and saves them locally and creates YOLO-format label files for each image.
    Also generates dataset.yaml configuration file
    Args:
        dataset_file (str): Path to JSON file containing dataset metadata with fields:
            - images (list): List of image entries, each containing:
                - url (str): URL to download the image
                - annotations (list): YOLO format annotations for the image
                - split (str, optional): Dataset split ('train' or 'val'). Defaults to 'train'
        run_path (str): Root directory path where the dataset and configuration will be created.
                       Creates 'dataset/' subdirectory and 'dataset.yaml' file here.
    Returns:
        int: Status code indicating success/failure.
    """
    with open(dataset_file) as json_data:
        dataset = json.load(json_data)

    output_path = f"{run_path}/dataset"

    # 1. Create directory structure
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)

    for entry in dataset["images"]:
        subset = entry.get("split", "train")  # if there is no split just use train

        url = entry["url"]
        annotations = entry["annotations"]

        # Create a unique filename based on the URL or index
        filename = Path(url).stem  # e.g., '0008837'
        img_ext = Path(url).suffix  # e.g., '.png'

        try:
            # 2. Download the image
            img_response = requests.get(url, timeout=10)
            if img_response.status_code == 200:
                img_path = os.path.join(
                    output_path, f"images/{subset}/{filename}{img_ext}"
                )
                with open(img_path, "wb") as f:
                    f.write(img_response.content)

                # 3. Create the label file
                label_path = os.path.join(
                    output_path, f"labels/{subset}/{filename}.txt"
                )
                with open(label_path, "w") as f:
                    for ann in annotations:
                        f.write(f"{ann}\n")
            else:
                print(f"Failed to download (Status {img_response.status_code}): {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    data = {
        "path": os.path.abspath(output_path),  # dataset root dir (absolut!)
        "train": "images/train",  # train images (relative to 'path')
        "val": "images/val",  # val images (relative to 'path')
        "names": {0: target_class},
    }

    with open(f"{run_path}/dataset.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def convert_to_yolo(bbox_data, class_map) -> List:
    """
    Convert bounding box annotations from Label Studio format to YOLO format.
    Args:
        bbox_data (List[Dict]): List of Label Studio annotation items, each containing
            a "value" key with rectangle label information including:
            - rectanglelabels (List[str]): List of class labels
            - x (float): Left edge x-coordinate as percentage of image width (0-100)
            - y (float): Top edge y-coordinate as percentage of image height (0-100)
            - width (float): Bounding box width as percentage of image width (0-100)
            - height (float): Bounding box height as percentage of image height (0-100)
        class_map (Dict[str, int]): Mapping from class label names to their corresponding
            class IDs in the YOLO format.
    Returns:
        List[str]: List of YOLO format strings, one per bounding box. Each string contains
            space-separated values: "class_id x_center y_center width height" where
            coordinates and dimensions are normalized to [0.0, 1.0] with 6 decimal places
            of precision.
    Note:
        - Label Studio uses percentages (0-100) while YOLO uses normalized coordinates (0-1).
        - YOLO format uses center coordinates, while Label Studio uses top-left corner.
    """

    yolo_lines = []

    for item in bbox_data:
        val = item["value"]

        # 1. Get the class ID from the label list
        label_name = val["rectanglelabels"][0]
        if label_name not in class_map:  # skip if no known label
            continue
        class_id = class_map.get(label_name, 0)  # Defaults to 0 if not found

        # 2. Extract Label Studio percentages (top-left)
        x_tl = val["x"]
        y_tl = val["y"]
        w_ls = val["width"]
        h_ls = val["height"]

        # 3. Calculate YOLO center coordinates (0.0 to 1.0)
        x_center = (x_tl + (w_ls / 2)) / 100
        y_center = (y_tl + (h_ls / 2)) / 100
        w_yolo = w_ls / 100
        h_yolo = h_ls / 100

        # 4. Format string: class x_center y_center width height
        # Using 6 decimal places for precision
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(line)

    return yolo_lines


def get_project_id(v_client, log_id: str, camera: str) -> int:
    """Get the Label Studio project ID for a given log and camera."""
    image_obj_list = v_client.image.list(log=log_id, camera=camera, validated=True)
    for img_obj in image_obj_list:
        return int(img_obj.labelstudio_url.split("/projects/")[1].split("/")[0])
    raise ValueError(f"No images found for log {log_id} and camera {camera}")


def predict_on_image(ls_client, task_id, predictions, score=0.0):
    """Push YOLO predictions to a Label Studio task as pre-annotations."""
    ls_client.predictions.create(task=task_id, score=score, result=predictions)


def get_log_ids_per_game(game_id):
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    logs = v_client.logs.list(game=game_id)
    return [log.id for log in logs]


def get_log_ids_per_event(event_id):
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    logs = v_client.logs.list(event=event_id)
    return [log.id for log in logs]
