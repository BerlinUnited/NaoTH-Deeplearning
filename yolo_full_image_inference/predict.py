"""
    Run Yolo v11 Models on our full images
"""
import os
import requests
from vaapi.client import Vaapi
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import argparse


def is_server_alive(url, timeout=2):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        print(f"Server {url} is alive.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Server {url} is not reachable: {e}")
        return False


if __name__ == "__main__":
    # TODO use argparse for setting the model for now, later maybe we can utilize mlflow to automatically select the best model and download it?
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-f", "--local", action="store_true", default=False)
    args = parser.parse_args()

    log_root_path = os.environ.get("VAT_LOG_ROOT")
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    online = is_server_alive("https://logs.berlin-united.com/")
    if online:
        base_url = "https://logs.berlin-united.com/"
    else:
        base_urlurl = "https://logs.naoth.de/"
    data = client.logs.list()

    #model = YOLO("yolo11n.pt")
    # TODO get the best performing model from mlflow
    model = YOLO("2024-04-16-yolov8s-bottom-resilient-eel-775.pt")
    def sort_key_fn(data):
        return data.log_path

    for data in sorted(data, key=sort_key_fn, reverse=True):
        log_id = data.id
        print(data.log_path)
        # TODO figure out a way to only get images that do not have annotations
        images = client.image.list(log=log_id, camera="TOP")

        for idx, img in enumerate(tqdm(images)):
            if args.local:
                image_path = Path(log_root_path) / img.image_url
                results = model.predict(image_path, conf=0.8, verbose=False)
            else:
                url = base_url + img.image_url
                results = model.predict(url, conf=0.8, verbose=False)

            for result in results:
                result.save(filename=Path(img.image_url).name) 
        break
