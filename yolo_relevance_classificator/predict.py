"""
    Run Yolo v11 Models on our full images
"""
import os
import sys
import requests
from vaapi.client import Vaapi
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import argparse


helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)

from helper import get_file_from_server


def get_log_server(timeout=2):
    url = "https://logs.berlin-united.com/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        print(f"Server {url} is alive.")
        return url
    except requests.exceptions.RequestException as e:
        print(f"Server {url} is not reachable: {e}")
        return "https://logs.naoth.de/"

if __name__ == "__main__":
    # TODO use argparse for setting the model for now, later maybe we can utilize mlflow to automatically select the best model and download it?
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="2024-04-27-yolov8s-top-indecisive-snake-51.pt")
    parser.add_argument("-f", "--local", action="store_true", default=False)
    args = parser.parse_args()

    log_root_path = os.environ.get("VAT_LOG_ROOT")
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    log_server = get_log_server()

    data = client.logs.list()
    print(f"https://models.naoth.de/{args.model}")
    if not os.path.isfile(f"./models/{args.model}"):
        get_file_from_server(f"https://models.naoth.de/{args.model}",f"./models/{args.model}")

    #model = YOLO("yolo11n.pt")
    # TODO get the best performing model from mlflow
    model = YOLO(f"./models/{args.model}")
    def sort_key_fn(data):
        return data.log_path

    for data in sorted(data, key=sort_key_fn, reverse=True):
        log_id = data.id
        print(data.log_path)
        #if exclude_annotated parameter is set all images with an existing annotation are not included in the response
        images = client.image.list(log=log_id,camera="TOP",exclude_annotated=True)
        interesting = []
        for idx, img in enumerate(tqdm(images)):
            if args.local:
                image_path = Path(log_root_path) / img.image_url
                results = model.predict(image_path, conf=0.7, verbose=False)
            else:
                url = log_server + img.image_url
                results = model.predict(url, conf=0.7, verbose=False)
            
            
            for result in results:
                #an image is labeled as interesting if the model could find a ball with a confidence over 70%
                if 0 in result.boxes.cls.tolist():
                        interesting.append(img.frame_number)
            
        print()
        print(len(interesting))
        print()
        # we include the list of interesting images in the frame filter, when enabling the filter only these images are returned
        client.frame_filter.create(
        log_id=log_id,
        frames={"frame_list": interesting}
        )
            
        break
