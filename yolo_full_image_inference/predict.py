"""
    Run Yolo v11 Models on our full images
"""
import os
import requests,uuid
from vaapi.client import Vaapi
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import argparse
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
import json
#specify model name, if not in local models folder it will attempt to downlaod it from models.naoth.de
model = "2024-04-27-yolov8s-top-indecisive-snake-51.pt"

# copied from tools/helper.py
def get_file_from_server(origin, target):
    # FIXME move to naoth python package
    def dl_progress(count, block_size, total_size):
        print(
            "\r",
            "Progress: {0:.2%}".format(min((count * block_size) / total_size, 1.0)),
            sep="",
            end="",
            flush=True,
        )

    if not Path(target).exists():
        target_folder = Path(target).parent
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        return

    error_msg = "URL fetch failure on {} : {} -- {}"
    try:
        try:
            urlretrieve(origin, target, dl_progress)
            print("\nFinished")
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.reason))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if Path(target).exists():
            Path(target).unlink()
        raise


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
    print(f"https://models.naoth.de/{model}")
    if not os.path.isfile(f"./models/{model}"):
        get_file_from_server(f"https://models.naoth.de/{model}",f"./models/{model}")

    #model = YOLO("yolo11n.pt")
    # TODO get the best performing model from mlflow
    model = YOLO(f"./models/{model}")
    def sort_key_fn(data):
        return data.log_path

    for data in sorted(data, key=sort_key_fn, reverse=True):
        log_id = data.id
        print(data.log_path)
        # TODO figure out a way to only get images that do not have annotations
        # annotation 0 -> only images with no annotations (not in main api as of now)
        images = client.image.list(log=log_id, camera="TOP",exclude_annotated=True)
        for idx, img in enumerate(tqdm(images)):
            if args.local:
                image_path = Path(log_root_path) / img.image_url
                results = model.predict(image_path, conf=0.8, verbose=False)
            else:
                url = base_url + img.image_url
                results = model.predict(url, conf=0.8, verbose=False)

            for result in results:
                
                result.save(filename=Path(img.image_url).name)
                bbox = []
                print(result.boxes.cls)
                #todo add uuid gen for id field
                # todo 
                for i,cls in enumerate(result.boxes.cls.tolist()):
                    bbox.append({
                        "x":result.boxes.xywh.tolist()[i][0]-(result.boxes.xywh.tolist()[i][2]/2),
                        "y":result.boxes.xywh.tolist()[i][1]-(result.boxes.xywh.tolist()[i][3]/2),
                        "id":str(uuid.uuid4()),
                        "label":result.names.get(cls),
                        "width":result.boxes.xywh.tolist()[i][2],
                        "height":result.boxes.xywh.tolist()[i][3]
                    })

                boxes = {
                "bbox": bbox
                }
                print(boxes)
                client.annotations.create(img.id,annotation=boxes)
                if idx==5:
                    quit()
                #TODO bulk create using sdk
        break
