"""
    Runs inference on specified images from VAT
"""
from vaapi.client import Vaapi
import argparse
import os, sys
import uuid
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)

#from helper import get_file_from_server

def get_data(log_id=168):
    # TODO check which state should I get here?
    response = client.behavior_frame_option.filter(
        log_id=log_id,
        option_name="decide_game_state",
        state_name="standby",
    )

    # TODO create a framefilter
    resp = client.frame_filter.create(
        log_id=log_id,
        frames={"frame_list": response},
    )
    images = client.image.list(log=log_id,camera="TOP",exclude_annotated=True, use_filter=1)
    return images

if __name__ == "__main__":
    # TODO use argparse for setting the model for now, later maybe we can utilize mlflow to automatically select the best model and download it?
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    args = parser.parse_args()

    log_root_path = os.environ.get("VAT_LOG_ROOT")
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    
    images = get_data(log_id=168)
    model = YOLO("yolo11n.pt")

    for idx, img in enumerate(tqdm(images)):
        url = url = "https://logs.berlin-united.com/" + img.image_url
        results = model.predict(url, conf=0.6, verbose=False, classes=[0])

        for result in results:
            #result.save(filename=Path(img.image_url).name)
            bbox = []
            print(result.boxes.cls)
            for i,cls in enumerate(result.boxes.cls.tolist()):
                # TODO: maybe we can use xywhn
                cx = result.boxes.xywh.tolist()[i][0]
                cy = result.boxes.xywh.tolist()[i][1]
                width = result.boxes.xywh.tolist()[i][2]
                height = result.boxes.xywh.tolist()[i][3]
                bbox.append({
                        "x": ( cx - ( width / 2 ) ) / 640,
                        "y": ( cy - ( height / 2 ) ) / 480,
                        "id":uuid.uuid4().hex[:9].upper(),
                        "label":result.names.get(cls),
                        "width": width / 640,
                        "height": height / 480
                    })

            boxes = {
                "bbox": bbox
            }
            client.annotations.create(img.id, annotation=boxes)

        if idx > 10:
            quit()
