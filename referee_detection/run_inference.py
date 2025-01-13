"""
    Runs inference on specified images from VAT
"""
from vaapi.client import Vaapi
import argparse
import os


def get_data():
    # TODO check which state should I get here?
    """
    response = client.behavior_frame_option.filter(
        log_id=168,
        option_name="decide_game_state",
        state_name="standby",
    )
    print(response)
    """
    pass

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
        images = client.image.list(log=log_id, camera="TOP",annotation=0)
        for idx, img in enumerate(tqdm(images)):
            if args.local:
                image_path = Path(log_root_path) / img.image_url
                results = model.predict(image_path, conf=0.8, verbose=False)
            else:
                url = base_url + img.image_url
                results = model.predict(url, conf=0.8, verbose=False)

            for result in results:
                
                # result.save(filename=Path(img.image_url).name)
                bbox = []
                print(result.boxes.cls)
                for i,cls in enumerate(result.boxes.cls.tolist()):
                    bbox.append({
                        "x":result.boxes.xywh.tolist()[i][0],
                        "y":result.boxes.xywh.tolist()[i][1],
                        "id":123,
                        "label":result.names.get(cls),
                        "width":result.boxes.xywh.tolist()[i][2],
                        "height":result.boxes.xywh.tolist()[i][3]
                    })

                boxes = {
                "bbox": bbox
                }
                # print(boxes)
                if idx==5:
                    quit()
        break
