from pathlib import Path
from datetime import datetime
import os
import sys
import yaml
import ultralytics
from vaapi.client import Vaapi

helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)

from helper import get_file_from_server

dataset_name = Path("datasets") / Path(f"referee_{datetime.now().strftime('%Y-%m-%d')}")   # TODO add date
Path(dataset_name).mkdir(parents=True, exist_ok=True)


def download_images():
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    # FIXME this is outdated now
    # TODO download all images and get all images where bounding box
    response = client.annotations.list(id=168)
    for annotation in response:
        # ignore empty annotations
        if not annotation.annotation:
            continue
        if annotation.annotation["bbox"] == []:
            continue

        # download image if annotation exists
        image = client.image.get(annotation.image)
        filename = Path(image.image_url).name
        destination = Path(dataset_name) / "images"
        get_file_from_server("https://logs.berlin-united.com/" + image.image_url,f"{destination}/{filename}")

        destination = Path(dataset_name) / "labels"
        destination.mkdir(parents=True, exist_ok=True)
        with open(f"{destination}/{Path(filename).with_suffix('.txt')}","w") as f:
            bboxes = annotation.annotation["bbox"]
            for box in bboxes:
                width_px = box["width"] * 640
                height_px = box["height"] * 480
                x_px = box["x"] * 640
                y_px = box["y"] * 480
                cx = x_px + (width_px / 2)
                cy = y_px + (height_px / 2)
                #print(f"{0} {cx} {cy} {width_px} {height_px}\n")
                f.write(f"{0} {cx / 640} {cy / 480} {box['width']} {box['height']}\n")
                print(f"{0} {cx / 640} {cy / 480} {box['width']} {box['height']}")


def main():
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    download_images(client)

    data = dict(
    # TODO why ../ ???
    path=f"{dataset_name}",
    train="autosplit_train.txt",
    val="autosplit_val.txt",
    names={0: "person"},
    )

    with open(f"{dataset_name}.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

    ultralytics.data.utils.autosplit(f"{dataset_name}/images", weights=(0.5, 0.5, 0.0), annotated_only=False)

if __name__ == "__main__":
    main()




