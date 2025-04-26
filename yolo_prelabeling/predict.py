"""
    Run Yolo v11 Models on our full images
"""
import os,sys
from vaapi.client import Vaapi
from tqdm import tqdm
from pathlib import Path
import argparse

from prelabeling_tools.model_selector import get_yolo_model

helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)

from helper import get_alive_fileserver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-l", "--local", action="store_true", default=False)
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    log_root_path = os.environ.get("VAT_LOG_ROOT")
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    log_server = get_alive_fileserver()
    model, class_mapping = get_yolo_model(camera=args.camera, model=args.model)
    log = client.logs.list()

    def sort_key_fn(log):
        return log.id

    # TODO when should we skip a log???
    for log in sorted(log, key=sort_key_fn, reverse=True):
        print(f"{log.id}: {log.log_path}")
        #if exclude_annotated parameter is set all images with an existing annotation are not included in the response
        # TODO maybe we want this to be handled differently?
        # like we want to filter for validated and maybe other values
        # We could add another endpoint for this

        # we could have: exclude_empty, exclude_annotated -> both set only images that we havent see before (if we set empty)
        # FIXME how to query images and annotations together? like if we want to create a dataset for further training, how would we do that?
        # - we need to be able to filter for camera and jpeg and so on
        images = client.image.list(log=log.id, camera=args.camera, exclude_annotated=True)

        for idx, img in enumerate(tqdm(images)):
            if args.local:
                image_path = Path(log_root_path) / img.image_url
                results = model.predict(image_path, conf=0.8, verbose=False)
            else:
                # TODO load that image manually in a temp folder or cleanup
                url = log_server + img.image_url
                results = model.predict(url, conf=0.8, verbose=False)

            # results is always of length 1 since we only put in one image at the time
            for result in results:
                for i, class_id in enumerate(result.boxes.cls.tolist()):
                    # ignore other classes then ball for now
                    if int(class_id) != 0:
                        continue
                    #result.save(filename=Path(img.image_url).name)
                    # TODO: maybe we can use xywhn
                    cx = result.boxes.xywh.tolist()[i][0]
                    cy = result.boxes.xywh.tolist()[i][1]
                    width = result.boxes.xywh.tolist()[i][2]
                    height = result.boxes.xywh.tolist()[i][3]

                    data = {
                        "x": ( cx - ( width / 2 ) ) / 640,
                        "y": ( cy - ( height / 2 ) ) / 480,
                        "width": width / 640,
                        "height": height / 480
                    }
                    # we don't include confidence in the database because that
                    # would be void if we adjust a bounding box
                    response = client.annotations.create(
                        image_id=img.id,
                        type="bbox",
                        class_name="ball",
                        is_empty=False,
                        data=data,
                    )
                    

