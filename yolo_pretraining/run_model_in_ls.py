"""
    run the best model on new data in labelstudio
"""
from pathlib import Path
from label_studio_sdk import Client
from minio import Minio
from ultralytics import YOLO


LABEL_STUDIO_URL = "https://ls.berlinunited-cloud.de/"
API_KEY = "6cb437fb6daf7deb1694670a6f00120112535687"
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
)

def download_from_minio(project, filename, output_folder):
    bucket_name = project.title
    output = Path(output_folder) / filename
    mclient.fget_object(bucket_name, filename, output)

if __name__ == "__main__":
    # load the current best model
    model = YOLO('./detect/train24/weights/best.pt')
    # select the labelstudio projects you wish to annotate with the model
    project_list = [39, 40, 41, 42, 43, 44]

    for id in project_list:
        project = ls.get_project(id)
        task_ids = project.get_unlabeled_tasks_ids()

        for task in task_ids:
            task_output = project.get_task(task)
            image_file_name = task_output["storage_filename"]
            download_from_minio(project, image_file_name, "/tmp/")
            image_path = Path("/tmp/") / image_file_name
            results = model.predict(image_path, conf=0.8, verbose=False)
            
            # TODO check that this works for multiple boxes
            for result in results:  
                if result.boxes.xywh.nelement() > 0:
                    label_studio_result_list = list()
                    for idx, box in enumerate(result.boxes.xywh.cpu().numpy()):
                        x = box[0] - box[2] / 2 # x and y of the output are the center coordinates
                        y = box[1] - box[3] / 2
                        w = box[2] / result.orig_img.shape[1] * 100
                        h = box[3] / result.orig_img.shape[0] * 100
                        x = x / result.orig_img.shape[1] * 100
                        y = y / result.orig_img.shape[0] * 100
                        print()
                        print(x ,y, w, h)
                        # create one annotation with multiple results
                        single_ls_result = {'type': 'rectanglelabels', 'value': {'x': x, 'y': y, 'width': w, 'height': h, 'rotation': 0, 'rectanglelabels': [result.names[result.boxes.cls.cpu().numpy()[idx]]]}, 'to_name': 'image', 'from_name': 'label', 'original_width': result.orig_img.shape[1], 'original_height': result.orig_img.shape[0]}
                        label_studio_result_list.append(single_ls_result)

                    # TODO add data to my user
                    # TODO add setting about data tracking
                    #ls_result = {"result": label_studio_result_list, 'completed_by': {'id': 2, 'first_name': 'eliza', 'last_name': 'bot', 'avatar': None, 'email': 'bot@berlinunited.com', 'initials': 'eb'}}
                    ls_result = {"result": label_studio_result_list, 'completed_by': 2}
                    a = project.create_annotation(task, **ls_result)
