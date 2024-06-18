"""
TODO generate patches and upload them to the bucket
TODO I want to use the same bucket for all images, make sure it does not break labelstudio -> write tests for that
TODO a thousand other things.
TODO sort those patches as well similar to the other patch exporter
TODO can we capture how many balls it misses completly?
"""
import os
import sys
import tempfile
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from helper import get_postgres_cursor, get_minio_client, get_labelstudio_client, load_image_as_yuv888_y_only

def get_buckets_with_bottom_images():

    select_statement = f"""        
    SELECT log_path, bucket_bottom, ls_project_bottom
    FROM robot_logs
    WHERE bucket_bottom IS NOT NULL
    AND ls_project_bottom IS NOT NULL
    AND bottom_validated = true
    """
    cur = get_postgres_cursor()

    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=False,
        help="Output folder for the downloaded images and generated patches. If not set a folder in /tmp will be used",
    )

    data_bottom = get_buckets_with_bottom_images()
    data_bottom = [
        (logpath, bucketname, ls_project_id, "bucket_bottom_patches")
        for logpath, bucketname, ls_project_id in data_bottom
    ]
    ls = get_labelstudio_client()
    mclient = get_minio_client()
    new_model = tf.keras.models.load_model('models/lyrical-sloth-580/best.keras')
    for logpath, bucketname, ls_project_id, db_field in data_bottom:
        ls_project = ls.get_project(id=ls_project_id)
        labeled_tasks = ls_project.get_labeled_tasks()

        if not labeled_tasks:
            print("\tNo labeled tasks found for this project, skipping...")
            break

        tmp_download_folder = tempfile.TemporaryDirectory()
        #output_patch_folder = Path(tmp_download_folder.name) / "patches_segmentation"
        tmp_image_folder = Path("test") / "images" / bucketname # FIXME that will overwrite things
        Path(tmp_image_folder).mkdir(exist_ok=True, parents=True)
        output_patch_folder = Path("test") / "patches_segmentation" / bucketname
        output_patch_folder.mkdir(exist_ok=True, parents=True)
        ball_folder = output_patch_folder / "ball"
        Path(ball_folder).mkdir(exist_ok=True, parents=True)

        print(f"\tCreated temporary directory {tmp_download_folder}")

        for task in tqdm(labeled_tasks):
            image_file_name = task["storage_filename"]
            output_file = Path(tmp_image_folder) / image_file_name
            if not output_file.exists():
                mclient.fget_object(bucketname, image_file_name, str(output_file))

            image_orig = cv2.imread(str(output_file))
            image = load_image_as_yuv888_y_only(str(output_file), rescale=True, subsampling=True)
            image_input = np.expand_dims(image, axis=0)
            result = new_model.predict(image_input)
            result = result[0]
            ball_result = result[:,:,0]
            factor = 32
            upscaled_array = np.repeat(np.repeat(ball_result, factor, axis=0), factor, axis=1)

            threshold_value = 0.5
            binary_mask = np.uint8(upscaled_array > threshold_value) * 255
            sum = np.sum(binary_mask)
            if sum == 0:
                continue
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                crop_img = image_orig[y : y+h, x : x+w]
                patch_file_name = Path(ball_folder) / (
                    bucketname
                    + "_"
                    + Path(output_file).stem
                    + f"_{idx}_threshold_{threshold_value}.png"
                )
                cv2.imwrite(str(patch_file_name), crop_img)
                break