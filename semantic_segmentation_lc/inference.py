import os
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import sys

# TODO iterate over all images
new_model = tf.keras.models.load_model("segment_test.keras")
helper_path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(helper_path)
from tools.image_loader import load_image_as_yuv422_pil


dataset_dir = "test/image"
for image_file in os.listdir(dataset_dir):
    image_file_path = os.path.join(dataset_dir, image_file)
    print(image_file_path)
    if os.path.isfile(image_file_path):
        # Load the image
        # image = Image.open(image_file_path)
        image_yuv = load_image_as_yuv422_pil(
            str(
                "/home/stella/robocup/naoth-deeplearning/semantic_segmentation_lc/test/image/wzkgvqiattfiqhbwyptmtm_0031451.png"
            )
        )
        # image = cv2.imread(str(image_file_path))
        image_yuv = cv2.resize(image_yuv, (320, 240))
        image_yuv = np.expand_dims(image_yuv, axis=0)
        # print(image_yuv.shape)
        # image_yuv = np.transpose(image_yuv, (0, 2, 1, 3))
        # print(image_yuv.shape)

        a = new_model.predict(image_yuv)
        print(a.shape)
        print(a)
        print(np.max(a[0, :, :, :]))
        print(np.min(a[0, :, :, :]))

        print()
        np.save("test", a)
        quit()
