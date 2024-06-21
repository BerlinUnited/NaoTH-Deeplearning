import cv2
import numpy as np
from PIL import Image as PIL_Image


def yuv888_bytes_to_yuv422_array(yuv888_bytes, width, height):
    yuv422 = np.ndarray(width * height * 2, np.uint8)

    for i in range(0, width * height, 2):
        yuv422[i * 2] = yuv888_bytes[i * 3]
        yuv422[i * 2 + 1] = (yuv888_bytes[i * 3 + 1] + yuv888_bytes[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = yuv888_bytes[i * 3 + 3]
        yuv422[i * 2 + 3] = (yuv888_bytes[i * 3 + 2] + yuv888_bytes[i * 3 + 5]) / 2.0

    return yuv422


def load_image_as_yuv422_original(image_filename, patch_size=16):
    """
    this functions loads an image from a file to the correct format for the naoth library.
    This function remains here for backwards compability. However use the newer functions. This one
    was only intented to be used with patches from our keypoint detection
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2

    # we load the image in opencv BGR format here
    cv_img = cv2.imread(image_filename)
    cv_img = cv2.resize(cv_img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    # convert image to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(patch_size * patch_size * 2, np.uint8)

    for i in range(0, patch_size * patch_size, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0
    # output format is 16x16x2
    # the first channel is all y and the second channel is interleaved u and v
    return yuv422


def load_image_as_yuv422_cv2(image_filename) -> np.ndarray:

    # standard resolution: 640x480
    cv_img = cv2.imread(image_filename)

    # cv2 shape is (height, width, channels)
    width, height = cv_img.shape[1], cv_img.shape[0]

    yuv888_bytes = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = yuv888_bytes_to_yuv422_array(yuv888_bytes, width=width, height=height)

    # cv2 size is (height, width)
    # Pillow size is (width, height)
    # we need to ensure consistent output shapes for all image loading functions
    yuv422 = yuv422.reshape(height, width, 2)

    return yuv422


def load_image_as_yuv422_y_only_cv2(image_filename) -> np.ndarray:
    yuv422 = load_image_as_yuv422_cv2(image_filename)
    height, width = yuv422.shape[:2]

    yuv422_y_only = yuv422[..., 0]
    yuv422_y_only = yuv422_y_only.reshape(height, width, 1)

    return yuv422_y_only


def load_image_as_yuv422_pil(image_filename) -> np.ndarray:
    im = PIL_Image.open(image_filename)
    ycbcr = im.convert("YCbCr")
    width, height = ycbcr.size

    yuv422 = yuv888_bytes_to_yuv422_array(ycbcr.tobytes(), width=width, height=height)

    # cv2 size is (height, width)
    # Pillow size is (width, height)
    # we need to ensure consistent output shapes for all image loading functions
    yuv422 = yuv422.reshape(height, width, 2)

    return yuv422


def load_image_as_yuv422_y_only_pil(image_filename) -> np.ndarray:
    yuv422 = load_image_as_yuv422_pil(image_filename)
    height, width = yuv422.shape[:2]

    yuv422_y_only = yuv422[..., 0]
    yuv422_y_only = yuv422_y_only.reshape(height, width, 1)

    return yuv422_y_only


def get_meta_from_png(img_path):
    return PIL_Image.open(img_path).info


def get_multiclass_from_meta(
    meta,
    min_ball_intersect=0.5,
    min_penalty_intersect=0.75,
    min_robot_intersect=0.4,
):
    ball_intersect = float(meta.get("ball_intersect", 0))
    penalty_intersect = float(meta.get("penalty_intersect", 0))
    robot_intersect = float(meta.get("robot_intersect", 0))

    ball_class = int(ball_intersect > min_ball_intersect)
    penalty_class = int(penalty_intersect > min_penalty_intersect)
    robot_class = int(robot_intersect > min_robot_intersect)

    return np.array([ball_class, penalty_class, robot_class])


def get_ball_center_radius_from_meta(meta):
    ball_center_x = float(meta.get("ball_center_x", 0))
    ball_center_y = float(meta.get("ball_center_y", 0))
    ball_radius = float(meta.get("ball_radius", 0))

    return np.array([ball_center_x, ball_center_y, ball_radius])
