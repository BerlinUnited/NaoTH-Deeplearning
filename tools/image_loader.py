import numpy as np
from PIL import Image as PIL_Image


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


def load_image_as_yuv422(image_filename, rescale=False):
    """
    this functions loads an image from a file to the correct format for the naoth library
    # FIXME: i don't trust this function
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2

    # y = 240
    # x = 320
    cv_img = cv2.imread(image_filename)
    x = cv_img.shape[1]
    y = cv_img.shape[0]
    # cv_img = cv2.resize(
    #    cv_img, (240,320), interpolation=cv2.INTER_NEAREST
    # )
    # print(cv_img.shape, x,y)
    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(x * y * 2, np.uint8)

    for i in range(0, x * y, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0

    # TODO is this the correct order?
    image_yuv = yuv422.reshape(y, x, 2)

    if rescale:
        image_yuv = image_yuv / 255.0

    return image_yuv


def load_image_as_yuv422_y_only(image_filename, rescale=False, subsample=False):
    """
    this functions loads an image from a file to the correct format for the naoth library
    # FIXME: i don't trust this function
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2

    cv_img = cv2.imread(image_filename)
    x = cv_img.shape[1]
    y = cv_img.shape[0]

    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(y * x * 2, np.uint8)

    for i in range(0, y * x, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0

    # TODO is this the correct order?
    image_yuv = yuv422.reshape(y, x, 2)
    image_y = image_yuv[..., 0]
    image_y = image_y.reshape(y, x, 1)
    print(image_y.shape)

    if subsample:
        # half the resolution because semantic segmentation requires it
        image_y = image_y[::2, ::2]

    if rescale:
        image_y = image_y / 255.0

    return image_y


def load_image_as_yuv888(
    image_filename,
    rescale=False,
    subsample=False,
    resize_to=None,
    resize_mode=PIL_Image.Resampling.NEAREST,
) -> np.ndarray:
    assert not (subsample and resize_to), "Cannot subsample and resize at the same time"

    im = PIL_Image.open(image_filename)
    ycbcr = im.convert("YCbCr")

    # either subsample or resize
    if subsample:
        yuv888 = np.ndarray(ycbcr.size[0] * ycbcr.size[1] * 3, "u1", ycbcr.tobytes())
        yuv888 = yuv888[::2, ::2]
        yuv888 = yuv888.reshape(ycbcr.size[0] // 2, ycbcr.size[1] // 2, 3)
    elif resize_to is not None:
        ycbcr = ycbcr.resize(resize_to, resample=resize_mode)
        yuv888 = np.ndarray(ycbcr.size[0] * ycbcr.size[1] * 3, "u1", ycbcr.tobytes())
        yuv888 = yuv888.reshape(ycbcr.size[0], ycbcr.size[1], 3)

    if rescale:
        yuv888 = yuv888 / 255.0

    return yuv888


def load_image_as_yuv888_y_only(
    image_filename,
    rescale=False,
    subsample=False,
    resize_to=None,
    resize_mode=PIL_Image.Resampling.NEAREST,
):
    yuv888 = load_image_as_yuv888(
        image_filename,
        rescale=rescale,
        subsample=subsample,
        resize_to=resize_to,
        resize_mode=resize_mode,
    )
    yuv888_y_only = yuv888[..., 0]

    return yuv888_y_only


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
