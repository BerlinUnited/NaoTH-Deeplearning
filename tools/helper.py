import shutil
import time
import zipfile
from dataclasses import dataclass
from enum import Enum
from os import environ
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import h5py
import numpy as np
import psycopg2
from label_studio_sdk import Client
from minio import Minio
from PIL import Image as PIL_Image
from sklearn.model_selection import train_test_split

label_dict = {"ball": 0, "nao": 1, "penalty_mark": 2, "referee": 3}


class PatchType(Enum):
    LEGACY = "patches"
    SEGMENTATION = "patches_segmentation"


class ColorMode(Enum):
    RGB = "RGB"
    YUV422 = "YUV422"
    YUV422_Y_ONLY = "YUV422_Y_ONLY"
    YUV888 = "YUV888"
    YUV888_Y_ONLY = "YUV888_Y_ONLY"


def get_minio_client():
    mclient = Minio(
        "minio.berlin-united.com",
        access_key="naoth",
        secret_key="HAkPYLnAvydQA",
    )
    return mclient


def get_postgres_cursor():
    """
    connects to logs database
    returns the cursor
    """
    if "KUBERNETES_SERVICE_HOST" in environ:
        postgres_host = "postgres-postgresql.postgres.svc.cluster.local"
        postgres_port = 5432
    else:
        postgres_host = "pg.berlin-united.com"
        postgres_port = 4000

    params = {
        "host": postgres_host,
        "port": postgres_port,
        "dbname": "logs",
        "user": "naoth",
        "password": environ.get("DB_PASS"),
        "connect_timeout": 10,
    }

    conn = psycopg2.connect(**params)
    return conn.cursor()


def get_labelstudio_client():
    LABEL_STUDIO_URL = "https://ls.berlin-united.com/"
    API_KEY = "6cb437fb6daf7deb1694670a6f00120112535687"

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    return ls


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


def download_from_minio(client, bucket_name, filename, output_folder):
    output = Path(output_folder) / str(bucket_name + "_" + filename)
    if not output.exists():
        client.fget_object(bucket_name, filename, output)
    return str(output)


def create_h5_file(file_path, key_list, shape):
    # FIXME this does not work with append yet but I think we should make it work eventually
    with h5py.File(file_path, "w") as f:
        for key in key_list:
            f.create_dataset(
                key,
                data=np.empty([1, shape[0], shape[1], shape[2]]),
                compression="gzip",
                chunks=True,
                maxshape=(None, shape[0], shape[1], shape[2]),
            )


def append_h5_file(file_path, key, array):
    with h5py.File(file_path, "a") as f:
        f[key].resize((f[key].shape[0] + array.shape[0]), axis=0)
        np.concatenate((f[key], array), axis=0)

        # f[key][-array.shape[0]:] = array


def load_image_as_yuv422_original(image_filename, patch_size=16):
    """
    this functions loads an image from a file to the correct format for the naoth library
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2

    cv_img = cv2.imread(image_filename)
    cv_img = cv2.resize(
        cv_img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST
    )

    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(patch_size * patch_size * 2, np.uint8)

    for i in range(0, patch_size * patch_size, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0

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


def compute_blurrines_laplacian(image):
    import cv2

    return cv2.Laplacian(image, cv2.CV_64F).var()


def combine_datasets_split_train_val_stratify(Xs, ys, test_size=0.15):
    # Combine multiple datasets and split them into train and validation sets
    # with stratification before merging them into a single dataset.
    # This ensures that the class distribution is preserved in the train and
    # validation sets.

    Xs_train = []
    Xs_val = []
    ys_train = []
    ys_val = []

    for X, y in zip(Xs, ys):
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

        Xs_train.append(X_train)
        Xs_val.append(X_val)
        ys_train.append(y_train)
        ys_val.append(y_val)

    X_train = np.concatenate(Xs_train)
    y_train = np.concatenate(ys_train)

    X_val = np.concatenate(Xs_val)
    y_val = np.concatenate(ys_val)

    return X_train, y_train, X_val, y_val


def download_devils_labeled_patches(
    save_dir,
    url="https://datasets.naoth.de/NaoDevils_Patches_GO24_32x32x3_nsamples_%20215820/patches_classification_naodevils_32x32x3_GO24.zip",
):
    save_dir = Path(save_dir)
    filename = "devils_dataset.zip"
    filepath = save_dir / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        print(f"Downloading dataset from {url}...")
        get_file_from_server(url, filepath)

    print(f"Extracting dataset to {save_dir}...")
    with zipfile.ZipFile(filepath, "r") as f_zip:
        f_zip.extractall(save_dir)


def get_patch_buckets(camera: str, validated=True, filter_: str = ""):
    select_statement = f"""
    SELECT bucket_{camera}_patches FROM robot_logs WHERE bucket_{camera}_patches IS NOT NULL
    """

    if validated:
        select_statement += f" AND {camera}_validated=True"

    if filter_:
        select_statement += f" AND {filter_}"

    select_statement += ";"

    # print(select_statement)

    cur = get_postgres_cursor()
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    result = [x for x in rtn_val]
    return result


def download_and_extract_patches_from_bucket(data, save_dir, filename):
    mclient = get_minio_client()

    for bucket_name in sorted(data):
        bucket_name = bucket_name[0]
        try:
            file_path = download_from_minio(
                client=mclient,
                bucket_name=bucket_name,
                filename=filename,
                output_folder=save_dir,
            )

            print(f"Working on {file_path}")

            with zipfile.ZipFile(file_path, "r") as f:
                f.extractall(save_dir / bucket_name)

        except Exception as e:
            print(f"Error downloading from bucket {bucket_name}: {e}")

        finally:
            # check if zip file_path is defined and if the file exists, delete it
            if "file_path" in locals() and Path(file_path).exists():
                Path(file_path).unlink()


def download_naoth_labeled_patches(
    save_dir,
    patch_type: PatchType = PatchType.LEGACY,
    validated=True,
    filter_top=None,
    filter_bottom=None,
    border=0,
):
    # prevent mutable default arguments
    filter_top = filter_top or ""
    filter_bottom = filter_bottom or ""

    filename = f"{patch_type.value}_border{border}.zip"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_top = get_patch_buckets("top", validated, filter_top)
    data_bottom = get_patch_buckets("bottom", validated, filter_bottom)

    download_and_extract_patches_from_bucket(data_top, save_dir / "top", filename)
    download_and_extract_patches_from_bucket(data_bottom, save_dir / "bottom", filename)


def get_classification_data_devils_combined(
    file_path,
    patch_size: Tuple[int, int],
    color_mode: ColorMode,
):
    X_top, y_top = get_classification_data_devils_top(
        file_path=file_path,
        patch_size=patch_size,
        color_mode=color_mode,
    )
    X_bottom, y_bottom = get_classification_data_devils_bottom(
        file_path=file_path,
        patch_size=patch_size,
        color_mode=color_mode,
    )

    X = np.concatenate([X_top, X_bottom])
    y = np.concatenate([y_top, y_bottom])

    return X, y


def get_classification_data_devils_top(
    file_path,
    patch_size: Tuple[int, int],
    color_mode: ColorMode,
):
    devils_patches = Path(file_path)

    devils_balls_top = list(devils_patches.rglob("*/1.00/*upper*.png"))
    devils_other_top = list(devils_patches.rglob("*/0.00/*upper*.png"))

    image_paths = devils_balls_top + devils_other_top

    X = load_images_from_paths(
        image_paths=image_paths, patch_size=patch_size, color_mode=color_mode
    )
    y = np.concatenate(
        [np.ones(len(devils_balls_top)), np.zeros(len(devils_other_top))]
    )

    return X, y


def get_classification_data_devils_bottom(
    file_path,
    patch_size: Tuple[int, int],
    color_mode: ColorMode,
):
    devils_patches = Path(file_path)

    devils_balls_bottom = list(devils_patches.rglob("*/1.00/*lower*.png"))
    devils_other_bottom = list(devils_patches.rglob("*/0.00/*lower*.png"))

    image_paths = devils_balls_bottom + devils_other_bottom

    X = load_images_from_paths(
        image_paths=image_paths, patch_size=patch_size, color_mode=color_mode
    )
    y = np.concatenate(
        [np.ones(len(devils_balls_bottom)), np.zeros(len(devils_other_bottom))]
    )

    return X, y


def get_classification_data_naoth_combined(
    file_path,
    color_mode: ColorMode,
    patch_size: Tuple[int, int],
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    X_top, y_top = get_classification_data_naoth_top(
        file_path=file_path,
        color_mode=color_mode,
        patch_size=patch_size,
        filter_ambiguous_balls=filter_ambiguous_balls,
    )
    X_bottom, y_bottom = get_classification_data_naoth_bottom(
        file_path=file_path,
        color_mode=color_mode,
        patch_size=patch_size,
        filter_ambiguous_balls=filter_ambiguous_balls,
    )

    X = np.concatenate([X_top, X_bottom])
    y = np.concatenate([y_top, y_bottom])

    return X, y


def get_classification_data_naoth_top(
    file_path,
    color_mode: ColorMode,
    patch_size: Tuple[int, int],
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    naoth_patches = Path(file_path)
    image_paths = list(naoth_patches.rglob("top/*/*/*.png"))
    X = load_images_from_paths(
        image_paths=image_paths, patch_size=patch_size, color_mode=color_mode
    )
    meta = [get_meta_from_png(img_path) for img_path in image_paths]
    y = np.array([get_multiclass_from_meta(m) for m in meta])

    if filter_ambiguous_balls:
        # only keep ball patches that do not contain robot class as well
        X_new = []
        y_new = []

        for img, target in zip(X, y):
            # target is [ball, penalty, robot]
            if target[0] == 1 and target[2] == 1:
                continue
            else:
                X_new.append(img)
                y_new.append(target)

        X = np.array(X_new)
        y = np.array(y_new)

    # only use ball / no ball labels
    y = np.array([1 if target[0] == 1 else 0 for target in y])

    return X, y


def get_classification_data_naoth_bottom(
    file_path,
    color_mode: ColorMode,
    patch_size: Tuple[int, int],
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    naoth_patches = Path(file_path)

    image_paths = list(naoth_patches.rglob("bottom/*/*/*.png"))
    X = load_images_from_paths(
        image_paths=image_paths, patch_size=patch_size, color_mode=color_mode
    )
    meta = [get_meta_from_png(img_path) for img_path in image_paths]
    y = np.array([get_multiclass_from_meta(m) for m in meta])

    if filter_ambiguous_balls:
        # only keep ball patches that do not contain robot class as well
        X_new = []
        y_new = []

        for img, target in zip(X, y):
            # target is [ball, penalty, robot]
            if target[0] == 1 and target[2] == 1:
                continue
            else:
                X_new.append(img)
                y_new.append(target)

        X = np.array(X_new)
        y = np.array(y_new)

    y = np.array([1 if target[0] == 1 else 0 for target in y])

    return X, y


def load_images_from_paths(image_paths, patch_size, color_mode):
    if color_mode == ColorMode.RGB:
        return np.array(
            [
                np.array(
                    PIL_Image.open(str(img_path)).resize(
                        patch_size, resample=PIL_Image.Resampling.NEAREST
                    )
                )
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 3)

    elif color_mode == ColorMode.YUV888:
        return np.array(
            [
                load_image_as_yuv888(
                    str(img_path),
                    resize_to=patch_size,
                    resize_mode=PIL_Image.Resampling.NEAREST,
                )
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 3)

    elif color_mode == ColorMode.YUV888_Y_ONLY:
        return np.array(
            [
                load_image_as_yuv888_y_only(
                    str(img_path),
                    resize_to=patch_size,
                    resize_mode=PIL_Image.Resampling.NEAREST,
                )
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 1)

    else:
        raise NotImplementedError(f"Color mode {color_mode} not implemented")


@dataclass
class Point2D:
    x: float
    y: float

    def __getitem__(self, index):
        assert index in (0, 1), "Index must be 0 or 1"
        return self.x if index == 0 else self.y

    def as_cv2_point(self):
        return int(self.x), int(self.y)


@dataclass
class BoundingBox:
    # FIXME this should go into our naoth python package
    top_left: Point2D
    bottom_right: Point2D

    @classmethod
    def from_coords(cls, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        return cls(
            Point2D(top_left_x, top_left_y), Point2D(bottom_right_x, bottom_right_y)
        )

    @classmethod
    def from_xywh(cls, top_left_x, top_left_y, width, height):
        return cls(
            Point2D(top_left_x, top_left_y),
            Point2D(top_left_x + width, top_left_y + height),
        )

    @property
    def width(self):
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self):
        return self.bottom_right.y - self.top_left.y

    @property
    def area(self):
        return self.width * self.height

    @property
    def radius(self):
        width = round(self.width / 2)
        height = round(self.height / 2)

        # FIXME if the patch is on the image border it should be max
        return min(width, height)

    @property
    def center(self):
        """
        this will calculate the center of the rectangle in the coordinate frame the coordinates are in
        """
        # FIXME if the patch is on the image border it imagine that its a square based on the max

        x = round(self.top_left.x + self.width / 2)
        y = round(self.top_left.y + self.height / 2)

        return x, y

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """
        Calculates the intersection of this bounding box with another one.

        Returns:
            BoundingBox or None: A new BoundingBox representing the intersection,
            or None if there is no intersection.
        """
        intersect_top_left_x = max(self.top_left.x, other.top_left.x)
        intersect_top_left_y = max(self.top_left.y, other.top_left.y)
        intersect_bottom_right_x = min(self.bottom_right.x, other.bottom_right.x)
        intersect_bottom_right_y = min(self.bottom_right.y, other.bottom_right.y)

        # Check if the bounding boxes overlap
        if (
            intersect_top_left_x < intersect_bottom_right_x
            and intersect_top_left_y < intersect_bottom_right_y
        ):
            # If they do overlap, return a new BoundingBox object representing the intersection
            return BoundingBox.from_coords(
                intersect_top_left_x,
                intersect_top_left_y,
                intersect_bottom_right_x,
                intersect_bottom_right_y,
            )
        else:
            # If they don't overlap, return None
            return None
