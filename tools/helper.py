from minio import Minio
from os import environ
import psycopg2
from pathlib import Path
from label_studio_sdk import Client
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import numpy as np
import h5py
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image as PIL_Image

label_dict = {
    "ball": 0,
    "nao": 1,
    "penalty_mark": 2,
    "referee": 3
}

def get_minio_client():
    mclient = Minio("minio.berlin-united.com",
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
    }

    conn = psycopg2.connect(**params)
    return conn.cursor()

def get_labelstudio_client():
    LABEL_STUDIO_URL = 'https://ls.berlin-united.com/'
    API_KEY = '6cb437fb6daf7deb1694670a6f00120112535687'

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    return ls


def get_file_from_server(origin, target):
    def dl_progress(count, block_size, total_size):
        print('\r', 'Progress: {0:.2%}'.format(min((count * block_size) / total_size, 1.0)), sep='', end='', flush=True)

    if not Path(target).exists():
        target_folder = Path(target).parent
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        return

    error_msg = 'URL fetch failure on {} : {} -- {}'
    try:
        try:
            urlretrieve(origin, target, dl_progress)
            print('\nFinished')
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.reason))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if Path(target).exists():
            Path(target).unlink()
        raise


def download_from_minio(client, bucket_name, filename, output_folder):
    output = Path(output_folder) / "image" / str(bucket_name + "_" + filename)
    if not output.exists():
        client.fget_object(bucket_name, filename, output)
    return str(output)

def create_h5_file(file_path, key_list, shape):
    # FIXME this does not work with append yet but I think we should make it work eventually
    with h5py.File(file_path, "w") as f:
        for key in key_list:
            f.create_dataset(key, data=np.empty([1, shape[0],shape[1],shape[2]]), compression="gzip", chunks=True, maxshape=(None, shape[0],shape[1],shape[2]))
    
def append_h5_file(file_path, key, array):
    with h5py.File(file_path, "a") as f:
        f[key].resize((f[key].shape[0] + array.shape[0]), axis = 0)
        np.concatenate((f[key], array), axis=0)
        
        #f[key][-array.shape[0]:] = array


def load_image_as_yuv422(image_filename):
    """
    this functions loads an image from a file to the correct format for the naoth library
    # FIXME: i don't trust this function
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2
    #y = 240
    #x = 320
    cv_img = cv2.imread(image_filename)
    x = cv_img.shape[1]
    y = cv_img.shape[0]
    #cv_img = cv2.resize(
    #    cv_img, (240,320), interpolation=cv2.INTER_NEAREST
    #)
    #print(cv_img.shape, x,y)
    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(x * y * 2, np.uint8)

    for i in range(0, x * y, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0

    # TODO is this the correct order?
    image_yuv = yuv422.reshape(y,x, 2)
    return image_yuv

def load_image_as_yuv422_y_only(image_filename):
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
    image_y =  image_yuv[..., 0]
    image_y = image_y.reshape(y,x,1)
    print(image_y.shape)
    image_y = image_y[::2, ::2]  # half the resolution because semantic segmentation requires it
    return image_y

def load_image_as_yuv422_y_only_better(image_filename):
    im = PIL_Image.open(image_filename)
    ycbcr = im.convert('YCbCr')
    reversed_yuv888 = np.ndarray(480 * 640 * 3, 'u1', ycbcr.tobytes())
    full_image_y = reversed_yuv888[0::3]
    full_image_y = full_image_y.reshape(480,640,1)
    half_image_y = full_image_y[::2, ::2]
    half_image_y = half_image_y / 255.0
    return half_image_y

def load_image_as_yuv422_y_only_better_generic(image_filename):
    # FIXME make subsampling configurable
    im = PIL_Image.open(image_filename)
    ycbcr = im.convert('YCbCr')
    reversed_yuv888 = np.ndarray(480 * 640 * 3, 'u1', ycbcr.tobytes())
    full_image_y = reversed_yuv888[0::3]
    full_image_y = full_image_y.reshape(480,640,1)
    full_image_y = full_image_y / 255.0
    #half_image_y = full_image_y[::2, ::2]
    #half_image_y = half_image_y / 255.0
    return full_image_y


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
            Point2D(top_left_x, top_left_y), Point2D(top_left_x+width, top_left_y+height)
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