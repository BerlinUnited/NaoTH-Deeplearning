import zipfile
from enum import Enum
from os import environ
from pathlib import Path
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import h5py
import numpy as np
import psycopg2
from image_loader import get_meta_from_png, get_multiclass_from_meta, load_image_as_yuv888, load_image_as_yuv888_y_only
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

    X = load_images_from_paths(image_paths=image_paths, patch_size=patch_size, color_mode=color_mode)
    y = np.concatenate([np.ones(len(devils_balls_top)), np.zeros(len(devils_other_top))])

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

    X = load_images_from_paths(image_paths=image_paths, patch_size=patch_size, color_mode=color_mode)
    y = np.concatenate([np.ones(len(devils_balls_bottom)), np.zeros(len(devils_other_bottom))])

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
    X = load_images_from_paths(image_paths=image_paths, patch_size=patch_size, color_mode=color_mode)
    meta = [get_meta_from_png(img_path) for img_path in image_paths]
    y = np.array([get_multiclass_from_meta(m) for m in meta])

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches(X, y)

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
    X = load_images_from_paths(image_paths=image_paths, patch_size=patch_size, color_mode=color_mode)
    meta = [get_meta_from_png(img_path) for img_path in image_paths]
    y = np.array([get_multiclass_from_meta(m) for m in meta])

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches(X, y)

    y = np.array([1 if target[0] == 1 else 0 for target in y])

    return X, y


def filter_ambiguous_ball_patches(X, y):
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

    return X, y


def load_images_from_paths(image_paths, patch_size, color_mode):
    if color_mode == ColorMode.RGB:
        return np.array(
            [
                np.array(PIL_Image.open(str(img_path)).resize(patch_size, resample=PIL_Image.Resampling.NEAREST))
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 3)

    elif color_mode == ColorMode.YUV888:
        return np.array(
            [
                load_image_as_yuv888(str(img_path), resize_to=patch_size, resize_mode=PIL_Image.Resampling.NEAREST)
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 3)

    elif color_mode == ColorMode.YUV888_Y_ONLY:
        return np.array(
            [
                load_image_as_yuv888_y_only(
                    str(img_path), resize_to=patch_size, resize_mode=PIL_Image.Resampling.NEAREST
                )
                for img_path in image_paths
            ]
        ).reshape(-1, *patch_size, 1)

    else:
        raise NotImplementedError(f"Color mode {color_mode} not implemented")
