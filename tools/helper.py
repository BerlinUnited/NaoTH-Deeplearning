import pickle
import zipfile
from enum import Enum
from os import environ
from pathlib import Path
import requests
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

#import cv2
#import h5py
import numpy as np
#import psycopg2
#from label_studio_sdk import Client
#from minio import Minio
#from PIL import Image as PIL_Image
#from sklearn.model_selection import train_test_split

label_dict = {"ball": 0, "nao": 1, "penalty_mark": 2, "referee": 3}


class PatchType(Enum):
    LEGACY = "patches"
    SEGMENTATION = "patches_segmentation"


class ColorMode(Enum):
    RGB = "RGB"  # debug
    YUV422_CV2 = "YUV422_CV2"
    YUV422_PIL = "YUV422_PIL"
    YUV422_Y_ONLY_CV2 = "YUV422_Y_ONLY_CV2"
    YUV422_Y_ONLY_PIL = "YUV422_Y_ONLY_PIL"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


def get_alive_fileserver(timeout=2):
    url = "https://logs.berlin-united.com/"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        print(f"Server {url} is alive.")
        return url
    except requests.exceptions.RequestException as e:
        print(e)
        url = "https://logs.naoth.de/"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Check for HTTP errors
            print(f"Server {url} is alive.")
            return url
        except requests.exceptions.RequestException as e:
            print(e)
            print("No fileserver is reachable")
            quit()


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


def download_from_minio(client, bucket_name, filename, output_folder, overwrite=False):
    output = Path(output_folder) / str(bucket_name + "_" + filename)

    if overwrite or not output.exists():
        client.fget_object(bucket_name, filename, output)
    else:
        print(f"File {output} already exists. Skipping download...")

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
    return cv2.Laplacian(image, cv2.CV_64F).var()


def resize_image_cv2_inter_nearest(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def combine_datasets_split_train_val(Xs, ys, test_size=0.15, stratify=True):
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
            stratify=y if stratify else None,
        )

        Xs_train.append(X_train)
        Xs_val.append(X_val)
        ys_train.append(y_train)
        ys_val.append(y_val)

    X_train = np.concatenate(Xs_train)
    y_train = np.concatenate(ys_train)

    X_val = np.concatenate(Xs_val)
    y_val = np.concatenate(ys_val)

    return X_train, X_val, y_train, y_val


def download_devils_labeled_patches(
    save_dir,
    url="https://datasets.naoth.de/NaoDevils_Patches_GO24_32x32x3_nsamples_%20215820/patches_classification_naodevils_32x32x3_GO24.zip",
    overwrite=False,
):
    save_dir = Path(save_dir)
    filename = "devils_dataset.zip"
    filepath = save_dir / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not filepath.exists():
        print(f"Downloading dataset from {url}...")
        get_file_from_server(url, filepath)

        print(f"Extracting dataset to {save_dir}...")
        with zipfile.ZipFile(filepath, "r") as f_zip:
            f_zip.extractall(save_dir)


def download_tk_03_dataset(save_dir, url="https://datasets.naoth.de/tk03_combined_detection.pkl", overwrite=False):
    save_dir = Path(save_dir)
    filename = url.rsplit("/", 1)[-1]
    filepath = save_dir / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not filepath.exists():
        print(f"Downloading dataset from {url}...")
        get_file_from_server(url, filepath)

    # no need to extract, tk03 datasets are pickle files


def get_ball_radius_center_data_tk_03_combined(file_path, balls_only=True, debug=False):
    with open(file_path, "rb") as f:
        mean = pickle.load(f)
        images = pickle.load(f)
        targets = pickle.load(f)  # radius, x_coord, y_coord, is_ball
        # filepaths = pickle.load(f) # not needed

    is_ball = [target[3] for target in targets]

    # load targets in same format as naoth data
    # radius, x_coord, y_coord
    targets = np.array([[target[0], target[1], target[2]] for target in targets])

    # add mean back to images for consistent data format
    images_norm = images + mean

    # scale images to [0, 255] for consistent data format
    images_norm = (images_norm * 255).astype(np.uint8)

    if debug:
        print("Images shape: ", images.shape)
        print("Images mean: ", mean)
        print("Images range before normalization: ", images.min(), images.max())
        print()
        print("Images range after normalization: ", images_norm.min(), images_norm.max())
        print("Images mean after normalization: ", images_norm.mean())

    if balls_only:
        print("Filtering out non-ball patches...")
        images_norm = [img for img, is_ball in zip(images_norm, is_ball) if is_ball]
        targets = [target for target, is_ball in zip(targets, is_ball) if is_ball]

    return images_norm, targets


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


def download_and_extract_patches_from_bucket(data, save_dir, filename, overwrite=False):
    mclient = get_minio_client()

    for bucket_name in sorted(data):
        bucket_name = bucket_name[0]
        try:
            file_path = download_from_minio(
                client=mclient,
                bucket_name=bucket_name,
                filename=filename,
                output_folder=save_dir,
                overwrite=overwrite,
            )

            # TODO: Only extract if file was downloaded without breaking existing code
            print(f"Extracting {file_path}")
            with zipfile.ZipFile(file_path, "r") as f:
                f.extractall(save_dir / bucket_name)

        except Exception as e:
            print(f"Error downloading from bucket {bucket_name}: {e}")


def download_naoth_labeled_patches(
    save_dir,
    patch_type: PatchType = PatchType.LEGACY,
    validated=True,
    filter_top=None,
    filter_bottom=None,
    border=0,
    overwrite=False,
):
    # prevent mutable default arguments
    filter_top = filter_top or ""
    filter_bottom = filter_bottom or ""

    filename = f"{patch_type.value}_border{border}.zip"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_top = get_patch_buckets("top", validated, filter_top)
    data_bottom = get_patch_buckets("bottom", validated, filter_bottom)

    download_and_extract_patches_from_bucket(
        data=data_top, save_dir=save_dir / "top", filename=filename, overwrite=overwrite
    )
    download_and_extract_patches_from_bucket(
        data=data_bottom, save_dir=save_dir / "bottom", filename=filename, overwrite=overwrite
    )


def get_classification_data_devils_combined(
    file_path,
    color_mode: ColorMode,
):
    X_top, y_top = get_classification_data_devils_top(
        file_path=file_path,
        color_mode=color_mode,
    )
    X_bottom, y_bottom = get_classification_data_devils_bottom(
        file_path=file_path,
        color_mode=color_mode,
    )

    X = X_top + X_bottom  # patches may be inconsistent in shape at this point
    y = y_top + y_bottom

    return X, y


def get_classification_data_devils_top(
    file_path,
    color_mode: ColorMode,
):
    devils_patches = Path(file_path)

    devils_balls_top = list(devils_patches.rglob("*/1.00/*upper*.png"))
    devils_other_top = list(devils_patches.rglob("*/0.00/*upper*.png"))

    image_paths = devils_balls_top + devils_other_top

    X = load_images_from_paths(image_paths=image_paths, color_mode=color_mode)
    y = [1] * len(devils_balls_top) + [0] * len(devils_other_top)

    return X, y


def get_classification_data_devils_bottom(
    file_path,
    color_mode: ColorMode,
):
    devils_patches = Path(file_path)

    devils_balls_bottom = list(devils_patches.rglob("*/1.00/*lower*.png"))
    devils_other_bottom = list(devils_patches.rglob("*/0.00/*lower*.png"))

    image_paths = devils_balls_bottom + devils_other_bottom

    X = load_images_from_paths(image_paths=image_paths, color_mode=color_mode)
    y = [1] * len(devils_balls_bottom) + [0] * len(devils_other_bottom)

    return X, y


def get_classification_data_naoth_combined(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    X_top, y_top = get_classification_data_naoth_top(
        file_path=file_path,
        color_mode=color_mode,
        filter_ambiguous_balls=filter_ambiguous_balls,
    )
    X_bottom, y_bottom = get_classification_data_naoth_bottom(
        file_path=file_path,
        color_mode=color_mode,
        filter_ambiguous_balls=filter_ambiguous_balls,
    )

    X = X_top + X_bottom  # patches may be inconsistent in shape at this point
    y = y_top + y_bottom

    return X, y


def get_classification_data_naoth_top(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    from tools.image_loader import get_multiclass_from_meta

    naoth_patches = Path(file_path)
    image_paths = list(naoth_patches.rglob("top/*/*/*.png"))

    X, meta = load_naoth_images_with_meta(image_paths=image_paths, color_mode=color_mode)
    y = [get_multiclass_from_meta(m) for m in meta]

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches(X, y)

    y = [1 if target[0] == 1 else 0 for target in y]

    return X, y


def get_classification_data_naoth_bottom(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    # TODO: Add parameter to filter out blurry balls
):
    from tools.helper import filter_ambiguous_ball_patches
    from tools.image_loader import get_multiclass_from_meta

    naoth_patches = Path(file_path)
    image_paths = list(naoth_patches.rglob("bottom/*/*/*.png"))

    X, meta = load_naoth_images_with_meta(image_paths=image_paths, color_mode=color_mode)
    y = np.array([get_multiclass_from_meta(m) for m in meta])

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches(X, y)

    y = [1 if target[0] == 1 else 0 for target in y]

    return X, y


def get_ball_radius_center_data_naoth_combined(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    balls_only=True,
    # TODO: Add parameter to filter out blurry balls
):
    X_top, y_top = get_ball_radius_center_data_naoth_top(
        file_path=file_path,
        color_mode=color_mode,
        filter_ambiguous_balls=filter_ambiguous_balls,
        balls_only=balls_only,
    )
    X_bottom, y_bottom = get_ball_radius_center_data_naoth_bottom(
        file_path=file_path,
        color_mode=color_mode,
        filter_ambiguous_balls=filter_ambiguous_balls,
        balls_only=balls_only,
    )

    X = X_top + X_bottom  # patches may be inconsistent in shape at this point
    y = y_top + y_bottom

    return X, y


def get_ball_radius_center_data_naoth_top(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    balls_only=True,
    # TODO: Add parameter to filter out blurry balls
):
    from tools.image_loader import get_ball_radius_center_from_meta, get_multiclass_from_meta

    naoth_patches = Path(file_path)
    image_paths = list(naoth_patches.rglob("top/*/*/*.png"))

    X, meta = load_naoth_images_with_meta(image_paths=image_paths, color_mode=color_mode)
    y = [get_ball_radius_center_from_meta(m) for m in meta]
    y_multiclass = [get_multiclass_from_meta(m) for m in meta]

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches_extra_target(X, y, y_multiclass)

    if balls_only:
        X, y = filter_ball_patches_extra_target(X, y, y_multiclass)

    return X, y


def get_ball_radius_center_data_naoth_bottom(
    file_path,
    color_mode: ColorMode,
    filter_ambiguous_balls=True,
    balls_only=True,
    # TODO: Add parameter to filter out blurry balls
):
    from tools.image_loader import get_ball_radius_center_from_meta, get_multiclass_from_meta

    naoth_patches = Path(file_path)
    image_paths = list(naoth_patches.rglob("bottom/*/*/*.png"))

    X, meta = load_naoth_images_with_meta(image_paths=image_paths, color_mode=color_mode)
    y = [get_ball_radius_center_from_meta(m) for m in meta]
    y_multiclass = [get_multiclass_from_meta(m) for m in meta]

    if filter_ambiguous_balls:
        X, y = filter_ambiguous_ball_patches_extra_target(X, y, y_multiclass)

    if balls_only:
        X, y = filter_ball_patches_extra_target(X, y, y_multiclass)

    return X, y


def load_naoth_images_with_meta(image_paths, color_mode: ColorMode):
    from tools.image_loader import get_meta_from_png

    X = load_images_from_paths(image_paths=image_paths, color_mode=color_mode)
    meta = [get_meta_from_png(img_path) for img_path in image_paths]

    return X, meta


def filter_ambiguous_ball_patches(X, y_multiclass):
    # only keep ball patches that do not contain robot class as well
    X_new = []
    y_new = []

    for img, target in zip(X, y_multiclass):
        # target is [ball, penalty, robot]
        if target[0] == 1 and target[2] == 1:
            continue
        else:
            X_new.append(img)
            y_new.append(target)

    return X_new, y_new


def filter_ambiguous_ball_patches_extra_target(X, y, y_multiclass):
    # only keep ball patches that do not contain robot class as well
    X_new = []
    y_new = []

    for img, target, target_multiclass in zip(X, y, y_multiclass):
        # target_multiclass is [ball, penalty, robot]
        if target_multiclass[0] == 1 and target_multiclass[2] == 1:
            continue

        X_new.append(img)
        y_new.append(target)

    return X_new, y_new


def filter_ball_patches_extra_target(X, y, y_multiclass):
    # only keep ball patches
    X_new = []
    y_new = []

    for img, target, class_ in zip(X, y, y_multiclass):
        if class_[0] != 1:
            continue
        else:
            X_new.append(img)
            y_new.append(target)

    return X_new, y_new


def load_images_from_paths(image_paths, color_mode) -> list:
    from tools.image_loader import (
        load_image_as_yuv422_cv2,
        load_image_as_yuv422_pil,
        load_image_as_yuv422_y_only_cv2,
        load_image_as_yuv422_y_only_pil,
    )

    color_mode_func_map = {
        ColorMode.RGB: PIL_Image.open,
        ColorMode.YUV422_CV2: load_image_as_yuv422_cv2,
        ColorMode.YUV422_Y_ONLY_CV2: load_image_as_yuv422_y_only_cv2,
        ColorMode.YUV422_PIL: load_image_as_yuv422_pil,
        ColorMode.YUV422_Y_ONLY_PIL: load_image_as_yuv422_y_only_pil,
    }

    load_image_func = color_mode_func_map.get(color_mode)

    if load_image_func is None:
        raise NotImplementedError(f"Color mode {color_mode} not implemented")

    # patches may not have a consistent shape at this point,
    # so we can not use np.array here
    return [load_image_func(str(img_path)) for img_path in image_paths]
