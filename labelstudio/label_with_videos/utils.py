import os
import cv2
import numpy as np
import requests
from config import LS1_URL, get_ls1_headers


def get_value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_image_from_url(url):
    if url.startswith("/"):
        url = LS1_URL + url
    try:
        response = requests.get(url, headers=get_ls1_headers(), timeout=15)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def get_image_dimensions_from_url(url):
    img = get_image_from_url(url)
    if img is not None:
        return img.shape[1], img.shape[0]
    return None, None


def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    return obj
