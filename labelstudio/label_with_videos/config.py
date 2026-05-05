import os
from label_studio_sdk import LabelStudio
import sys
import requests

LS1_URL = "https://labelstudio-api.berlin-united.com"
LS1_TOKEN = os.getenv("LABELSTUDIO_API_KEY", "DEIN_LS1_TOKEN")

LS2_URL = os.getenv("LS2_URL", "http://localhost:8080")
LS2_TOKEN = os.getenv("LS2_TOKEN", "34db6a8a93ad9446a946adc234577a3f2587dc96")

LS_TARGET_PROJECT_ID = 4
LS_FROM_NAME = "label"
LS_TO_NAME = "image"
DOWNLOAD_DIR = "temp_videos"

PROJECT_IDS = [
    7694,
    7695,
    7696,
    7697,
    7698,
    7699,
    7700,
    7701,
    7702,
    7703,
    7704,
    7705,
    7706,
    7707,
    7708,
    7709,
    7710,
    7711,
    7712,
    7713,
    7714,
    7715,
    7716,
    7717,
    7686,
    7687,
    7688,
    7689,
    7690,
    7691,
    7692,
    7693,
    7676,
    7677,
    7678,
    7679,
    7680,
    7681,
    7682,
    7683,
    7684,
    7685,
]


def check_connection(base_url, token, timeout=10):
    url = f"{base_url.rstrip('/')}/api/projects"
    headers = {"Authorization": f"Token {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        print(f"Connection to {base_url} was successful")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"Servererror {e}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error")
    except requests.exceptions.Timeout:
        print(f"Timout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        print(f"Unknown error {e}")

    sys.exit(-1)


def use_ls2():
    return bool(LS2_URL and LS2_TOKEN)


def get_ls_client(url, token):
    if check_connection(url, token):
        ls = LabelStudio(base_url=url, api_key=token)
        return ls
    sys.exit(-1)


def get_ls1_client():
    return get_ls_client(LS1_URL, LS1_TOKEN)


def get_ls2_client():
    return get_ls_client(LS2_URL, LS2_TOKEN)


def get_video_client():
    if use_ls2():
        return get_ls2_client()
    return get_ls1_client()


def get_video_url():
    if use_ls2():
        return LS2_URL
    return LS1_URL


def get_video_token():
    if use_ls2():
        return LS2_TOKEN
    return LS1_TOKEN


def get_video_headers():
    return {"Authorization": f"Token {get_video_token()}"}


def get_ls1_headers():
    return {"Authorization": f"Token {LS1_TOKEN}"}
