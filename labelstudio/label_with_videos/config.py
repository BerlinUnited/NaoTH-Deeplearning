import os
from label_studio_sdk import LabelStudio

LS_URL = "https://labelstudio-api.berlin-united.com"
LS_TOKEN = os.getenv("LABELSTUDIO_API_KEY", "DEIN_LS1_TOKEN")
LS_TARGET_PROJECT_ID = 4
LS_FROM_NAME = "label"
LS_TO_NAME = "image"
DOWNLOAD_DIR = "temp_videos"

PROJECT_IDS = [
    7694, 7695, 7696, 7697, 7698, 7699, 7700, 7701, 7702, 7703,
    7704, 7705, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713,
    7714, 7715, 7716, 7717,
    7686, 7687, 7688, 7689, 7690, 7691, 7692, 7693,
    7676, 7677, 7678, 7679, 7680, 7681, 7682, 7683, 7684, 7685,
]


def get_ls_client():
    ls = LabelStudio(base_url=LS_URL, api_key=LS_TOKEN)
    ls.check_connection()
    return ls


def get_headers():
    return {"Authorization": f"Token {LS_TOKEN}"}
