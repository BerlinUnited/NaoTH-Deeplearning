from minio import Minio
from os import environ
import psycopg2
from pathlib import Path
from label_studio_sdk import Client
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError


label_dict = {
    "ball": 0,
    "nao": 1,
    "penalty_mark": 2
}

def get_minio_client():
    mclient = Minio("minio.berlin-united.com",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
    )
    return mclient

def get_postgres_cursor():
    params = {
    "host": "pg.berlin-united.com",
    "port": 4000,
    "dbname": "logs",
    "user": "naoth",
    "password": environ.get('DB_PASS')
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