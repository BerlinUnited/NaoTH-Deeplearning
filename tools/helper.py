from minio import Minio
from os import environ
import psycopg2
from label_studio_sdk import Client

label_dict = {
    "ball": 0,
    "nao": 1,
    "penalty_mark": 2
}

def get_minio_client():
    mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
    )
    return mclient

def get_postgres_cursor():
    params = {
    "host": "pg.berlinunited-cloud.de",
    "port": 4000,
    "dbname": "logs",
    "user": "naoth",
    "password": environ.get('DB_PASS')
    }

    conn = psycopg2.connect(**params)
    return conn.cursor()

def get_labelstudio_client():
    LABEL_STUDIO_URL = 'https://ls.berlinunited-cloud.de/'
    API_KEY = '6cb437fb6daf7deb1694670a6f00120112535687'

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    return ls