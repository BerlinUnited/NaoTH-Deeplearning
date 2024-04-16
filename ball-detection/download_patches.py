import psycopg2
from minio import Minio


from os import environ
from label_studio_sdk import Client



params = {
    "host": "pg.berlinunited-cloud.de",
    "port": 4000,
    "dbname": "logs",
    "user": "naoth",
    "password": environ.get('DB_PASS')
}
conn = psycopg2.connect(**params)
cur = conn.cursor()

mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key=environ.get("MINIO_PASS"),
)

LABEL_STUDIO_URL = 'https://ls.berlinunited-cloud.de/'
API_KEY = '6cb437fb6daf7deb1694670a6f00120112535687'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

def get_logs_with_top_images():
    select_statement = f"""
    SELECT bucket_top_patches FROM robot_logs WHERE bucket_top_patches IS NOT NULL AND top_validated IS TRUE AND robot_version = 'v6'
    """
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs


def get_logs_with_bottom_images():
    select_statement = f"""
    SELECT bucket_bottom_patches FROM robot_logs WHERE bucket_bottom_patches IS NOT NULL AND bottom_validated IS TRUE AND robot_version = 'v6'
    """
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs

if __name__ == "__main__":

    data_top = get_logs_with_top_images()
    data_bottom = get_logs_with_bottom_images()

    for bucket_name in data_top:
        mclient.fget_object(bucket_name[0], "patches.zip", bucket_name[0] +"_top.zip")
        break

    for bucket_name in data_bottom:
        mclient.fget_object(bucket_name[0], "patches.zip", bucket_name[0] +"_bottom.zip")
        break
        