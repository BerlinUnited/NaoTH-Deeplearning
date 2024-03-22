"""
    example of getting all the patch data and training autoencoder on it
"""

import os
import shutil
import time
from pathlib import Path

import psycopg2
from minio import Minio

if not os.environ.get("LOG_DB_PASSWORD") or not os.environ.get("MINIO_SECRET_KEY"):
    raise ValueError(
        "Please set LOG_DB_PASSWORD and MINIO_SECRET_KEY environment variables!"
    )

params = {
    "host": os.environ.get("LOG_DB_HOST") or "pg.berlinunited-cloud.de",
    "port": os.environ.get("LOG_DB_PORT") or 4000,
    "dbname": os.environ.get("LOG_DB_NAME") or "logs",
    "user": os.environ.get("LOG_DB_USER") or "naoth",
    "password": os.environ.get("LOG_DB_PASSWORD"),
}
conn = psycopg2.connect(**params)
cur = conn.cursor()

mclient = Minio(
    os.environ.get("MINIO_HOST") or "minio.berlinunited-cloud.de",
    access_key=os.environ.get("MINIO_ACCESS_KEY") or "naoth",
    secret_key=os.environ.get("MINIO_SECRET_KEY"),
)


def get_patches_bottom():
    select_statement = f"""
    SELECT bucket_bottom_patches FROM robot_logs WHERE bucket_bottom_patches IS NOT NULL 
    """
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs


def get_patches_top():
    select_statement = f"""
    SELECT bucket_top_patches FROM robot_logs WHERE bucket_top_patches IS NOT NULL 
    """
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs


def download_patches():
    data = get_patches_top() + get_patches_bottom()

    timestamp = time.strftime("%Y-%m-%dT%H:%M")

    output_root = Path(f"../data/patches_{timestamp}")
    output_root.mkdir(exist_ok=True)

    print(f"Downloading patches from {len(data)} buckets...")

    for bucketname in data:
        bucketname = bucketname[0]
        objects = mclient.list_objects(bucketname)
        output_folder = output_root / bucketname

        print(f"Downloading {len(list(objects))} ZIP archive(s) from {bucketname}...")

        for obj in objects:
            # Download zip files from patch buckets to local working directory
            mclient.fget_object(bucketname, obj.object_name, obj.object_name)
            shutil.unpack_archive(filename=obj.object_name, extract_dir=output_folder)
            Path("patches.zip").unlink()


if __name__ == "__main__":
    download_patches()
