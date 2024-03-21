"""
    example of getting all the patch data and training autoencoder on it
"""
import psycopg2

from keras import layers
from keras.datasets import mnist
from keras.models import Model
from minio import Minio
import shutil
from pathlib import Path

params = {
    "host": "pg.berlinunited-cloud.de",
    "port": 4000,
    "dbname": "logs",
    "user": "naoth",
    "password": "fsdjhwzuertuqg",
}
conn = psycopg2.connect(**params)
cur = conn.cursor()

mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
)

def get_patches_bottom():
    select_statement = f"""
    SELECT bucket_bottom_patches FROM robot_logs WHERE bucket_bottom_patches IS NOT NULL 
    """
    cur.execute(select_statement)
    rtn_val = cur.fetchall()
    logs = [x for x in rtn_val]
    return logs

def download_patches():
    data = get_patches_bottom()
    output_root = Path("output_folder")
    output_root.mkdir(exist_ok=True) 
    for bucketname in data:
        bucketname = bucketname[0]
        print(bucketname)

        objects = mclient.list_objects(bucketname)
        output_folder = output_root / bucketname
        for count, obj in enumerate(objects):
            # Download zip files from patch buckets to local working directory
            mclient.fget_object(bucketname, obj.object_name, obj.object_name)
            shutil.unpack_archive(filename=obj.object_name, extract_dir=output_folder)
            Path("patches.zip").unlink()


def model():
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()
    
    return autoencoder

if __name__ == "__main__":
    download_patches()    
