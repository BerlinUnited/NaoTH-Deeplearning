import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from pathlib import Path
import argparse
from helper import get_file_from_server
from zipfile import ZipFile
import shutil

def get_dataset(dataset_name):
    full_path = Path("datasets") / dataset_name
    if not full_path.exists():
        if not full_path.with_suffix(".zip").exists():
            remote_path = Path(os.environ["REPL_ROOT"]) / "datasets" / Path(dataset_name).with_suffix(".zip")
            get_file_from_server(f"https://datasets.naoth.de/{dataset_name}.zip", full_path.with_suffix(".zip"))
        
        print("unpack")                      
        with ZipFile(str(full_path.with_suffix(".zip")), 'r') as f:
            f.extractall("datasets")
        
        #shutil.unpack_archive(str(full_path.with_suffix(".zip")))
        # TODO download it as zip and unzip it
    else:
        print("dataset already exists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", required=True)

    args = parser.parse_args()
    get_dataset(args.dataset)