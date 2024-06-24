import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(helper_path)

from pathlib import Path
import argparse
from helper import get_file_from_server
from zipfile import ZipFile


def get_dataset(dataset_name, output="datasets"):
    full_path = Path(output) / dataset_name
    if not full_path.exists():
        if not full_path.with_suffix(".zip").exists():
            get_file_from_server(f"https://datasets.naoth.de/{dataset_name}.zip", full_path.with_suffix(".zip"))

        print("unpack")
        with ZipFile(str(full_path.with_suffix(".zip")), "r") as f:
            f.extractall("datasets")
    else:
        print("dataset already exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", required=True)
    parser.add_argument("-o", "--output", required=False)

    args = parser.parse_args()
    get_dataset(args.dataset, args.output)
