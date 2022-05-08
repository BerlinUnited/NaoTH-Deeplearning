"""
    Adapted from https://github.com/bhuman/DeepFieldBoundary/blob/master/Dataset-Pipeline/downloader.py

    TODO: add some visualization here to see if the hdf5 works
    TODO: add upload to cvat here
"""

from pathlib import Path
import h5py
import os
import requests

from bs4 import BeautifulSoup
from core.main import get_data_root, download_function


def generate_field_detection_dataset():
    # TODO back this up on datasets.naoth.de
    bhuman_root_path = Path(get_data_root()) / "data_fielddetection/bhuman"
    root_url = "https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary/"

    # original server is https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary/
    download_function("https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary/fieldboundary.hdf5",
                      f"{bhuman_root_path}/fieldboundary.hdf5")
    download_function("https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary/labels-as-csv.zip",
                      f"{bhuman_root_path}/labels-as-csv.zip")
    download_function("https://sibylle.informatik.uni-bremen.de/public/datasets/fieldboundary/readme.txt",
                      f"{bhuman_root_path}/readme.txt")

    soup = BeautifulSoup(requests.get(f'{root_url}/Locations').text, features='html.parser')
    for link in soup.select("a[href$='.hdf5']"):
        link_ = link.get('href')
        if not os.path.isfile(os.path.join(str(bhuman_root_path), "Locations", link_)):
            download_function(f'{root_url}/Locations/{link_}', os.path.join(str(bhuman_root_path), "Locations", link_))


if __name__ == '__main__':
    generate_field_detection_dataset()
