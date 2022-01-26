from pathlib import Path

from common_tools import get_data_root, download_function


def download_runswift_segmentation_dataset():
    # TODO move this to datasets.naoth.de
    data_root_path = get_data_root()
    data_root_path = Path(data_root_path).resolve() / "data_runswift" / "field_segmentation"

    download_function("https://logs.naoth.de/Experiments/RunswiftDatasets/Downloads.zip",
                      f"{data_root_path}/Downloads.zip")
    download_function("https://arxiv.org/pdf/2108.12809.pdf",
                      f"{data_root_path}/segmentation_dataset.pdf")


if __name__ == '__main__':
    download_runswift_segmentation_dataset()
