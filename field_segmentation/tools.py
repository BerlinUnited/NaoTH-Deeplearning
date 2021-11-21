from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError


def download_function(origin, target):

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


def download_runswift_segmentation_dataset():
    data_root_path = "../data_segmentation"
    data_root_path = Path(data_root_path).resolve()

    download_function("https://logs.naoth.de/Experiments/RunswiftDatasets/Downloads.zip",
                        f"{data_root_path}/Downloads.zip")
    download_function("https://arxiv.org/pdf/2108.12809.pdf",
                        f"{data_root_path}/segmentation_dataset.pdf")


if __name__ == '__main__':
    download_runswift_segmentation_dataset()