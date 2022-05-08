from pathlib import Path
import h5py

from core.main import get_data_root, download_function


def generate_ball_lc_dataset():
    bhuman_root_path = Path(get_data_root()) / "data_balldetection/bhuman"

    # original server is https://sibylle.informatik.uni-bremen.de/public/datasets/balldetector_lc/
    download_function("https://sibylle.informatik.uni-bremen.de/public/datasets/balldetector_lc/dataset-v1.hdf5",
                      f"{bhuman_root_path}/dataset-v1.hdf5")

    # TODO figure out how to read those images
    f = h5py.File(f'{bhuman_root_path}/dataset-v1.hdf5', 'r')
    print("Keys: %s" % f.keys())
    print(f["DasKaenguru_GW2B1820"]["labels"])
    print(f["DasKaenguru_GW2B1820"]["images"])

    labels = list(f["DasKaenguru_GW2B1820"]["labels"])
    images = list(f["DasKaenguru_GW2B1820"]["images"])
    print(len(labels))
    print(len(images))
    print(images[0])
    for i in images:
        print(i.shape)
    print(images[0].shape, (images[0].shape[0] / len(images)) )
    print(type(images[0]))


if __name__ == '__main__':
    generate_ball_lc_dataset()
