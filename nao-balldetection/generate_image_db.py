"""
    This scripts creates a pickle file containing all the images used in training. The generated file is also used
    during export with the devil_code_generator1

    Downloading from kaggle only works if $HOME/.kaggle/kaggle.json with a working API Token exists. You can sign up to
    kaggle and create your own: https://www.kaggle.com/docs/api

    NaoTH Members can use the team credentials found in the accounts wiki page
"""
import argparse
import os
import pickle
from pathlib import Path
import numpy as np

from loader import create_natural_dataset, calculate_mean, subtract_mean

DATA_DIR = Path(Path(__file__).parent.absolute() / "data").resolve()


def str2bool(v):
    print(repr(v), type(v))
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def store_output(output_file, mean, x, y, p=None):
    with open(output_file, "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(x, f)
        pickle.dump(y, f)
        pickle.dump(p, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the image database for training etc. '
                                                 'using a folder with 0, 1 etc. subfolders with png images.')
    parser.add_argument('-i', '--image-folder', dest='img_path', help='Path to the CSV file(s) with region annotation.')
    parser.add_argument("-l", "--limit-noball", type=str2bool, nargs='?', dest="limit_noball",
                        const=True, help="Randomly select at most |balls| from no balls class")
    parser.add_argument("--data_type", dest="data_type", choices=["classification"], default="classification")

    args = parser.parse_args()

    # set default values for resolution
    res = {"x": 32, "y": 32}

    x, y, p = create_natural_dataset(args.img_path, res, args.limit_noball, "classification")
    mean = calculate_mean(x)
    

    print("save classification dataset with natural images")
    output_name = str(DATA_DIR / f'{Path(args.img_path).name}.pkl')
    store_output(output_name, mean, x, y, p)

