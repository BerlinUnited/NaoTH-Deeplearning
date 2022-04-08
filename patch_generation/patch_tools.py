"""

"""

import cppyy
import cppyy.ll
from cppyy_tools import *
from PatchStuff import PatchExecutor
from pathlib import Path


def get_coco_dataset_paths():
    """
    finds the parent folder of each image under the coco root. The set of all those parent folders will be used for
    input to cppyy
    For reference the coco folder structure:
    coco_1.0
        -> 4                            #  (those numbers are cvat project ids)
        -> 5
            -> images                   #  (this folder is created by the cvat download)
            -> annotations              #  (this folder is created by the cvat download)
            -> debug_images             #  (this folder is created by this script)
            --> all_patches             #  (this folder iis created by this script)
    """
    # FIXME use config here for base path
    coco_root = Path("/home/stella/Documents/datasets/data_cvat/RoboCup2019/combined/COCO_1.0/")
    all_images = coco_root.glob('**/images/**/*.png')  # don't accidentally include debug images
    all_parents = [f.parent for f in all_images]
    all_parents = list(set(all_parents))

    return all_parents


if __name__ == "__main__":
    sub_folders = get_coco_dataset_paths()
    evaluator = PatchExecutor()
    with cppyy.ll.signals_as_exception():
        evaluator.execute(sub_folders)
