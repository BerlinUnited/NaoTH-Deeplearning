"""
    TODO figure out what representation of the labeled images i should use to get the real ball data in the best way. YOLO, cvat or coco
        TODO: next step for this is to export patches regardless of ball or no ball but the patches must be sampled to a specific size (e.g. 16x16)
"""

import cppyy
import cppyy.ll
from cppyy_tools import *
from PatchStuff import PatchExecutor
from pathlib import Path

def get_coco_dataset_paths():
    """
    finds the parent folder of each image under the coco root. The set of all those parent folders will be used for input to cppyy
    """
    coco_root = Path("/home/stella/Documents/datasets/data_cvat/RoboCup2019/combined/COCO_1.0/")
    all_images = coco_root.glob('**/images/**/*.png')  # dont accidently include debug images
    all_parents = [f.parent for f in all_images]
    all_parents = list(set(all_parents))

    return all_parents

if __name__ == "__main__":
    sub_folders = get_coco_dataset_paths()
    evaluator = PatchExecutor()
    with cppyy.ll.signals_as_exception():
        evaluator.execute(sub_folders)

        
