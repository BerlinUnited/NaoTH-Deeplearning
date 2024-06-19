"""
    Each function corresponds to one experiment.
    - an experiment can create multiple models
    - an experiment should not change after creation
    - it should be clear what dataset was used
    - its best to use datasets that are actually already uploaded to datasets.naoth.de
"""

import toml
import tensorflow as tf
from pathlib import Path
import inspect
from utility_functions.model_zoo import fy_1500_new, fy_1500_new2, fy_1500_old, model1
from train import run_training

# HACK set gpu to invisible. For some reason tensorflow got stuck on my machine when using gpu
tf.config.set_visible_devices([], "GPU")


def get_data_root():
    # FIXME this is a copy
    with open("../config.toml", "r") as f:
        config_dict = toml.load(f)

    return config_dict["data_root"]


def run_detection_multiple_times():
    """
    This represents the training we have done up till 2019
    """
    dataset = Path(get_data_root()) / "data_balldetection/naoth/tk03_natural_detection.pkl"
    function_name = inspect.currentframe().f_code.co_name
    model_folder = (Path(get_data_root()) / "../Models" / function_name).resolve()

    model_functions = [fy_1500_new2, fy_1500_new, fy_1500_old, model1]

    train_params = {"batch_size": 256, "epochs": 20}
    for model_f in model_functions:
        for i in range(10):
            model = model_f()
            # TODO make it a format string
            model_output_name = "iter" + str(i) + "_" + model.name + "_" + Path(dataset).stem + ".h5"
            model_output_name = Path(model_folder) / model_output_name
            run_training(model=model, dataset=dataset, modelname=model_output_name, params=train_params)
        break


def run_classification_multiple_times():
    """
    this can be used to compare the classification results from the detection networks to those networks

    I expect the classification networks to be better while being smaller
    """
    dataset = Path(get_data_root()) / "data_balldetection/naoth/tk03_natural_classification.pkl"
    function_name = inspect.currentframe().f_code.co_name
    model_folder = (Path(get_data_root()) / "../Models" / function_name).resolve()

    model_functions = []  # TODO

    train_params = {"batch_size": 256, "epochs": 20}
    for model_f in model_functions:
        for i in range(10):
            model = model_f()
            # TODO make it a format string
            model_output_name = "iter" + str(i) + "_" + model.name + "_" + Path(dataset).stem + ".h5"
            model_output_name = Path(model_folder) / model_output_name
            run_training(model=model, dataset=dataset, modelname=model_output_name, params=train_params)
        break


run_detection_multiple_times()
