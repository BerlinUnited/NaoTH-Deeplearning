import pickle
import toml
from datetime import datetime
from inspect import isclass, isfunction
from pathlib import Path
from sys import exit
import numpy as np

# TODO encode dataset into output model name
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.callbacks import ReduceLROnPlateau

import utility_functions.classification_models as model_zoo
import utility_functions.lr_schedules as lr


def load_model(cfg):
    if "model_name" in cfg.keys():
        method_to_call = getattr(model_zoo, cfg["model_name"])
        model = method_to_call()
        # TODO how to handle exceptions of getattr?
        return model
    else:
        print(
            "ERROR: No model specified, you have to specify model_name in the config")
        exit(1)


def main(config_name):
    with open('classification.toml', 'r') as f:
        config_dict = toml.load(f)

    cfg = config_dict[config_name]
    model = load_model(cfg)
    DATA_DIR = Path(cfg["data_root_path"]).resolve()

    data_file = str(DATA_DIR / cfg["trainings_data"])
    with open(data_file, "rb") as f:
        mean = pickle.load(f)
        mean_x = pickle.load(f)  # x are all input images
        y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

    # adjust the mean scaling
    if cfg["mean_subtraction"]:
        x = mean_x 
    else:
        # revert mean subtraction
        x = (mean_x + mean)    

    # generate validation set
    rng = np.random.default_rng(42)
    a = rng.choice(np.arange(len(x)), int(len(x) * 0.2), replace=False)
    val_x = x[a]
    val_y = y[a]

    """ 
        The save callback will overwrite the previous models if the new model is better then the last. Restarting the 
        training will always overwrite the models.
    """
    filepath = Path(cfg["output_path"]) / (model.name + "_" + Path(cfg["trainings_data"]).stem + ".h5")
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(filepath), monitor='loss', verbose=1,
                                                       save_best_only=True, mode='auto')

    log_path = Path(cfg["output_path"]) / "logs" / (
            model.name + "_" + str(datetime.now()).replace(" ", "_").replace(":", "-"))
    log_callback = keras.callbacks.TensorBoard(log_dir=log_path, profile_batch=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    callbacks = [save_callback, log_callback, reduce_lr]

    # TODO prepare an extra validation set, that is consistent over multiple runs
    # history = model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
    # validation_data=(X_test, Y_test),callbacks=callbacks)
    # TODO set seed so that validation split is the same for all
    # FIXME sometimes training gets stuck early on, then reducing the learning rate is not helpful
    # https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
    history = model.fit(x, y, batch_size=cfg["batch_size"], epochs=cfg["epochs"], verbose=1,
                        validation_data=(val_x, val_y), shuffle=True,
                        callbacks=callbacks)
    history_filename = "history_" + model.name + "_" + Path(cfg["trainings_data"]).stem + ".pkl"

    # save history in same folder as model
    history_filepath = Path(cfg["output_path"]) / history_filename
    with open(str(history_filepath), "wb") as f:
        pickle.dump(history.history, f)

    return history, history_filename


if __name__ == '__main__':
    # TODO do evaluations on fixed validation set again
    #main("classification_tk_natural")
    #main("classification_tk_synthetic")
    #main("classification_tk_combined")
    main("classification_tk_combined-balanced")

    # TODO patch size evaluation on more epochs
    #main("classification_1")
    #main("classification_2")
