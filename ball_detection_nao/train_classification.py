import pickle
import toml
from datetime import datetime
from inspect import isclass, isfunction
from pathlib import Path
from sys import exit

# TODO encode dataset into output model name
import tensorflow as tf
from tensorflow import keras as keras

import utility_functions.classification_models as model_zoo


def load_model(cfg):
    if "model_name" in cfg.keys():
        method_to_call = getattr(model_zoo, cfg["model_name"])
        model = method_to_call()
        # TODO how to handle execptions of getattr?
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
        pickle.load(f)  # skip mean
        x = pickle.load(f)  # x are all input images
        y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

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

    callbacks = [save_callback, log_callback]

    # TODO prepare an extra validation set, that is consistent over multiple runs
    # history = model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
    # validation_data=(X_test, Y_test),callbacks=callbacks)
    # TODO set seed so that validation split is the same for all
    print("blablabla")
    history = model.fit(x, y, batch_size=cfg["batch_size"], epochs=cfg["epochs"], verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)
    history_filename = "history_" + model.name + "_" + Path(cfg["trainings_data"]).stem + ".pkl"

    # save history in same folder as model
    history_filepath = Path(cfg["output_path"]) / history_filename
    with open(str(history_filepath), "wb") as f:
        pickle.dump(history.history, f)

    return history, history_filename


if __name__ == '__main__':
    main("classification_2")

