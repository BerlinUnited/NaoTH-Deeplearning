import pickle
from datetime import datetime
from inspect import isclass, isfunction
from pathlib import Path
from sys import exit

# TODO encode dataset into output model name
import tensorflow as tf

from tensorflow import keras as keras


from models import mbc_36ksm_finetuned_crop

DATA_DIR = Path(Path(__file__).parent.absolute() / "data").resolve()
batch_size = 256
epochs = 10000

@keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce

@keras.saving.register_keras_serializable(name="WeightedBinaryCrossentropy")
class WeightedBinaryCrossentropy:
    def __init__(
        self,
        label_smoothing=0.0,
        weights=[1.0, 1.0],
        axis=-1,
        name="weighted_binary_crossentropy",
        loss_fn=weighted_binary_crossentropy,
    ):
        """Initializes `WeightedBinaryCrossentropy` instance.

        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).

          TODO: Check if this might be helpful?
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.

          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          name: Name for the op. Defaults to 'weighted_binary_crossentropy'.
        """
        super().__init__()
        self.weights = weights  # tf.convert_to_tensor(weights)
        self.label_smoothing = label_smoothing
        self.name = name
        self.loss_fn = weighted_binary_crossentropy if loss_fn is None else loss_fn

    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        def _smooth_labels():
            return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

        return tf.reduce_mean(self.loss_fn(y_true, y_pred, self.weights), axis=-1)

    def get_config(self):
        config = {"name": self.name, "weights": self.weights, "loss_fn": self.loss_fn}

        return dict(list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def main(pkl_data_file, output_path):

    model = mbc_36ksm_finetuned_crop()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=WeightedBinaryCrossentropy(
            weights=[1.0, 10.0],
        ),
        #loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    data_file = str(DATA_DIR /pkl_data_file)
    with open(data_file, "rb") as f:
        pickle.load(f)  # skip mean
        x = pickle.load(f)  # x are all input images
        y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

    print(y[:10])
    y_one_hot = keras.utils.to_categorical(y, num_classes=2)
    print(y_one_hot[:10])
    #quit()
    """ 
        The save callback will overwrite the previous models if the new model is better then the last. Restarting the 
        training will always overwrite the models.
    """
    output_path= "./"
    filepath = Path(output_path) / (model.name + "_" + Path(data_file).stem + "_notlimited_brigth1.3.h5")
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(filepath), monitor='loss', verbose=1,
                                                       save_best_only=True, mode='max')

    log_path = Path(output_path) / "logs" / (
            model.name + "_" + str(datetime.now()).replace(" ", "_").replace(":", "-"))
    log_callback = keras.callbacks.TensorBoard(log_dir=log_path, profile_batch=0)

    callbacks = [save_callback, log_callback]

    # TODO prepare an extra validation set, that is consistent over multiple runs
    # history = model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
    # validation_data=(X_test, Y_test),callbacks=callbacks)

    history = model.fit(x, y_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)
    history_filename = "history_" + model.name + "_" + Path(data_file).stem + ".pkl"

    # save history in same folder as model
    history_filepath = Path(output_path) / history_filename
    with open(str(history_filepath), "wb") as f:
        pickle.dump(history.history, f)

    return history, history_filename


if __name__ == '__main__':
    main(pkl_data_file="BOTTOM_GO26_notlimited_brigth1.3.pkl", output_path="./")

