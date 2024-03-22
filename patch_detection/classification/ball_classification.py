import pickle
import time

import keras


def make_classifier_mean(encoder, input_shape=(16, 16, 1)):
    encoder.trainable = False

    input_layer = keras.layers.Input(shape=input_shape)

    # TODO: Skip the sampling step and just grab the mean layer directly
    mean, var, z = encoder(input_layer)

    dense = keras.layers.Dense(64, activation="relu")(mean)
    dense = keras.layers.Dense(16, activation="relu")(dense)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(dense)

    classifier = keras.models.Model(input_layer, output_layer)

    classifier.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return classifier


def make_naoth_classifier():
    input_shape = (16, 16, 1)
    model = keras.models.Sequential()

    # we don't know the kernel size b-human used
    model.add(
        keras.layers.Convolution2D(
            16, (3, 3), input_shape=input_shape, padding="same", name="Conv2D_1"
        )
    )
    # Batch Norm
    model.add(keras.layers.ReLU(name="activation_1"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(keras.layers.Convolution2D(16, (3, 3), padding="same", name="Conv2D_2"))
    # Batch Norm
    model.add(keras.layers.ReLU(name="activation_2"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(keras.layers.Convolution2D(32, (3, 3), padding="same", name="Conv2D_3"))
    # Batch Norm
    model.add(keras.layers.ReLU(name="activation_3"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(keras.layers.Flatten(name="flatten_1"))
    model.add(keras.layers.Dense(32, activation="relu", name="dense_1"))
    model.add(keras.layers.Dense(64, activation="relu", name="dense_2"))
    model.add(keras.layers.Dense(16, activation="relu", name="dense_3"))
    model.add(keras.layers.Dense(1, activation="sigmoid", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def load_encoder(file_path):
    model = keras.models.load_model(file_path)
    return model.encoder


def train_classifier_early_stop(
    model_name,
    model,
    train_ds,
    output_path,
    validation_data=None,
    epochs=500,
    class_weights=None,
    timestamp_iso=None,
):

    if timestamp_iso is None:
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    save_dir = output_path / f"classifiers/{timestamp_iso}/"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(save_dir / f"{model_name}.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
    )
    reduce_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=5, verbose=0, mode="auto"
    )
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    callbacks = [save_callback, reduce_callback, early_stopping_callback]

    history = model.fit(
        x=train_ds,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    history_name = str(save_dir / f"{model_name}.history.pkl")
    pickle.dump(history, open(history_name, "wb"))
