import tensorflow as tf
import h5py
import numpy as np
import inquirer
from tensorflow.keras.utils import Sequence
from models import tiny_classifier_segmentation_output_v1, tiny_classifier_segmentation_output_v2, tiny_classifier_segmentation_output_v3

class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        with h5py.File(file_path, 'r') as f:
            self.num_samples = len(f['X'])

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            start_index = index * self.batch_size
            end_index = min((index + 1) * self.batch_size, self.num_samples)
            X_batch = f['X'][start_index:end_index] 
            y_batch = f['Y'][start_index:end_index]
        return X_batch, y_batch
    


def make_callbacks():
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10, min_lr=1e-7
        ),
    ]

    return callbacks


def train_v1():
    train_generator = DataGenerator('fy1500_segmentationdata_y.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_segmentationdata_y.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_segmentationdata_y.keras")
    print("Model saved to fy1500_segmentationdata_y.keras")


def train_v1_mean():
    train_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_segmentationdata_y_mean_subtracted.keras")
    print("Model saved to fy1500_segmentationdata_y_mean_subtracted.keras")


def train_v1_orig():
    train_generator = DataGenerator('fy1500_originaldata_y.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_originaldata_y.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_originaldata_y.keras")
    print("Model saved to fy1500_originaldata_y.keras")


def train_v1_orig_mean():
    train_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_originaldata_y_mean_subtracted.keras")
    print("Model saved to fy1500_originaldata_y_mean_subtracted.keras")


def train_v2():
    train_generator = DataGenerator('training_ds_y.h5', batch_size=32)
    validation_generator = DataGenerator('training_ds_y.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("tiny_classifier_segmentation_output_v2.keras")
    print("Model saved to tiny_classifier_segmentation_output_v2.keras")


def train_v3():
    train_generator = DataGenerator('fy1500_segmentationdata_y_meta_info.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_segmentationdata_y_meta_info.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v3()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_segmentationdata_y_meta_info.keras")
    print("Model saved to fy1500_segmentationdata_y_meta_info.keras")

def train_v3_mean():
    train_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted_meta_info.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted_meta_info.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v3()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_segmentationdata_y_mean_subtracted_meta_info.keras")
    print("Model saved to fy1500_segmentationdata_y_mean_subtracted_meta_info.keras")

def train_v3_original():
    train_generator = DataGenerator('fy1500_originaldata_y_meta_info.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_originaldata_y_meta_info.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v3()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_originaldata_y_meta_info.keras")
    print("Model saved to fy1500_originaldata_y_meta_info.keras")

def train_v3_original_mean():
    train_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted_meta_info.h5', batch_size=32)
    validation_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted_meta_info.h5', batch_size=32)
    
    model = tiny_classifier_segmentation_output_v3()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2000,
        callbacks=callbacks,
    )
    model.save("fy1500_originaldata_y_mean_subtracted_meta_info.keras")
    print("Model saved to fy1500_originaldata_y_mean_subtracted_meta_info.keras")


if __name__ == "__main__":
    functions = {name: obj for name, obj in globals().items() if callable(obj) and name.startswith('train_')}

    # Use inquirer to prompt the user to select a function
    questions = [
        inquirer.List('function',
                    message="Which function do you want to execute?",
                    choices=list(functions.keys()))
    ]

    answers = inquirer.prompt(questions)

    # Get the selected function name and execute the corresponding function
    selected_function = answers['function']
    functions[selected_function]()
