import argparse
import ast
import pickle
from pathlib import Path

import keras
from sklearn.model_selection import train_test_split

from patch_detection.autoencoder.vae import build_vae
from patch_detection.autoencoder.vaegan import build_vae_gan
from patch_detection.datasets import load_ds_patches_21_23, make_autoencoder_dataset


def parse_tuple(arg_value, value_type=int):
    """Helper function to parse a tuple from a string."""
    try:
        value = ast.literal_eval(arg_value)
        if isinstance(value, tuple) and all(isinstance(item, value_type) for item in value):
            return value
        else:
            raise argparse.ArgumentTypeError("Value must be a tuple of " + str(value_type))
    except:
        raise argparse.ArgumentTypeError("Value must be a valid tuple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model configuration parameters.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="test",
        help="Name of the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vae",
        choices=["vae", "vaegan", "wvaegan"],
        help="Model to train.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimensionality of the latent space.",
    )
    parser.add_argument(
        "--input_shape",
        type=lambda s: parse_tuple(s, int),
        default=(16, 16, 1),
        help="Input shape as a tuple (height, width, channels).",
    )
    parser.add_argument(
        "--output_shape",
        type=lambda s: parse_tuple(s, int),
        default=(16, 16, 1),
        help="Output shape as a tuple (height, width, channels).",
    )
    parser.add_argument(
        "--filters",
        type=lambda s: parse_tuple(s, int),
        default=(16, 16, 32, 32),
        help="Tuple of filters in each conv layer.",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=lambda s: parse_tuple(s, tuple),
        default=((3, 3), (3, 3), (3, 3), (3, 3)),
        help="Tuple of kernel sizes for each conv layer.",
    )
    parser.add_argument(
        "--strides",
        type=lambda s: parse_tuple(s, tuple),
        default=((1, 1), (2, 2), (2, 2), (2, 2)),
        help="Tuple of strides for each conv layer.",
    )
    parser.add_argument(
        "--paddings",
        type=str,
        nargs="+",
        default=["same", "same", "same", "same"],
        help="List of padding for each conv layer.",
    )
    parser.add_argument(
        "--activation_function",
        type=str,
        default="leaky_relu",
        choices=["relu", "leaky_relu", "sigmoid", "tanh"],
        help="Activation function for the conv layers.",
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Whether to use batch normalization.",
    )
    parser.add_argument(
        "--n_dense",
        type=int,
        help="Number of dense layers, if any.",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../models",
        help="Path to save the trained model.",
    )

    args = parser.parse_args()

    latent_dim = args.latent_dim
    input_shape = args.input_shape
    output_shape = args.output_shape
    filters = args.filters
    kernel_sizes = args.kernel_sizes
    strides = args.strides
    paddings = args.paddings
    use_batch_norm = args.use_batch_norm
    n_dense = args.n_dense
    batch_size = args.batch_size
    epochs = args.epochs
    output_path = Path(args.output_path)

    model_builder_map = {
        "vae": build_vae,
        "vaegan": build_vae_gan,
    }
    model_builder = model_builder_map[args.model]

    # Convert activation function from string to keras layer
    activation_function_map = {
        "relu": keras.layers.ReLU,
        "leaky_relu": keras.layers.LeakyReLU,
        "sigmoid": keras.layers.Activation("sigmoid"),
        "tanh": keras.layers.Activation("tanh"),
    }
    activation_function = activation_function_map[args.activation_function]

    # load dataset , rescale, augment training data with random flips and noise
    X = load_ds_patches_21_23()
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    X_train = make_autoencoder_dataset(
        X_train,
        rescale=True,
        augment=False,
        batch_size=batch_size,
    )
    X_val = make_autoencoder_dataset(
        X_val,
        rescale=True,
        augment=False,
        batch_size=batch_size,
    )

    # build model
    model = model_builder(
        latent_dim=latent_dim,
        input_shape=input_shape,
        output_shape=output_shape,
        filters=filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation_function=activation_function,
        use_batch_norm=use_batch_norm,
        n_dense=n_dense,
    )

    # count number of parameters
    n_params = model.count_params()

    model_name = f"{args.model}_{latent_dim}_{args.model_name}_{n_params}"

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / f"{model_name}.chkpnt.keras"),
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        validation_data=X_val,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
    )

    model.save(str(output_path / f"{model_name}.keras"))

    with open(str(output_path / f"{model_name}.history"), "wb") as f:
        pickle.dump(history.history, f)
