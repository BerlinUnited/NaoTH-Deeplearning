from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
)
from keras.regularizers import L1L2, L2
from keras.models import Model


def make_naoth_classifier_generic_functional(
    input_shape=(16, 16, 1),
    filters=(8, 8, 16, 16),
    n_dense=64,
    dropout=0.33,
    regularize=True,
    softmax=True,
):
    inputs = Input(shape=input_shape)

    # Conv-LReLU-Pool Block 1
    x = Conv2D(filters[0], (3, 3), padding="same", name="Conv2D_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = LeakyReLU(name="activation_1")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_1")(x)

    # Conv-LReLU-Pool Block 2
    x = Conv2D(
        filters[1],
        (3, 3),
        padding="same",
        name="Conv2D_2",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_2")(x)
    x = LeakyReLU(name="activation_2")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_2")(x)

    # Conv-LReLU-Pool Block 3
    x = Conv2D(
        filters[2],
        (3, 3),
        padding="same",
        name="Conv2D_3",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_3")(x)
    x = LeakyReLU(name="activation_3")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_3")(x)

    # Conv-LReLU 2x2
    x = Conv2D(
        filters[3],
        (2, 2),
        padding="valid",
        name="Conv2D_4",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_4")(x)
    x = LeakyReLU(name="activation_5")(x)

    # Flatten and Dense Layers
    x = Flatten(name="flatten_1")(x)
    x = Dense(
        n_dense,
        activation="leaky_relu",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)

    if dropout is not None and dropout > 0.0:
        x = Dropout(dropout)(x)

    if softmax:
        outputs = Dense(2, activation="softmax")(x)
    else:
        outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
