from tensorflow.keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import L1L2, L2
from tensorflow.keras.models import Model


def build_classifier_cnn_ball_gopen24_functional():
    input_shape = (16, 16, 1)
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), padding="same", name="Conv2D_1")(inputs)
    x = ReLU(name="activation_1")(x)

    x = Conv2D(
        16,
        (5, 5),
        padding="valid",
        strides=(2, 2),
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_2",
    )(x)
    x = ReLU(name="activation_2")(x)

    x = Conv2D(
        16,
        (3, 3),
        padding="valid",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_3",
    )(x)
    x = ReLU(name="activation_3")(x)

    x = Conv2D(
        16,
        (3, 3),
        padding="valid",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_4",
    )(x)
    x = ReLU(name="activation_4")(x)

    x = Flatten(name="flatten_1")(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=L2(1e-4),
    )(x)
    x = Dropout(0.1)(x)

    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=L2(1e-4),
    )(x)

    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
