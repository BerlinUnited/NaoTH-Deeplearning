import tensorflow as tf


def bhuman_segmentation_y_channel_120x160x1_sigmoid():
    inputs = tf.keras.Input(shape=(120, 160, 1))

    x = tf.keras.layers.Convolution2D(16, (5, 5), padding="same", name="Conv2D_1")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x1 = tf.keras.layers.ReLU(name="activation_1")(x)
    x = tf.keras.layers.SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x1)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU(name="activation_2")(x)
    x = tf.keras.layers.SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU(name="activation_3")(x)
    x2 = tf.keras.layers.Add()([x1, x])
    x = tf.keras.layers.SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x2)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU(name="activation_4")(x)
    x = tf.keras.layers.SeparableConv2D(
        filters=16, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.ReLU(name="activation_5")(x)
    x3 = tf.keras.layers.Add()([x2, x])
    x3l = tf.keras.layers.Convolution2D(24, (1, 1), strides=(2, 2), padding="valid", name="Conv2D_2")(x3)
    x3l = tf.keras.layers.BatchNormalization(axis=3)(x3l)

    x3r = tf.keras.layers.SeparableConv2D(
        filters=24, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="linear"
    )(x3)
    x3r = tf.keras.layers.BatchNormalization(axis=3)(x3r)
    x3r = tf.keras.layers.ReLU(name="activation_6")(x3r)
    x3r = tf.keras.layers.SeparableConv2D(
        filters=24, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x3r)
    x3r = tf.keras.layers.BatchNormalization(axis=3)(x3r)
    x = tf.keras.layers.Add()([x3r, x3l])
    x4 = tf.keras.layers.ReLU(name="activation_7")(x)
    x4r = tf.keras.layers.SeparableConv2D(
        filters=24, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x4)
    x4r = tf.keras.layers.BatchNormalization(axis=3)(x4r)
    x4r = tf.keras.layers.ReLU(name="activation_8")(x4r)
    x4r = tf.keras.layers.SeparableConv2D(
        filters=24, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x4r)
    x4r = tf.keras.layers.BatchNormalization(axis=3)(x4r)
    x4r = tf.keras.layers.ReLU(name="activation_9")(x4r)
    x5 = tf.keras.layers.Add()([x4, x4r])
    x5r = tf.keras.layers.SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="linear"
    )(x5)
    x5r = tf.keras.layers.BatchNormalization(axis=3)(x5r)
    x5r = tf.keras.layers.ReLU(name="activation_10")(x5r)
    x5r = tf.keras.layers.SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x5r)
    x5r = tf.keras.layers.BatchNormalization(axis=3)(x5r)
    x5l = tf.keras.layers.Convolution2D(32, (1, 1), strides=(2, 2), padding="valid", name="Conv2D_3")(x5)
    x5l = tf.keras.layers.BatchNormalization(axis=3)(x5l)

    x6 = tf.keras.layers.Add()([x5l, x5r])
    x6 = tf.keras.layers.ReLU(name="activation_11")(x6)

    x6r = tf.keras.layers.SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x6)
    x6r = tf.keras.layers.BatchNormalization(axis=3)(x6r)
    x6r = tf.keras.layers.ReLU(name="activation_12")(x6r)
    x6r = tf.keras.layers.SeparableConv2D(
        filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="linear"
    )(x6r)
    x6r = tf.keras.layers.BatchNormalization(axis=3)(x6r)
    x6r = tf.keras.layers.ReLU(name="activation_13")(x6r)

    x7 = tf.keras.layers.Add()([x6, x6r])
    # x7 = tf.keras.layers.Convolution2D(4, (3, 3), strides=(1, 1), padding="same", name="Conv2D_4")(x7)
    x7 = tf.keras.layers.Convolution2D(
        3, (3, 3), strides=(1, 1), padding="same", name="Conv2D_4", activation="sigmoid"
    )(x7)

    classifier = tf.keras.Model(inputs=inputs, outputs=x7, name="bhuman-bop")
    return classifier
