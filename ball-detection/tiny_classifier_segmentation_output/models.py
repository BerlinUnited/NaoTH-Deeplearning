import tensorflow as tf


def build_classifier_cnn_ball_gopen24():
    input_shape = (16, 16, 1)
    classifier = tf.keras.models.Sequential()

    classifier.add(tf.keras.layers.Convolution2D(16, (5, 5), input_shape=input_shape, padding="same", name="Conv2D_1"))
    classifier.add(tf.keras.layers.ReLU(name="activation_1"))

    classifier.add(
        tf.keras.layers.Convolution2D(
            16,
            (5, 5),
            padding="valid",
            name="Conv2D_2",
            strides=(2, 2),
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )
    classifier.add(tf.keras.layers.ReLU(name="activation_2"))

    classifier.add(
        tf.keras.layers.Convolution2D(
            16,
            (3, 3),
            padding="valid",
            name="Conv2D_3",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )

    classifier.add(tf.keras.layers.ReLU(name="activation_3"))

    classifier.add(
        tf.keras.layers.Convolution2D(
            16,
            (3, 3),
            padding="valid",
            name="Conv2D_4",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )

    classifier.add(tf.keras.layers.ReLU(name="activation_4"))

    classifier.add(tf.keras.layers.Flatten(name="flatten_1"))

    classifier.add(
        tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4),
        )
    )
    classifier.add(tf.keras.layers.Dropout(0.1))
    classifier.add(
        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4),
        )
    )
    classifier.add(
        tf.keras.layers.Dense(2, activation="softmax"),
    )

    return classifier


def tiny_classifier_segmentation_output_v1(input_shape=(16, 16, 1)):
    """
    first I want to see how well we can actually classify already. So we don't add any new data
    """
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # First Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # Second Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fourth Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def tiny_classifier_segmentation_output_v2(input_shape=(16, 16, 1)):
    """
    first I want to see how well we can actually classify already. So we don't add any new data
    """
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # First Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # Second Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


def tiny_classifier_segmentation_output_v3(input_shape=(16, 16, 2)):
    """
    first I want to see how well we can actually classify already. So we don't add any new data
    """
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # First Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # Second Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fourth Convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"))
    # Leaky ReLU activation
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

    # MaxPooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    model = tiny_classifier_segmentation_output_v1()
    model.summary()
