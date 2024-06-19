import tensorflow as tf


class CustomCNN(tf.keras.Model):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.LeakyReLU()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.LeakyReLU()
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.LeakyReLU()

        self.conv5 = tf.keras.layers.Conv2D(filters=40, kernel_size=3, padding="same")
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.LeakyReLU()

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=1, activation=None)  # Output size is 1, change it according to your task

    def call(self, inputs):
        x = self.pool1(self.relu1(self.bn1(self.conv1(inputs))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Instantiate the model
model = CustomCNN()
