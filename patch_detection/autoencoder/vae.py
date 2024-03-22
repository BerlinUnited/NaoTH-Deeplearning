import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + K.exp(log_variance / 2) * epsilon

        return random_sample


@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

    def compile(self, loss, optimizer):
        super().compile(optimizer=optimizer, loss=loss)
        self.vae_loss = loss
        self.vae_optimizer = optimizer

        self.vae_loss_metric = keras.metrics.Mean(name="loss")
        self.vae_rec_loss_metric = keras.metrics.Mean(name="rec_loss")
        self.vae_kl_loss_metric = keras.metrics.Mean(name="kl_loss")

        self.vae_val_rec_loss_metric = keras.metrics.Mean(name="loss")

        self.vae_optimizer.build(self.vae_trainable_weights)
        self.built = True

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))

        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]

    @property
    def metrics(self):
        metrics = [
            self.vae_loss_metric,
            self.vae_rec_loss_metric,
            self.vae_kl_loss_metric,
        ]

        return metrics

    @property
    def vae_trainable_weights(self):
        return self.encoder.trainable_weights + self.decoder.trainable_weights

    def call(self, inputs, training, **kwargs):
        if training:
            return self.train_step(inputs)
        else:
            return self.predict(inputs)

    def predict(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    def train_step(self, data):
        x = data

        #############
        # Train VAE #
        #############

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.vae_loss(x, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

            vae_gradients = tape.gradient(total_loss, self.vae_trainable_weights)

            self.vae_optimizer.apply_gradients(
                zip(vae_gradients, self.vae_trainable_weights)
            )

        # Update metrics
        self.vae_loss_metric.update_state(total_loss)
        self.vae_rec_loss_metric(reconstruction_loss)
        self.vae_kl_loss_metric.update_state(kl_loss)

        losses = {
            "loss": self.vae_loss_metric.result(),
            "rec_loss": self.vae_rec_loss_metric.result(),
            "kl_loss": self.vae_kl_loss_metric.result(),
        }

        return losses

    def test_step(self, data):
        x = data

        _, _, z = self.encoder(x, training=False)
        reconstruction = self.decoder(z, training=False)

        rec_loss = self.vae_loss(x, reconstruction)
        self.vae_val_rec_loss_metric.update_state(rec_loss)

        return {"loss": self.vae_val_rec_loss_metric.result()}


def build_encoder(
    latent_dim,
    input_shape,
    filters=(16, 16, 32),
    kernel_sizes=((3, 3), (3, 3), (3, 3)),
    strides=((1, 1), (1, 1), (1, 1)),
    paddings=("valid", "valid", "valid"),
    activation_function=keras.layers.ReLU,
    batch_norm=False,
    n_dense=None,
):
    encoder_input = keras.layers.Input(shape=input_shape, name="encoder_input")
    enc_layer = encoder_input

    for i, (filter_num, kernel_size, stride, padding) in enumerate(
        zip(filters, kernel_sizes, strides, paddings)
    ):
        enc_layer = keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name=f"encoder_conv2D_{i}",
        )(enc_layer)
        if batch_norm:
            enc_layer = keras.layers.BatchNormalization(name=f"encoder_bn_{i}")(
                enc_layer
            )
        enc_layer = activation_function(name=f"encoder_activation_{i}")(enc_layer)

    shape_before_flatten = K.int_shape(enc_layer)[1:]
    encoder_flatten = keras.layers.Flatten(name="encoder_flatten")(enc_layer)

    if n_dense is not None:
        encoder_flatten = keras.layers.Dense(n_dense, name="extra_dense")(
            encoder_flatten
        )
        if batch_norm:
            encoder_flatten = keras.layers.BatchNormalization()(encoder_flatten)
        encoder_flatten = activation_function()(encoder_flatten)

    z_mean = layers.Dense(latent_dim, name="z_mean")(encoder_flatten)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_flatten)

    z = Sampling()([z_mean, z_log_var])

    encoder_output = [z_mean, z_log_var, z]

    encoder = keras.models.Model(encoder_input, encoder_output, name="encoder_model")
    encoder.shape_before_flatten = shape_before_flatten

    return encoder


def build_decoder(
    latent_dim,
    output_shape,
    shape_before_flatten,
    filters=(16, 16, 32),
    kernel_sizes=((3, 3), (3, 3), (3, 3)),
    strides=((1, 1), (1, 1), (1, 1)),
    paddings=("valid", "valid", "valid"),
    activation_function=keras.layers.ReLU,
    final_activation_function=None,
    batch_norm=False,
):
    decoder_input = keras.layers.Input(shape=(latent_dim), name="decoder_input")

    decoder_dense_layer1 = keras.layers.Dense(
        units=np.prod(shape_before_flatten), name="decoder_dense_0"
    )(decoder_input)
    dec_layer = keras.layers.Reshape(target_shape=shape_before_flatten)(
        decoder_dense_layer1
    )

    filters_decode = (
        output_shape[-1],
        *filters[1:],
    )
    for i, (filter_num, kernel_size, stride, padding) in enumerate(
        reversed(list(zip(filters_decode, kernel_sizes, strides, paddings)))
    ):
        dec_layer = keras.layers.Conv2DTranspose(
            filters=filter_num,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name=f"decoder_conv2DTranspose_{i}",
        )(dec_layer)

        if i < len(filters_decode) - 1:
            if batch_norm:
                dec_layer = keras.layers.BatchNormalization(name=f"decoder_bn_{i}")(
                    dec_layer
                )
            dec_layer = activation_function(name=f"decoder_activation_{i}")(dec_layer)

        if final_activation_function and i == len(filters_decode) - 1:
            dec_layer = final_activation_function(dec_layer)

    return keras.models.Model(decoder_input, dec_layer, name="decoder_model")


def build_vae(
    latent_dim,
    input_shape,
    output_shape,
    filters=(16, 16, 32),
    kernel_sizes=((3, 3), (3, 3), (3, 3)),
    strides=((1, 1), (1, 1), (1, 1)),
    paddings=("valid", "valid", "valid"),
    activation_function=keras.layers.ReLU,
    vae_loss_function=keras.losses.MeanSquaredError(reduction="sum"),
    vae_optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    use_batch_norm=False,
    n_dense=None,
    final_activation_function=None,
):
    assert (
        len(filters) == len(kernel_sizes) == len(strides) == len(paddings)
    ), "filters, kernel_sizes, strides and paddings must have the same length"

    encoder = build_encoder(
        latent_dim=latent_dim,
        input_shape=input_shape,
        filters=filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation_function=activation_function,
        batch_norm=use_batch_norm,
        n_dense=n_dense,
    )

    decoder = build_decoder(
        latent_dim=latent_dim,
        output_shape=output_shape,
        shape_before_flatten=encoder.shape_before_flatten,
        filters=filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation_function=activation_function,
        batch_norm=use_batch_norm,
        final_activation_function=final_activation_function,
    )

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        name="VAE-model",
    )

    vae.compile(
        loss=vae_loss_function,
        optimizer=vae_optimizer,
    )

    return vae


if __name__ == "__main__":
    vae = build_vae(
        latent_dim=32,
        input_shape=(16, 16, 1),
        output_shape=(16, 16, 1),
        filters=(16, 16, 16, 32, 32),
        kernel_sizes=((3, 3), (2, 2), (3, 3), (3, 3), (3, 3)),
        strides=((1, 1), (2, 2), (1, 1), (1, 1), (1, 1)),
        paddings=("same", "valid", "valid", "valid", "valid"),
    )
    print(vae.encoder.summary())
    print(vae.decoder.summary())
