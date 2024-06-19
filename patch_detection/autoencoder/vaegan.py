import keras
import tensorflow as tf

from patch_detection.autoencoder.vae import build_vae


@keras.saving.register_keras_serializable()
class VAEGAN(keras.Model):
    def __init__(self, encoder, decoder, discriminator, **kwargs):
        super(VAEGAN, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def compile(self, vae_loss, vae_optimizer, disc_loss, disc_optimizer):
        # Need to set this for LR Reduction callback to work
        super().compile(optimizer=vae_optimizer, loss=vae_loss)

        # VAE
        self.vae_loss = vae_loss
        self.vae_optimizer = vae_optimizer

        self.vae_loss_metric = keras.metrics.Mean(name="loss")
        self.vae_rec_loss_metric = keras.metrics.Mean(name="rec_loss")
        self.vae_kl_loss_metric = keras.metrics.Mean(name="kl_loss")
        self.vae_val_rec_loss_metric = keras.metrics.Mean(name="loss")

        self.vae_optimizer.build(self.vae_trainable_weights)

        # Discriminator
        self.disc_loss = disc_loss
        self.disc_optimizer = disc_optimizer

        self.disc_loss_metric = keras.metrics.Mean(name="disc_loss")
        self.disc_optimizer.build(self.discriminator.trainable_weights)

        # Model
        self.built = True

    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        discriminator = keras.saving.deserialize_keras_object(config.pop("discriminator"))

        return cls(
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            **config,
        )

    @property
    def latent_dim(self):
        return self.decoder.input_shape[-1]

    @property
    def metrics(self):
        metrics = [
            self.vae_loss_metric,
            self.vae_rec_loss_metric,
            self.vae_kl_loss_metric,
            self.disc_loss_metric,
        ]

        return metrics

    @property
    def vae_trainable_weights(self):
        return self.encoder.trainable_weights + self.decoder.trainable_weights

    @property
    def discriminator_trainable_weights(self):
        return self.discriminator.trainable_weights

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
            "discriminator": keras.saving.serialize_keras_object(self.discriminator),
        }
        return {**base_config, **config}

    def train_step(self, data):
        x = data
        batch_size = tf.shape(x)[0]

        #######################
        # Train Discriminator #
        #######################
        self.discriminator.trainable = True
        self.encoder.trainable = False
        self.decoder.trainable = False

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            combined_images = tf.concat([reconstruction, tf.cast(x, dtype="float32")], axis=0)

            # Assemble labels discriminating real from fake images, 0 = fake, 1 = real
            labels = tf.concat(
                [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))],
                axis=0,
            )
            # Add random noise to the labels - important trick!
            # labels += 0.05 * tf.random.uniform(tf.shape(labels))

            predictions = self.discriminator(combined_images)
            disc_loss = self.disc_loss(labels, predictions)

            disc_gradients = tape.gradient(disc_loss, self.discriminator_trainable_weights)
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator_trainable_weights))

        #############
        # Train VAE #
        #############
        self.discriminator.trainable = False
        self.encoder.trainable = True
        self.decoder.trainable = True

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.vae_loss(x, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Assemble labels that say "all real images"
            misleading_labels = tf.ones((batch_size, 1))
            disc_loss_vae = self.disc_loss(misleading_labels, self.discriminator(reconstruction))

            # total_loss = reconstruction_loss + 2 * disc_loss_vae + 0.2 * kl_loss
            total_loss = reconstruction_loss + disc_loss_vae + kl_loss

            vae_gradients = tape.gradient(total_loss, self.vae_trainable_weights)

            self.vae_optimizer.apply_gradients(zip(vae_gradients, self.vae_trainable_weights))

        # Update metrics
        self.vae_loss_metric.update_state(total_loss)
        self.vae_rec_loss_metric(reconstruction_loss)
        self.vae_kl_loss_metric.update_state(kl_loss)
        self.disc_loss_metric.update_state(disc_loss)

        losses = {
            "disc_loss": self.disc_loss_metric.result(),
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


def build_discriminator(
    output_shape,
    n_filters=(16, 16, 32),
):
    discriminator_input = keras.layers.Input(shape=output_shape, name="discriminator_input")
    disc_layer = discriminator_input

    for i, filter_num in enumerate(n_filters):
        disc_layer = keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            padding="same",
            strides=2,
            name=f"discriminator_conv_{i}",
        )(disc_layer)
        disc_layer = keras.layers.BatchNormalization(name=f"discriminator_norm_{i}")(disc_layer)
        disc_layer = keras.layers.LeakyReLU(alpha=0.2, name=f"discriminator_activation_{i}")(disc_layer)

    disc_layer = keras.layers.Flatten()(disc_layer)
    disc_layer = keras.layers.Dense(256)(disc_layer)
    disc_layer = keras.layers.LeakyReLU(alpha=0.2)(disc_layer)
    disc_layer = keras.layers.Dropout(0.2)(disc_layer)
    discriminator_output = keras.layers.Dense(1, activation="sigmoid", name="discriminator_output")(disc_layer)

    return keras.models.Model(discriminator_input, discriminator_output, name="discriminator_model")


def build_vae_gan(*args, **kwargs):
    vae_loss_function = kwargs.pop("vae_loss_function", keras.losses.MeanSquaredError(reduction="sum"))
    vae_optimizer = kwargs.pop("vae_optimizer", keras.optimizers.Adam(learning_rate=0.0005))
    disc_loss_function = kwargs.pop("disc_loss_function", keras.losses.BinaryCrossentropy())
    disc_optimizer = kwargs.pop("disc_optimizer", keras.optimizers.Adam(learning_rate=0.0005))

    n_disc_filters = kwargs.pop("n_disc_filters", (16, 16, 32))

    # Build discriminator
    discriminator = build_discriminator(
        output_shape=kwargs["output_shape"],
        n_filters=n_disc_filters,
    )

    # Build VAE
    vae = build_vae(*args, **kwargs)

    # Build VAE-GAN
    vaegan = VAEGAN(
        encoder=vae.encoder,
        decoder=vae.decoder,
        discriminator=discriminator,
    )

    vaegan.compile(
        vae_loss=vae_loss_function,
        vae_optimizer=vae_optimizer,
        disc_loss=disc_loss_function,
        disc_optimizer=disc_optimizer,
    )

    return vaegan


if __name__ == "__main__":
    vae_gan = build_vae_gan(
        latent_dim=32,
        input_shape=(16, 16, 1),
        output_shape=(16, 16, 1),
        n_disc_filters=(32, 64, 128),
        filters=(16, 16, 16, 32, 32),
        kernel_sizes=((3, 3), (2, 2), (3, 3), (3, 3), (3, 3)),
        strides=((1, 1), (2, 2), (1, 1), (1, 1), (1, 1)),
        paddings=("same", "valid", "valid", "valid", "valid"),
    )

    print(vae_gan.encoder.summary())
    print(vae_gan.decoder.summary())
    print(vae_gan.discriminator.summary())
