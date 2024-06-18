

@keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce


@keras.saving.register_keras_serializable(name="WeightedBinaryCrossentropy")
class WeightedBinaryCrossentropy:
    def __init__(
        self,
        label_smoothing=0.0,
        weights=[1.0, 1.0],
        axis=-1,
        name="weighted_binary_crossentropy",
        loss_fn=weighted_binary_crossentropy,
    ):
        """Initializes `WeightedBinaryCrossentropy` instance.

        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).

          TODO: Check if this might be helpful?
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.

          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          name: Name for the op. Defaults to 'weighted_binary_crossentropy'.
        """
        super().__init__()
        self.weights = weights  # tf.convert_to_tensor(weights)
        self.label_smoothing = label_smoothing
        self.name = name
        self.loss_fn = weighted_binary_crossentropy if loss_fn is None else loss_fn

    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(
            self.label_smoothing, dtype=y_pred.dtype
        )

        def _smooth_labels():
            return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(
            self.label_smoothing, _smooth_labels, lambda: y_true
        )

        return tf.reduce_mean(self.loss_fn(y_true, y_pred, self.weights), axis=-1)

    def get_config(self):
        config = {"name": self.name, "weights": self.weights, "loss_fn": self.loss_fn}

        return dict(list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
