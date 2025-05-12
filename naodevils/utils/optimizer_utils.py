from utils import *
from utils.setup_tensorflow_utils import tf, keras, tensorflow_addons

class DevilsAdam(keras.optimizers.Adam):
    def __init__(self, **kwargs):
        #clipnorm_kwargs = {'clipnorm': 0.001}
        # clipnorm_kwargs = {'clipnorm': 1.0, 'clipvalue': 0.5}
        #clipnorm_kwargs.update(kwargs)
        super(DevilsAdam, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsLAMB(tensorflow_addons.optimizers.LAMB):
    def __init__(self, **kwargs):
        super(DevilsLAMB, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsAdamW(tensorflow_addons.optimizers.AdamW):
    def __init__(self, **kwargs):
        super(DevilsAdamW, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsRAdam(tensorflow_addons.optimizers.RectifiedAdam):
    def __init__(self, **kwargs):
        super(DevilsRAdam, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsLookahead(tensorflow_addons.optimizers.Lookahead):
    def __init__(self, **kwargs):
        clipnorm_kwargs = {'clipnorm': 1.0, 'clipvalue': 1.0}
        clipnorm_kwargs.update(kwargs)
        super(DevilsLookahead, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self._optimizer.learning_rate

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer=optimizer, **config)

class DevilsAMSGrad(keras.optimizers.Adam):
    def __init__(self, **kwargs):
        super(DevilsAMSGrad, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsSGD(keras.optimizers.SGD):
    def __init__(self, **kwargs):
        super(DevilsSGD, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsRMSprop(keras.optimizers.RMSprop):
    def __init__(self, **kwargs):
        super(DevilsRMSprop, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsAdagrad(keras.optimizers.Adagrad):
    def __init__(self, **kwargs):
        super(DevilsAdagrad, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsAdadelta(keras.optimizers.Adadelta):
    def __init__(self, **kwargs):
        super(DevilsAdadelta, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsAdamax(keras.optimizers.Adamax):
    def __init__(self, **kwargs):
        super(DevilsAdamax, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate

class DevilsNadam(keras.optimizers.Nadam):
    def __init__(self, **kwargs):
        super(DevilsNadam, self).__init__(**kwargs)

    def get_actual_learning_rate(self):
        return self.learning_rate