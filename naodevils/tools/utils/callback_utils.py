import warnings
import time
import numpy as np
import pickle
from utils import *
from utils.setup_tensorflow_utils import keras
from keras.utils import io_utils
from utils.plotting_utils import PLOT_FORMAT

class DevilsCallback(keras.callbacks.Callback):
    def __init__(self, evaluate, checkpoint_filepath, file_name, monitor_1='val_loss', monitor_2='val_loss', verbose=1,
                 save_best_only=False, save_weights_only=False, initial_step=0,
                 mode_1='auto', mode_2='auto', period=2, log_dir=os.path.join("data", "summary"), warmup_epochs=0, callbacks_to_save=None):
        from tensorflow.python.ops.summary_ops_v2 import create_file_writer_v2

        super(DevilsCallback, self).__init__()
        self.callbacks_to_save = callbacks_to_save
        self.checkpoint_filepath = checkpoint_filepath
        self.filepath = str(checkpoint_filepath + file_name)
        self.cb_filepath = str(checkpoint_filepath + "-{epoch:04d}-callbacks.pkl")
        self.optimizer_filepath = str(checkpoint_filepath + "-{epoch:04d}-optimizer.pkl")
        self.last_filepath = ""
        self.last_cb_filepath = ""
        self.last_optimizer_filepath = ""
        self.monitor_1 = monitor_1
        self.monitor_2 = monitor_2
        self.verbose = verbose
        # self.generated_model_file = ""
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.evaluate = evaluate
        self.warmup_epochs = warmup_epochs
        self.epochs_since_last_evaluate = 0

        if mode_1 not in ['auto', 'min', 'max'] or mode_2 not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode_1),
                          RuntimeWarning)
            mode_1 = 'auto'
            mode_2 = 'auto'

        self.monitor_1_op, self.best_1, self.best_1_epoch = self.check_mode(monitor_1, mode_1)
        self.monitor_2_op, self.best_2, self.best_2_epoch = self.check_mode(monitor_2, mode_2)

        self._total_batches_seen = initial_step
        self.log_dir = log_dir
        self.train_log_dir = os.path.join(self.log_dir, "tools")
        self.train_writer_name = self.train_log_dir[len(FLAGS.summary_dir):]
        self.train_writer = create_file_writer_v2(self.train_log_dir, name=self.train_writer_name)

    def check_mode(self, monitor, mode):
        if mode == 'min':
            return np.less, np.Inf, 0
        elif mode == 'max':
            return np.greater, -np.Inf, 0
        else:
            if 'acc' in monitor or \
                    'iou' in monitor or \
                    'recall' in monitor or \
                    'precision' in monitor or \
                    'fscore' in monitor or \
                    monitor.startswith('fmeasure'):
                return np.greater, -np.Inf, 0
            else:
                return np.less, np.Inf, 0

    def on_epoch_begin(self, epoch, logs=None):
        print()
        print_seperator()

    def on_epoch_end(self, epoch, logs=None):
        if type(self.model.layers[1]) == keras.Sequential or type(self.model.layers[1]) == keras.Model:
            func_model = self.model.layers[1]
            func_model.optimizer = self.model.optimizer
            # func_model.loss = None  # self.model.loss
            # func_model._compile_metrics = None  # self.model._compile_metrics
            # func_model._compile_weighted_metrics = None  # self.model._compile_weighted_metrics
            # func_model.sample_weight_mode = self.model.sample_weight_mode
            # func_model.loss_weights = None  # self.model.loss_weights
        else:
            func_model = self.model

        logs = logs or {}
        if epoch >= self.warmup_epochs and "val_loss" in logs:
            epoch_string = "-" + "{epoch:04d}".format(epoch=epoch + 1)
            filepath = self.filepath.format(epoch_string=epoch_string, **logs)
            if self.save_best_only:
                current_1 = logs.get(self.monitor_1)
                current_2 = logs.get(self.monitor_2)
                if current_1 is None or current_2 is None:
                    warnings.warn('Can save best model only with %s and %s available, '
                                  'skipping.' % (self.monitor_1, self.monitor_2), RuntimeWarning)
                else:
                    new_best = False
                    if self.monitor_1_op(current_1, self.best_1):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f (%05d) to %0.5f' %
                                  (epoch + 1, self.monitor_1, self.best_1, self.best_1_epoch, current_1))
                        self.best_1 = current_1
                        self.best_1_epoch = epoch + 1
                        new_best = True
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did NOT improve from %0.5f (%05d) to %0.5f' %
                                  (epoch + 1, self.monitor_1, self.best_1, self.best_1_epoch, current_1))

                    if self.monitor_2_op(current_2, self.best_2):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f (%05d) to %0.5f' %
                                  (epoch + 1, self.monitor_2, self.best_2, self.best_2_epoch, current_2))
                        self.best_2 = current_2
                        self.best_2_epoch = epoch + 1
                        new_best = True
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did NOT improve from %0.5f (%05d) to %0.5f' %
                                  (epoch + 1, self.monitor_2, self.best_2, self.best_2_epoch, current_2))

                    if new_best:
                        self.save_model(func_model, filepath, epoch, new_best=True)
                    else:
                        self.save_model(func_model, filepath, epoch, new_best=False)
            else:
                self.save_model(func_model, filepath, epoch)

    def on_batch_end(self, batch, logs=None):
        self.log_learning_rate("batch_", self._total_batches_seen)
        self._total_batches_seen += 1

    def save_model(self, func_model, filepath, epoch, new_best=False):
        if self.save_weights_only:
            func_model.save_weights(filepath, overwrite=True)
        else:
            func_model.save(filepath, overwrite=True)

        # Save Callback State
        cb_filepath = self.cb_filepath.format(epoch=epoch + 1)
        optimizer_filepath = self.optimizer_filepath.format(epoch=epoch + 1)
        saved_data = []
        with open(cb_filepath, "wb") as f:
            for cb in self.callbacks_to_save:
                save_config = getattr(cb, "save_config", None)
                if callable(save_config):
                    saved_data.append(save_config())
            pickle.dump(saved_data, f)

        # Save optimizer weights.
        symbolic_weights = getattr(func_model.optimizer, 'weights')
        if symbolic_weights:
            weight_values = keras.backend.batch_get_value(symbolic_weights)
            with open(optimizer_filepath, "wb") as f:
                pickle.dump(weight_values, f)

        self.epochs_since_last_evaluate += 1

        if self.save_best_only:
            if self.last_filepath != "":
                try:
                    os.remove(self.last_filepath)
                except OSError:
                    pass
            if self.last_cb_filepath != "":
                try:
                    os.remove(self.last_cb_filepath)
                except OSError:
                    pass
            if self.last_optimizer_filepath != "":
                try:
                    os.remove(self.last_optimizer_filepath)
                except OSError:
                    pass
            if new_best:
                self.last_filepath = ""
                self.last_cb_filepath = ""
                self.last_optimizer_filepath = ""

                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))

                if self.evaluate and self.epochs_since_last_evaluate > self.period:
                    self.epochs_since_last_evaluate = 0
                    evaluate_file = self.checkpoint_filepath + "-" + "{epoch:04d}".format(epoch=epoch + 1)
                    self.evaluate(evaluate_file)
            else:
                self.last_filepath = filepath
                self.last_cb_filepath = cb_filepath
                self.last_optimizer_filepath = optimizer_filepath

        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))

            if self.evaluate and self.epochs_since_last_evaluate > self.period:
                self.epochs_since_last_evaluate = 0
                evaluate_file = self.checkpoint_filepath + "-" + "{epoch:04d}".format(epoch=epoch + 1)
                self.evaluate(evaluate_file)

    def log_learning_rate(self, prefix, step):
        from tensorflow.python.ops.summary_ops_v2 import always_record_summaries, scalar
        from tensorflow.python.eager.context import eager_mode

        learning_rate = self.model.optimizer.get_actual_learning_rate().numpy()
        with eager_mode():
            with always_record_summaries():
                with self.train_writer.as_default():
                    scalar(prefix + "learning_rate", learning_rate, step=step)

    def on_train_end(self, logs=None):
        print("Training finished ...")

    def save_config(self):
        d = {
            "string_repr": self.__str__()[1:self.__str__().find("object")-1],
            "last_filepath": self.last_filepath,
            "last_cb_filepath": self.last_cb_filepath,
            "last_optimizer_filepath": self.last_optimizer_filepath,
            "epochs_since_last_evaluate": self.epochs_since_last_evaluate,
            "monitor_1": self.monitor_1,
            "monitor_1_op": self.monitor_1_op,
            "best_1": self.best_1,
            "best_1_epoch": self.best_1_epoch,
            "monitor_2": self.monitor_2,
            "monitor_2_op": self.monitor_2_op,
            "best_2": self.best_2,
            "best_2_epoch": self.best_2_epoch,
            "_total_batches_seen": self._total_batches_seen,
        }
        return d

    def load_config(self, dict):
        if "last_filepath" in dict: self.last_filepath = dict["last_filepath"]
        if "last_cb_filepath" in dict: self.last_cb_filepath = dict["last_cb_filepath"]
        if "last_optimizer_filepath" in dict: self.last_optimizer_filepath = dict["last_optimizer_filepath"]
        if "epochs_since_last_evaluate" in dict: self.epochs_since_last_evaluate = dict["epochs_since_last_evaluate"]
        if "monitor_1" in dict: self.monitor_1 = dict["monitor_1"]
        if "monitor_1_op" in dict: self.monitor_1_op = dict["monitor_1_op"]
        if "best_1" in dict: self.best_1 = dict["best_1"]
        if "best_1_epoch" in dict: self.best_1_epoch = dict["best_1_epoch"]
        if "monitor_2" in dict: self.monitor_2 = dict["monitor_2"]
        if "monitor_2_op" in dict: self.monitor_2_op = dict["monitor_2_op"]
        if "best_2" in dict: self.best_2 = dict["best_2"]
        if "best_2_epoch" in dict: self.best_2_epoch = dict["best_2_epoch"]
        if "_total_batches_seen" in dict: self._total_batches_seen = dict["_total_batches_seen"]

class IncreaseAugmentationOnPlateau(keras.callbacks.Callback):
    def __init__(self,
                 train_generator,
                 monitor='val_loss',
                 mode='auto',
                 increment=0.1,
                 patience=10,
                 min_delta=1e-4,
                 cooldown=0,
                 verbose=0,
                 **kwargs):
        super(IncreaseAugmentationOnPlateau, self).__init__()
        self.actual_augmentation_threshold = 0.0
        self.train_generator = train_generator
        self.monitor = monitor
        if increment >= 1.0:
            raise ValueError('IncreaseAugmentationOnPlateau ' 'does not support an increment >= 1.0.')

        self.increment = increment
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        from tensorflow.python.platform.tf_logging import warning
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warning('Increase Augmentation on Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and
                 ('acc' not in self.monitor and
                    'iou' not in self.monitor and
                    'recall' not in self.monitor and
                    'precision' not in self.monitor and
                    'fscore' not in self.monitor and
                    not self.monitor.startswith('fmeasure')))):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf

        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self.train_generator.set_augmentation_threshold(self.actual_augmentation_threshold)
        if self.verbose > 0:
            print('IncreaseAugmentationOnPlateau starts with %.2f augmentation.' % (self.actual_augmentation_threshold))

    def reset(self, epoch):
        if self.verbose > 0:
            print('Epoch %05d: Resetting IncreaseAugmentationOnPlateau.' % (epoch + 1))
        self._reset()
        self.actual_augmentation_threshold = 0.0
        self.train_generator.set_augmentation_threshold(self.actual_augmentation_threshold)

    def on_epoch_end(self, epoch, logs=None):
        from tensorflow.python.platform.tf_logging import info
        logs = logs or {}
        current = logs.get(self.monitor)
        print()
        if current is None:
            info('Increase Augmentation on Plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        else:
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                self.cooldown_counter = 0
                #if self.verbose > 0:
                #    print('\nEpoch %05d: IncreaseAugmentationOnPlateau(%.2f) new best -> resetting.' % (epoch + 1, self.actual_augmentation_threshold))
            elif self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
                if self.verbose > 0:
                    print('Epoch %05d: IncreaseAugmentationOnPlateau(%.2f) still in cooldown for #%s Epochs.' % (epoch + 1, self.actual_augmentation_threshold, self.cooldown_counter))
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_augmentation = self.train_generator.augmentation_threshold
                    if old_augmentation < 1.0:
                        self.actual_augmentation_threshold = old_augmentation + self.increment
                        self.actual_augmentation_threshold = min(self.actual_augmentation_threshold, 1.0)
                        self.train_generator.set_augmentation_threshold(self.actual_augmentation_threshold)
                        if self.verbose > 0:
                            print('Epoch %05d: IncreaseAugmentationOnPlateau increased augmentation to %.2f.' % (epoch + 1, self.actual_augmentation_threshold))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                    else:
                        self.reset(epoch)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: IncreaseAugmentationOnPlateau(%.2f) waiting for improvement for #%s/%s Epochs.' % (epoch + 1, self.actual_augmentation_threshold, self.wait, self.patience))

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def save_config(self):
        d = {
            "string_repr": self.__str__()[1:self.__str__().find("object")-1],
            "actual_augmentation_threshold": self.actual_augmentation_threshold,
            "cooldown_counter": self.cooldown_counter,
            "best": self.best,
            "wait": self.wait,
        }
        return d

    def load_config(self, dict):
        if "actual_augmentation_threshold" in dict: self.actual_augmentation_threshold = dict["actual_augmentation_threshold"]
        if "cooldown_counter" in dict: self.cooldown_counter = dict["cooldown_counter"]
        if "best" in dict: self.best = dict["best"]
        if "wait" in dict: self.wait = dict["wait"]

class DevilsStatistics(keras.callbacks.Callback):
    def __init__(self,
                 train_generator,
                 valid_generator,
                 verbose,
                 **kwargs):
        super(DevilsStatistics, self).__init__()
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.verbose = verbose
    def on_test_begin(self, logs=None):
        if self.verbose == 1:
            print("\nImage Cache: " + str(self.train_generator.load_image_and_bbox_training.cache_info()))
            print("NP Cache: " + str(self.train_generator.interpolate_blurriness.cache_info()))
    def on_epoch_end(self, epoch, logs=None):
        self.train_generator.shuffle_images()
        if self.verbose == 1:
            print("\nImage Cache: " + str(self.valid_generator.load_image_and_bbox_validation.cache_info()))
            print("NP Cache: " + str(self.valid_generator.interpolate_blurriness.cache_info()))


class DevilsProgbarLogger(keras.callbacks.ProgbarLogger):
    # Callback that prints metrics to stdout.

    def __init__(self, count_mode='samples', stateful_metrics=None, validation_steps=None):
        super(DevilsProgbarLogger, self).__init__(count_mode=count_mode, stateful_metrics=stateful_metrics)
        self.is_validation = False
        self.validation_target = validation_steps
        self.last_train_logs = None

    def on_train_begin(self, logs=None):
        super(DevilsProgbarLogger, self).on_train_begin(logs=logs)
        self._called_in_fit = False
        self.is_validation = False

    def on_test_begin(self, logs=None):
        self._finalize_progbar(self.last_train_logs, self._train_step)
        self.is_validation = True
        super(DevilsProgbarLogger, self).on_test_begin(logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        super(DevilsProgbarLogger, self).on_train_batch_end(batch=batch, logs=logs)
        self.last_train_logs = logs

    def on_test_end(self, logs=None):
        self.previous_target = self.target
        self.target = self.validation_target
        super(DevilsProgbarLogger, self).on_test_end(logs=logs)
        self.is_validation = False
        self.target = self.previous_target

    def on_epoch_end(self, epoch, logs=None):
        pass

    def _maybe_init_progbar(self):
        """Instantiate a `Progbar` if not yet, and update the stateful
        metrics."""
        # TODO(rchao): Legacy TF1 code path may use list for
        # `self.stateful_metrics`. Remove "cast to set" when TF1 support is
        # dropped.
        self.stateful_metrics = set(self.stateful_metrics)

        if self.model:
            # Update the existing stateful metrics as `self.model.metrics` may
            # contain updated metrics after `MetricsContainer` is built in the
            # first train step.
            self.stateful_metrics = self.stateful_metrics.union(
                set(m.name for m in self.model.metrics)
            )

        if self.progbar is None:
            self.progbar = DevilsProgbar(
                target=self.target if not self.is_validation else self.validation_target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
                unit_name="step" if self.use_steps else "sample",
                is_validation=self.is_validation
            )

        self.progbar._update_stateful_metrics(self.stateful_metrics)

class DevilsProgbar(keras.utils.Progbar):
    """Displays a progress bar. Updated for validation!!!

    Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
          be averaged over time. Metrics in this list will be displayed as-is.
          All others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """
    def __init__(
            self,
            target,
            width=30,
            verbose=1,
            interval=0.05,
            stateful_metrics=None,
            unit_name='step',
            is_validation=False
    ):
        super(DevilsProgbar, self).__init__(
            target=target,
            width=width,
            verbose=verbose,
            interval=interval,
            stateful_metrics=stateful_metrics,
            unit_name=unit_name)

        self._dynamic_display = True
        self._is_validation = is_validation

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Args:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is
              in `stateful_metrics`, `value_for_last_step` will be displayed
              as-is. Else, an average of the metric over time will be
              displayed.
            finalize: Whether this is the last update for the progress bar. If
              `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in
                # the first epoch, both on_batch_end and on_epoch_end will be
                # called, which will cause 'current' and 'self._seen_so_far' to
                # have the same value. Force the minimal value to 1 here,
                # otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        message = ""
        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if current == self.target:
            self._time_at_epoch_end = now
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                message += "\b" * prev_total_width
                message += "\r"
            else:
                message += "\n"

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            message += bar

            time_per_unit = self._estimate_step_duration(current, now)

            if self.target is None or finalize:
                info += self._format_time(time_per_unit, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format

            for k in self._values_order:
                if self._is_validation:
                    info += ' - val_%s:' % k
                else:
                    if str(k).startswith("val_"):
                        continue
                    info += ' -     %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if abs(avg) > 1e-3:
                        info += " %8.4f" % avg
                    else:
                        info += " %8.1e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if finalize:
                info += "\n"

            message += info
            io_utils.print_msg(message, line_break=False)
            message = ""

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    if self._is_validation:
                        info += ' - val_%s:' % k
                    else:
                        if str(k).startswith("val_"):
                            continue
                        info += ' -     %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if avg > 1e-3:
                        info += " %8.4f" % avg
                    else:
                        info += " %8.1e" % avg
                if self._time_at_epoch_end:
                    time_per_epoch = (
                        self._time_at_epoch_end - self._time_at_epoch_start
                    )
                    avg_time_per_step = time_per_epoch / self.target
                    self._time_at_epoch_start = now
                    self._time_at_epoch_end = None
                    info += " -" + self._format_time(time_per_epoch, "epoch")
                    info += " -" + self._format_time(
                        avg_time_per_step, self.unit_name
                    )
                    info += "\n"
                message += info
                io_utils.print_msg(message, line_break=False)
                message = ""

        self._last_update = now


class DevilsEarlyStopping(keras.callbacks.Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Arguments:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used.
    """

    def __init__(self,
                 monitor_1='val_loss',
                 monitor_2='val_fscore',
                 patience=0,
                 verbose=0,
                 mode_1='auto',
                 mode_2='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(DevilsEarlyStopping, self).__init__()

        self.monitor_1 = monitor_1
        self.monitor_2 = monitor_2
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode_1 not in ['auto', 'min', 'max'] or mode_2 not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode_1),
                          RuntimeWarning)
            mode_1 = 'auto'
            mode_2 = 'auto'

        self.monitor_1_op, self.best_1, self.best_1_epoch = self.check_mode(monitor_1, mode_1)
        self.monitor_2_op, self.best_2, self.best_2_epoch = self.check_mode(monitor_2, mode_2)


    def check_mode(self, monitor, mode):
        if mode == 'min':
            return np.less, np.Inf, 0
        elif mode == 'max':
            return np.greater, -np.Inf, 0
        else:
            if 'acc' in monitor or \
                    'iou' in monitor or \
                    'recall' in monitor or \
                    'precision' in monitor or \
                    'fscore' in monitor or \
                    monitor.startswith('fmeasure'):
                return np.greater, -np.Inf, 0
            else:
                return np.less, np.Inf, 0

    def on_epoch_end(self, epoch, logs=None):
        current_1 = self.get_monitor_value(self.monitor_1, logs)
        current_2 = self.get_monitor_value(self.monitor_2, logs)

        if current_1 is None or current_2 is None:
            return

        new_best = False
        if self.monitor_1_op(current_1, self.best_1):
            self.best_1 = current_1
            self.best_1_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            new_best = True

        if self.monitor_2_op(current_2, self.best_2):
            self.best_2 = current_2
            self.best_2_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            new_best = True

        if not new_best:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping waiting for improvement for #%s/%s Epochs.' % (epoch + 1, self.wait, self.patience))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, monitor, logs):
        from tensorflow.python.platform.tf_logging import info
        logs = logs or {}
        monitor_value = logs.get(monitor)
        if monitor_value is None:
            info('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            monitor, ','.join(list(logs.keys())))
        return monitor_value

    def save_config(self):
        d = {
            "string_repr": self.__str__()[1:self.__str__().find("object")-1],
            "best_1": self.best_1,
            "best_1_epoch": self.best_1_epoch,
            "best_2": self.best_2,
            "best_2_epoch": self.best_2_epoch,
            "wait": self.wait,
        }
        return d

    def load_config(self, dict):
        if "best_1" in dict: self.best_1 = dict["best_1"]
        if "best_1_epoch" in dict: self.best_1_epoch = dict["best_1_epoch"]
        if "best_2" in dict: self.best_2 = dict["best_2"]
        if "best_2_epoch" in dict: self.best_2_epoch = dict["best_2_epoch"]
        if "wait" in dict: self.wait = dict["wait"]