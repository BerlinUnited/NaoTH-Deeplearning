from utils import *
from utils.setup_tensorflow_utils import tf, keras

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

class MultiStageBallYoloLoss(object):
    def __init__(self, width=32, height=32, threshold=0.5, f_beta=1/4, object_scale=0.1, coord_scale=1.0, warmup_batches=300):
        super(MultiStageBallYoloLoss, self).__init__()
        self.width = tf.convert_to_tensor(width, dtype=tf.float32)
        self.height = tf.convert_to_tensor(height, dtype=tf.float32)
        self.threshold = tf.convert_to_tensor(threshold, dtype=tf.float32)
        self.f_beta = tf.convert_to_tensor(f_beta, dtype=tf.float32)
        self.f_beta_squared = tf.convert_to_tensor(f_beta ** 2, dtype=tf.float32)
        self.f_beta_inv_squared = tf.convert_to_tensor((1/f_beta) ** 2, dtype=tf.float32)
        self.zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.zero_point_five = tf.convert_to_tensor(0.5, dtype=tf.float32)
        self.one = tf.convert_to_tensor(1.0, dtype=tf.float32)
        self.onehundred = tf.convert_to_tensor(100.0, dtype=tf.float32)
        self.coord_scale = tf.convert_to_tensor(coord_scale, dtype=tf.float32)
        self.object_scale = tf.convert_to_tensor(object_scale, dtype=tf.float32)
        self.wh_tensor = tf.convert_to_tensor([self.width, self.height], dtype=tf.float32)
        self.c_precision = None
        self.c_recall = None
        self.c_fscore = None
        self.p_precision = None
        self.p_recall = None
        self.p_fscore = None
        self.p_deviation = None
        self.p_loss_confidence = None
        self.p_loss_xy = None

        self.seen_p = tf.Variable(0.)
        self.seen_c = tf.Variable(0.)
        self.warmup_batches = tf.convert_to_tensor(warmup_batches, dtype=tf.float32)

    @rename('p')
    def confidence_precision(self, y_true, y_pred):
        if not tf.is_tensor(self.c_precision):
            self.confidence_loss(y_true, y_pred)
        return self.c_precision

    @rename('p')
    def position_precision(self, y_true, y_pred):
        if not tf.is_tensor(self.p_precision):
            self.position_loss(y_true, y_pred)
        return self.p_precision

    @rename('r')
    def confidence_recall(self, y_true, y_pred):
        if not tf.is_tensor(self.c_recall):
            self.confidence_loss(y_true, y_pred)
        return self.c_recall

    @rename('r')
    def position_recall(self, y_true, y_pred):
        if not tf.is_tensor(self.p_recall):
            self.position_loss(y_true, y_pred)
        return self.p_recall

    @rename('f')
    def confidence_fscore(self, y_true, y_pred):
        if not tf.is_tensor(self.c_fscore):
            self.confidence_loss(y_true, y_pred)
        return self.c_fscore

    @rename('f')
    def position_fscore(self, y_true, y_pred):
        if not tf.is_tensor(self.p_fscore):
            self.position_loss(y_true, y_pred)
        return self.p_fscore

    @rename('d')
    def position_deviation(self, y_true, y_pred):
        if not tf.is_tensor(self.p_deviation):
            self.position_loss(y_true, y_pred)
        return self.p_deviation

    @rename('lc')
    def position_loss_confidence(self, y_true, y_pred):
        if not tf.is_tensor(self.p_loss_confidence):
            self.position_loss(y_true, y_pred)
        return self.p_loss_confidence

    @rename('lp')
    def position_loss_xy(self, y_true, y_pred):
        if not tf.is_tensor(self.p_loss_xy):
            self.position_loss(y_true, y_pred)
        return self.p_loss_xy

    def recall_precision_fscore_loss(self, true_confidence, y_pred, pred_threshold, f_beta_squared, tp_factor, fn_factor, fp_factor, tn_factor, focal=True):
        true_confidence_binary = tf.cast(true_confidence >= self.threshold, 'float')

        ####################################
        #### F-Score, Recall, Precision ####
        ####################################
        pred_confidence = tf.sigmoid(y_pred[..., 0])
        pred_confidence_binary = tf.cast((pred_confidence >= pred_threshold), 'float')

        tp = tf.cast(true_confidence_binary * pred_confidence_binary, 'float')
        tn = tf.cast((1 - true_confidence_binary) * (1 - pred_confidence_binary), 'float')
        fp = tf.cast((1 - true_confidence_binary) * pred_confidence_binary, 'float')
        fn = tf.cast(true_confidence_binary * (1 - pred_confidence_binary), 'float')

        tp_sum = tf.reduce_sum(tp, axis=0)
        tn_sum = tf.reduce_sum(tn, axis=0)
        fp_sum = tf.reduce_sum(fp, axis=0)
        fn_sum = tf.reduce_sum(fn, axis=0)

        p = tp_sum / (tp_sum + fp_sum + keras.backend.epsilon())
        p = tf.cond(tf.math.equal(tp_sum, self.zero), lambda: tf.ones_like(p), lambda: p)
        r = tp_sum / (tp_sum + fn_sum + keras.backend.epsilon())

        f_score = (1 + f_beta_squared) * tp_sum / ((1 + f_beta_squared) * tp_sum + f_beta_squared * fn_sum + fp_sum + keras.backend.epsilon())
        f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)

        #########################
        #### Confidence Loss ####
        #########################
        confidence_loss_mask = tp * tp_factor
        confidence_loss_mask += fn * fn_factor
        confidence_loss_mask += fp * fp_factor
        confidence_loss_mask += tn * tn_factor
        if focal:
            loss_confidence = keras.backend.binary_focal_crossentropy(true_confidence, pred_confidence)
        else:
            loss_confidence = tf.square(true_confidence - pred_confidence)
        loss_confidence = tf.multiply(loss_confidence, confidence_loss_mask)
        loss_confidence = tf.where(tf.math.is_nan(loss_confidence), tf.zeros_like(loss_confidence), loss_confidence)

        return r, p, f_score, loss_confidence

    def confidence_loss(self, y_true, y_pred):
        true_confidence, loss_epsilon, fscore_epsilon = tf.cond(tf.less(self.seen_c, self.warmup_batches + 1),
                                               lambda: [tf.ones_like(y_true[..., 0]), 0.5, -20.0],
                                               lambda: [y_true[..., 0], 0.0, 0.0])
        visibility_factor = y_true[..., 2]
        influence_factor = tf.add(self.one, tf.minimum(true_confidence, visibility_factor))
        r, p, f_score, loss = self.recall_precision_fscore_loss(true_confidence, y_pred,
                                                                pred_threshold=self.threshold,
                                                                f_beta_squared=self.f_beta_inv_squared,
                                                                tp_factor=1.0,
                                                                fn_factor=100.0,
                                                                fp_factor=10.0,
                                                                tn_factor=1.0,
                                                                focal=True)
        self.c_recall = tf.multiply(r, self.onehundred)
        self.c_precision = tf.multiply(p, self.onehundred)
        self.c_fscore = tf.add(tf.multiply(f_score, self.onehundred), fscore_epsilon)

        total_loss = tf.multiply(loss, influence_factor)
        total_loss = tf.add(total_loss, loss_epsilon)
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.ones_like(total_loss), total_loss)
        self.seen_c.assign_add(1.)
        return total_loss

    def position_loss(self, y_true, y_pred):
        true_confidence, true_position, loss_epsilon, fscore_epsilon = tf.cond(tf.less(self.seen_p, self.warmup_batches + 1),
                                                                 lambda: [tf.zeros_like(y_true[..., 0]), tf.ones_like(y_true[..., 1:3]) * self.zero_point_five * self.wh_tensor, 0.5, -20.0],
                                                                 lambda: [y_true[..., 0], y_true[..., 1:3], 0.0, 0.0])

        true_confidence_binary = tf.cast(true_confidence >= self.threshold, 'float')
        visibility_factor = y_true[..., 5]
        influence_factor = tf.add(self.one, tf.minimum(true_confidence, visibility_factor))
        r, p, f_score, conf_loss = self.recall_precision_fscore_loss(true_confidence, y_pred,
                                                                     pred_threshold=self.threshold,
                                                                     f_beta_squared=self.f_beta_squared,
                                                                     tp_factor=1.0,
                                                                     fn_factor=10.0,
                                                                     fp_factor=100.0,
                                                                     tn_factor=1.0,
                                                                     focal=True)
        self.p_recall = tf.multiply(r, self.onehundred)
        self.p_precision = tf.multiply(p, self.onehundred)
        self.p_fscore = tf.add(tf.multiply(f_score, self.onehundred), fscore_epsilon)
        self.p_loss_confidence = tf.multiply(conf_loss, tf.multiply(self.object_scale, influence_factor))

        ##################
        #### Position ####
        ##################
        pred_position = tf.sigmoid(y_pred[..., 1:3]) * self.wh_tensor
        diff_position = (true_position - pred_position)

        distance_manhattan = (tf.abs(diff_position[..., 0]) + tf.abs(diff_position[..., 1]))
        distance_manhattan = tf.multiply(distance_manhattan, influence_factor)
        distance = distance_manhattan * true_confidence_binary

        self.p_deviation = distance
        self.p_loss_xy = tf.multiply(distance, self.coord_scale)

        total_loss = tf.add(self.p_loss_confidence, self.p_loss_xy)
        total_loss = tf.add(total_loss, loss_epsilon)
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.ones_like(total_loss), total_loss)

        self.seen_p.assign_add(1.)
        return total_loss
