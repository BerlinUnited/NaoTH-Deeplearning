from utils import *
from utils.setup_tensorflow_utils import tf, keras

class BallYoloLoss(object):
    def __init__(self, width=32, height=32, threshold=0.5, f_beta=1/4, object_scale=0.1, coord_scale=1.0):
        super(BallYoloLoss, self).__init__()
        self.width = float(width)
        self.height = float(height)
        self.threshold = tf.convert_to_tensor(threshold, dtype=tf.float32)
        self.f_beta = tf.convert_to_tensor(f_beta, dtype=tf.float32)
        self.f_beta_squared = tf.convert_to_tensor(f_beta ** 2, dtype=tf.float32)
        self.zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
        self.one = tf.convert_to_tensor(1.0, dtype=tf.float32)
        self.two = tf.convert_to_tensor(2.0, dtype=tf.float32)
        self.coord_scale = tf.convert_to_tensor(coord_scale, dtype=tf.float32)
        self.object_scale = tf.convert_to_tensor(object_scale, dtype=tf.float32)
        self.wh_tensor = tf.convert_to_tensor([self.width, self.height], dtype=tf.float32)
        self.calculated_loss_conf = None
        self.calculated_loss_xy = None
        self.calculated_precision = None
        self.calculated_recall = None
        self.calculated_deviation = None
        self.calculated_fscore = None

    def loss_xy(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_loss_xy):
            self.loss(y_true, y_pred)
        return self.calculated_loss_xy

    def loss_conf(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_loss_conf):
            self.loss(y_true, y_pred)
        return self.calculated_loss_conf

    def precision(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_precision):
            self.loss(y_true, y_pred)
        return self.calculated_precision

    def recall(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_recall):
            self.loss(y_true, y_pred)
        return self.calculated_recall

    def fscore(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_fscore):
            self.loss(y_true, y_pred)
        return self.calculated_fscore

    def deviation(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_deviation):
            self.loss(y_true, y_pred)
        return self.calculated_deviation

    def ball_cnn_loss(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        return loss

    def loss(self, y_true, y_pred):
        ####################
        #### Confidence ####
        ####################
        true_confidence = y_true[..., 0]
        true_confidence_binary = tf.cast(true_confidence >= self.threshold, 'float')
        pred_confidence = tf.sigmoid(y_pred[..., 0])
        visibility_factor = y_true[..., 6]

        ####################################
        #### F-Score, Recall, Precision ####
        ####################################
        selected_pred_confidence = tf.cast((pred_confidence >= self.threshold), 'float')
        tp = tf.cast(true_confidence_binary * selected_pred_confidence, 'float')
        tn = tf.cast((1 - true_confidence_binary) * (1 - selected_pred_confidence), 'float')
        fp = tf.cast((1 - true_confidence_binary) * selected_pred_confidence, 'float')
        fn = tf.cast(true_confidence_binary * (1 - selected_pred_confidence), 'float')

        tp_sum = tf.reduce_sum(tp, axis=0)
        tn_sum = tf.reduce_sum(tn, axis=0)
        fp_sum = tf.reduce_sum(fp, axis=0)
        fn_sum = tf.reduce_sum(fn, axis=0)

        p = tp_sum / (tp_sum + fp_sum + keras.backend.epsilon())
        p = tf.cond(tf.math.equal(tp_sum, self.zero), lambda: tf.ones_like(p), lambda: p)
        r = tp_sum / (tp_sum + fn_sum + keras.backend.epsilon())

        f_score_class1 = (1 + self.f_beta_squared) * tp_sum / ((1 + self.f_beta_squared) * tp_sum + (self.f_beta_squared) * fn_sum + fp_sum + keras.backend.epsilon())
        f_score_class1 = tf.where(tf.math.is_nan(f_score_class1), tf.zeros_like(f_score_class1), f_score_class1)

        self.calculated_precision = p * 100.0
        self.calculated_recall = r * 100.0
        self.calculated_fscore = f_score_class1 * 100.0

        #########################
        #### Confidence Loss ####
        #########################
        tp_factor = 1.0
        fn_factor = 10.0
        tn_factor = 1.0
        fp_factor = 1000.0
        influence_factor = tf.add(self.one, tf.minimum(true_confidence, visibility_factor))

        confidence_loss_mask = tp * tp_factor
        confidence_loss_mask += fn * fn_factor
        confidence_loss_mask += tn * tn_factor
        confidence_loss_mask += fp * fp_factor

        loss_confidence_bce = tf.multiply(keras.backend.binary_focal_crossentropy(tf.minimum(true_confidence, visibility_factor), pred_confidence), confidence_loss_mask)
        loss_confidence_bce = tf.where(tf.math.is_nan(loss_confidence_bce), tf.zeros_like(loss_confidence_bce), loss_confidence_bce)

        self.calculated_loss_conf = tf.multiply(loss_confidence_bce, tf.multiply(self.object_scale, influence_factor))

        ##################
        #### Position ####
        ##################
        true_position = y_true[..., 1:3]
        pred_position = tf.sigmoid(y_pred[..., 1:3]) * self.wh_tensor
        diff_position = (true_position - pred_position)

        distance_manhattan = (tf.abs(diff_position[..., 0]) + tf.abs(diff_position[..., 1])) * influence_factor
        distance = distance_manhattan * true_confidence_binary

        self.calculated_deviation = distance
        self.calculated_loss_xy = tf.multiply(tf.divide(distance, self.two), tf.multiply(self.coord_scale, influence_factor))

        total_loss = tf.add(self.calculated_loss_conf, self.calculated_loss_xy)
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.ones_like(total_loss), total_loss)
        return total_loss
