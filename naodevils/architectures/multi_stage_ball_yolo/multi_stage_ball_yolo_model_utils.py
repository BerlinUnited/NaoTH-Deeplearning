import cv2
import numpy as np
import threading
from utils import *
from utils.util_functions import _sigmoid
from architectures.multi_stage_ball_yolo.multi_stage_ball_yolo_loss import MultiStageBallYoloLoss

predict_lock = threading.Condition()

def predict_batch_multi_stage_ball(model, valid_generator, idx, obj_threshold=0.5, nms_threshold=0.1, draw=False, force=False, resize_factor=1, t=None):
    x_batch, y_true_batch = valid_generator.__getitem__(idx)

    confidence_index = 1
    position_index = 1 - confidence_index

    y_true_confidence = y_true_batch[confidence_index]
    y_true_position = y_true_batch[position_index]

    with predict_lock:
        netouts = model(x_batch)
    if t:
        t.update(len(x_batch))

    loss = MultiStageBallYoloLoss(width=FLAGS.width, height=FLAGS.height,
                                  threshold=FLAGS.iou_threshold,
                                  f_beta=FLAGS.f_beta,
                                  coord_scale=FLAGS.coord_scale,
                                  object_scale=FLAGS.object_scale)
    l_confidence = loss.confidence_loss(y_true_confidence, netouts[confidence_index]).numpy()
    l_position = loss.position_loss(y_true_position, netouts[position_index]).numpy()

    shape_0 = x_batch[0].shape[0]
    shape_1 = x_batch[0].shape[1]
    pre_converted_netouts = _sigmoid(netouts[confidence_index][..., 0])
    converted_netouts = [_sigmoid(netouts[position_index][..., 0]),
                         _sigmoid(netouts[position_index][..., 1]) * shape_0,
                         _sigmoid(netouts[position_index][..., 2]) * shape_1]

    batch_predictions = []
    batch_images = []
    for i in range(len(x_batch)):
        binary_y_pred = int(pre_converted_netouts[i] > FLAGS.iou_threshold and converted_netouts[0][i] > obj_threshold)
        binary_y_true = int(y_true_position[i][0] >= FLAGS.iou_threshold)
        if binary_y_true == 1:
            deviation = np.abs(converted_netouts[1][i] - y_true_position[i][1]) + np.abs(converted_netouts[2][i] - y_true_position[i][2])
        else:
            deviation = 0.0

        batch_prediction = []
        batch_prediction.append(binary_y_pred)                         # 0 - binary y_pred
        batch_prediction.append(binary_y_true)                         # 1 - binary y_true
        batch_prediction.append(converted_netouts[0][i])               # 2 - y_pred_position
        batch_prediction.append(deviation)                             # 3 - deviation
        batch_prediction.append(converted_netouts[1][i])               # 4 - ball_x_pos
        batch_prediction.append(converted_netouts[2][i])               # 5 - ball_y_pos
        batch_prediction.append(y_true_confidence[i][1])               # 6 - blurriness
        batch_prediction.append(l_position[i] + 0.5 * l_confidence[i]) # 7 - loss
        batch_prediction.append(pre_converted_netouts[i])              # 8 - y_pred_confidence
        batch_prediction.append(y_true_confidence[i][2])               # 9 - visibility
        batch_predictions.append(batch_prediction)

        if draw:
            image = x_batch[i]
            if FLAGS.norm:
                image = image * 255.
            image = image.astype('uint8')

            image = cv2.resize(image, (resize_factor*FLAGS.width, resize_factor*FLAGS.height), interpolation=cv2.INTER_NEAREST)

            if binary_y_true == 1:
                cv2.drawMarker(image, (int(round(y_true_position[i][1] * resize_factor)), int(round(y_true_position[i][2] * resize_factor))), (0, 0, 255), cv2.MARKER_CROSS, resize_factor * 2, max(2, resize_factor))
                cv2.circle(image, (int(round(y_true_position[i][1] * resize_factor)), int(round(y_true_position[i][2] * resize_factor))), int(round(y_true_position[i][3] * resize_factor / 2.0)), (0, 0, 255), max(2, resize_factor))

            if binary_y_pred == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.drawMarker(image, (int(round(converted_netouts[1][i] * resize_factor)), int(round(converted_netouts[2][i] * resize_factor))), color, cv2.MARKER_CROSS, resize_factor * 2, max(2, resize_factor))
            batch_images.append(image)

    return batch_predictions, batch_images