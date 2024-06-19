import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from keras.preprocessing.image import img_to_array

# from config import yolo_config as cfg

img_h = 60  # img_h=grid_h*cell_h
img_w = 80  # img_w=grid_w*cell_w
nb_boxes = 1  # number of prediction per grid cell
grid_h = 6
grid_w = 8
cell_w = 10
cell_h = 10


def yolo_loss(y_true, y_pred):
    # define a grid of offsets
    # [[[ 0.  0.]]
    # [[ 1.  0.]]
    # [[ 0.  1.]]
    # [[ 1.  1.]]]
    grid = np.array([[[float(x), float(y)]] * nb_boxes for y in range(grid_h) for x in range(grid_w)])

    # first three values are classes : cat, rat, and none.
    # However yolo doesn't predict none as a class, none is everything else and is just not predicted
    # so I don't use it in the loss
    y_true_class = y_true[..., 0:2]
    y_pred_class = y_pred[..., 0:2]

    # reshape array as a list of grid / grid cells / boxes / of 5 elements
    pred_boxes = tf.reshape(y_pred[..., 3:], (-1, grid_w * grid_h, nb_boxes, 5))
    true_boxes = tf.reshape(y_true[..., 3:], (-1, grid_w * grid_h, nb_boxes, 5))

    # sum coordinates of center of boxes with cell offsets.
    # as pred boxes are limited to 0 to 1 range, pred x,y + offset is limited to predicting elements inside a cell
    y_pred_xy = pred_boxes[..., 0:2] + tf.convert_to_tensor(grid, dtype=np.float32)
    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh = pred_boxes[..., 2:4]
    # probability that there is something to predict here
    y_pred_conf = pred_boxes[..., 4]

    # same as predicate except that we don't need to add an offset, coordinate are already between 0 and cell count
    y_true_xy = true_boxes[..., 0:2]
    # with and height
    y_true_wh = true_boxes[..., 2:4]
    # probability that there is something in that cell. 0 or 1 here as it's a certitude.
    y_true_conf = true_boxes[..., 4]

    clss_loss = tf.keras.backend.sum(tf.keras.backend.square(y_true_class - y_pred_class), axis=-1)
    xy_loss = tf.keras.backend.sum(
        tf.keras.backend.sum(tf.keras.backend.square(y_true_xy - y_pred_xy), axis=-1) * y_true_conf, axis=-1
    )
    wh_loss = tf.keras.backend.sum(
        tf.keras.backend.sum(
            tf.keras.backend.square(tf.keras.backend.sqrt(y_true_wh) - tf.keras.backend.sqrt(y_pred_wh)), axis=-1
        )
        * y_true_conf,
        axis=-1,
    )

    # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
    # however the training is a bit unstable

    # compute the intersection of all boxes at once (the IOU)
    intersect_wh = tf.keras.backend.maximum(
        tf.keras.backend.zeros_like(y_pred_wh),
        (y_pred_wh + y_true_wh) / 2 - tf.keras.backend.square(y_pred_xy - y_true_xy),
    )
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_area = y_true_wh[..., 0] * y_true_wh[..., 1]
    pred_area = y_pred_wh[..., 0] * y_pred_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    iou = intersect_area / union_area

    conf_loss = tf.keras.backend.sum(tf.keras.backend.square(y_true_conf * iou - y_pred_conf), axis=-1)

    # final loss function
    d = xy_loss + wh_loss + conf_loss + clss_loss

    if False:
        d = tf.Print(d, [d], "loss")
        d = tf.Print(d, [xy_loss], "xy_loss")
        d = tf.Print(d, [wh_loss], "wh_loss")
        d = tf.Print(d, [clss_loss], "clss_loss")
        d = tf.Print(d, [conf_loss], "conf_loss")

    return d


def load_data():
    x_train = []
    y_train = []
    # FIXME this must be much more generic
    image_root = Path("datasets/ball_only_dataset_80-60/images/rqeruwggyqijgzvnxvimav")
    images = list(image_root.glob("*.png"))
    for image_path in images:
        print(image_path)
        img = cv2.imread(str(image_path))
        x_t = img_to_array(img)
        label_file = Path("datasets/ball_only_dataset_80-60/labels/rqeruwggyqijgzvnxvimav") / Path(
            image_path.name
        ).with_suffix(".txt")
        with open(str(label_file), "r") as f:
            y_t = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                else:
                    # FIXME, not sure if my example also used center coordinated
                    # FIXME cant handle multiple objects in an image
                    # FIXME cant handle empty images yet
                    print(line)
                    a = line.split()
                    # {label_id} {cx} {cy} {width} {height}
                    print(a[0])
                    print(line)
                    x_train.append(x_t)
                    elt = [1.0, 0.0, 0.0, float(a[1]) / cell_w, float(a[2]) / cell_h, float(a[3]), float(a[4]), 1]

                    y_t.append(elt)
                    y_train.append(y_t)
                    print(y_train)
                    quit()

    return np.array(x_train), np.array(y_train)


def stellas_cool_model():
    i = tf.keras.layers.Input(shape=(img_h, img_w, 3))

    x = tf.keras.layers.Conv2D(16, (1, 1))(i)
    x = tf.keras.layers.Conv2D(32, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="sigmoid")(x)
    x = tf.keras.layers.Dense(grid_w * grid_h * (3 + nb_boxes * 5), activation="sigmoid")(x)
    x = tf.keras.layers.Reshape((grid_w * grid_h, (3 + nb_boxes * 5)))(x)

    model = tf.keras.Model(i, x)
    return model


def train():

    # Load all images and append to vector
    x_train, y_train = load_data()

    model = stellas_cool_model()
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=yolo_loss, optimizer=adam)  # better

    print(model.summary())

    model.fit(x_train, y_train, batch_size=64, epochs=10)
    model.save_weights("yolo_weights.h5")


train()
