import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from main import annotations_to_yolo_format, YOLOLoss

import cv2
S = 7  # Grid size
B = 1  # Number of bounding boxes per grid cell
C = 2  # Number of classes
lambda_coord = 5
lambda_noobj = 0.5
num_samples = 100
input_shape = (60, 80, 3)
output_shape = (S, S, B * 5 + C)  # Proper output shape: (7 x 7 x 30)
yolo_loss = YOLOLoss(S, B, C, lambda_coord, lambda_noobj)

img = cv2.imread("0005419.png")
model = tf.keras.models.load_model('yolo.h5', compile=False)
model.compile(optimizer='adam', loss=yolo_loss, metrics=['accuracy'])
yolo_output = model.predict(img)
print(yolo_output.shape)