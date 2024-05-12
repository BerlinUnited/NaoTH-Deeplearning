import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
from grid_coordinates import bounding_boxes_to_grid
import matplotlib.patches as patches
from pathlib import Path

nboxes = 1
grid_size = (7,7)
H, W = 60, 80

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Each box is [center_x, center_y, width, height] in relative coordinates.
    """
    # Convert from center-relative to corner coordinates
    box1_x1 = box1[..., 0] - 0.5 * box1[..., 2]
    box1_y1 = box1[..., 1] - 0.5 * box1[..., 3]
    box1_x2 = box1[..., 0] + 0.5 * box1[..., 2]
    box1_y2 = box1[..., 1] + 0.5 * box1[..., 3]

    box2_x1 = box2[..., 0] - 0.5 * box2[..., 2]
    box2_y1 = box2[..., 1] - 0.5 * box2[..., 3]
    box2_x2 = box2[..., 0] + 0.5 * box2[..., 2]
    box2_y2 = box2[..., 1] + 0.5 * box2[..., 3]

    # Calculate overlap
    inter_x1 = tf.maximum(box1_x1, box2_x1)
    inter_y1 = tf.maximum(box1_y1, box2_y1)
    inter_x2 = tf.minimum(box1_x2, box2_x2)
    inter_y2 = tf.minimum(box1_y2, box2_y2)

    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)  # Avoid division by zero
    return iou

# Custom YOLO loss function
def yolo_loss(y_true, y_pred, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO loss function.
    """
    # Object mask
    object_mask = y_true[..., 0:1]  # 1 if there's an object, 0 otherwise

    # Calculate IoU for confidence loss
    iou = calculate_iou(y_true[..., 1:5], y_pred[..., 1:5])

    # Confidence loss
    object_confidence_loss = tf.reduce_sum(object_mask * tf.square(1.0 - iou))  # For cells with objects
    no_object_confidence_loss = tf.reduce_sum((1 - object_mask) * tf.square(y_pred[..., 0:1]))  # For empty cells

    confidence_loss = object_confidence_loss + lambda_noobj * no_object_confidence_loss

    # Localization loss for cells with objects
    localization_loss = tf.reduce_sum(
        object_mask * (tf.square(y_true[..., 1:3] - y_pred[..., 1:3]) +  # Position (x, y)
                       tf.square(tf.sqrt(y_true[..., 3:5]) - tf.sqrt(y_pred[..., 3:5])))  # Size (w, h)
    )

    # Total loss with weighted components
    total_loss = lambda_coord * localization_loss + confidence_loss

    return total_loss

def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

    # calculate the left up point & right down point
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)    

offset = []
offset_tran = []
for i in range(grid_size[0]):
    row = []
    row_trans = []
    for j in range(grid_size[0]):
        row.append(j)
        row_trans.append(i)
    offset.append(row)
    offset_tran.append(row_trans)
offset = np.tile(np.array(offset)[None, :, :, None], reps=[1,1,1,nboxes])
offset_tran = np.tile(np.array(offset_tran)[None, :, :, None], reps=[1,1,1,nboxes])

offset = tf.constant(offset, dtype=tf.float32)
offset_tran = tf.constant(offset_tran, dtype=tf.float32)

def yolo_loss2(y_true=None, y_pred=None, eval=False):
    nboxes = 1
    grid_size = (7,7)
    pred_obj_conf = y_pred[:,:,:,:nboxes]
    pred_box_classes = y_pred[:,:,:,5*nboxes:]
    pred_box_offset_coord = y_pred[:,:,:, nboxes:5*nboxes]
    pred_box_offset_coord = tf.reshape(pred_box_offset_coord, shape=[-1, grid_size[0], grid_size[0], nboxes, 4])
    pred_box_normalized_coord = tf.stack([(pred_box_offset_coord[:,:,:,:,0] + offset)/grid_size[0],
                                         (pred_box_offset_coord[:,:,:,:,1] + offset_tran)/grid_size[0],
                                         tf.square(pred_box_offset_coord[:,:,:,:,2]),
                                         tf.square(pred_box_offset_coord[:,:,:,:,3])], axis=-1)
    if eval:
        return pred_obj_conf, pred_box_classes, pred_box_normalized_coord
    target_obj_conf = y_true[:,:,:,:1]
    target_box_classes = y_true[:,:,:,5:]
    target_box_coord = y_true[:,:,:,1:5]
    target_box_coord = tf.reshape(target_box_coord, shape=[-1, grid_size[0], grid_size[1], 1, 4])
    target_box_coord = tf.tile(target_box_coord, multiples=[1,1,1,nboxes,1])
    target_box_normalized_coord = target_box_coord / H
    target_box_offset_coord = tf.stack([target_box_normalized_coord[:,:,:,:,0]*grid_size[0] - offset,
                                        target_box_normalized_coord[:,:,:,:,1]*grid_size[0] - offset_tran,
                                        tf.sqrt(target_box_normalized_coord[:,:,:,:,2]),
                                        tf.sqrt(target_box_normalized_coord[:,:,:,:,3])], axis=-1)

    pred_ious = compute_iou(target_box_normalized_coord, pred_box_normalized_coord)
    predictor_mask = tf.reduce_max(pred_ious, axis=3, keepdims=True)
    predictor_mask = tf.cast(pred_ious>=predictor_mask, tf.float32) * target_obj_conf
    noobj_mask = tf.ones_like(predictor_mask) - predictor_mask

    # Computing the class loss
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_obj_conf*(target_box_classes - pred_box_classes)), axis=[1, 2, 3]))

    # computing the confidence loss
    obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask*(pred_obj_conf - pred_ious)), axis=[1, 2, 3]))
    noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask*(pred_obj_conf)), axis=[1, 2, 3]))

    # computing the localization loss
    predictor_mask = predictor_mask[:,:,:,:,None]
    loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask*(target_box_offset_coord - pred_box_offset_coord)), axis=[1, 2, 3]))

    loss = 10 * loc_loss + obj_loss + 0.1 * noobj_loss + 0.5 * class_loss
    return loss

"""
def yolo_loss(y_true, y_pred, grid_size=7):
    #y_pred[..., 0:3] = tf.nn.sigmoid(y_pred[..., 0:3])  # sigmoid for class confidence, center_x, center_y
    #y_pred[..., 3:5] = tf.nn.relu(y_pred[..., 3:5])  # relu for width, height
    
    # Class confidence
    class_true = y_true[..., 0:1]  # True class
    class_pred = tf.nn.sigmoid(y_pred[..., 0:1])  # Predicted class

    # Relative centers
    x_true = y_true[..., 1:2]  # True center x
    y_true = y_true[..., 2:3]  # True center y
    x_pred = tf.nn.sigmoid(y_pred[..., 1:2])  # Predicted center x
    y_pred = tf.nn.sigmoid(y_pred[..., 2:3])  # Predicted center y

    # Relative sizes
    w_true = y_true[..., 3:4]  # True width
    h_true = y_true[..., 4:5]  # True height
    w_pred = tf.nn.relu(y_pred[..., 3:4])  # Predicted width
    h_pred = tf.nn.relu(y_pred[..., 4:5])  # Predicted height

    # Class loss
    class_loss = tf.reduce_mean(tf.square(class_true - class_pred))

    # Center loss
    center_loss = tf.reduce_mean(tf.square(x_true - x_pred) + tf.square(y_true - y_pred))

    # Size loss
    size_loss = tf.reduce_mean(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    # Total loss with weighting factors
    total_loss = 1.0 * class_loss + 1.0 * center_loss + 1.0 * size_loss

    return total_loss
"""
# Define the YOLO-like model
def create_yolo_model(input_shape=(60, 80, 3), grid_size=7):
    B = 1 # Number of bounding boxes per grid cell
    C = 1 # Number of classes
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(60, 80, 3)),  # Image input
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # Basic convolution
        tf.keras.layers.MaxPooling2D((2, 2)),  # Simple pooling
        tf.keras.layers.Flatten(),  # Flatten the features
        tf.keras.layers.Dense(256, activation='relu'),  # Intermediate dense layer
        tf.keras.layers.Dense(grid_size * grid_size * (B * 5+C)),  # Final dense layer for YOLO output
        tf.keras.layers.Reshape((grid_size, grid_size, B * 5+C))  # Reshape to the YOLO output shape
    ])
    return model

model = create_yolo_model()
model.compile(optimizer='adam', loss=yolo_loss2, metrics=['accuracy'])

# Create a synthetic dataset with basic bounding box annotations
def create_synthetic_dataset(batch_size, image_shape=(60, 80), grid_size=7):
    """
    Create a synthetic dataset with a set number of bounding boxes per image.
    Each bounding box is [class_id, center_x, center_y, width, height] in absolute image coordinates.
    """
    images = []
    annotations = []

    # Generate synthetic images and annotations
    for _ in range(batch_size):
        # Create a random synthetic image
        image = np.random.rand(image_shape[0], image_shape[1], 3)  # Random image
        images.append(image)

        # Create random bounding box annotations
        num_boxes = 1  # Number of bounding boxes per image
        boxes = []
        for _ in range(num_boxes):
            class_id = 1  # Only one class
            center_x = np.random.uniform(10, image_shape[1] - 10)  # Avoiding edge cases
            center_y = np.random.uniform(10, image_shape[0] - 10)
            box_width = np.random.uniform(5, 15)
            box_height = np.random.uniform(5, 15)
            boxes.append([1, center_x, center_y, box_width, box_height, class_id])

        annotations.append(boxes)

    return np.array(images), annotations

# Test the synthetic dataset
# Test the synthetic dataset
images, annotations = create_synthetic_dataset(100)  # Create 10 images with annotations

img = cv2.imread("0005419.png")
print("img shape", img.shape)
images = np.repeat(img[np.newaxis,...], 100, axis=0)
batch_bounding_boxes = [
    # Annotations for the first image
    [
        [1, 39, 30, 13, 13, 1],
    ]
]
#annotations = batch_bounding_boxes * 100


# Verify the synthetic data
print("Sample Image Shape:", images[0].shape)  # Should be (60, 80, 3)
print("Sample Annotations:", annotations[0])  # Should be a list of bounding boxes

# Visualize a sample image with its annotation
def visualize_image_with_annotation(image, annotation):
    """
    Visualize an image with its bounding box annotations.
    """
    plt.imshow(image)
    for bbox in annotation:
        conf, center_x, center_y, box_width, box_height, class_id = bbox
        rect = patches.Rectangle(
            (center_x - box_width / 2, center_y - box_height / 2),
            box_width,
            box_height,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        plt.gca().add_patch(rect)

    plt.savefig("testtest.png", bbox_inches='tight', pad_inches=0.1)  # Save with minimal padding

# Display the first image and its annotation
#visualize_image_with_annotation(images[0], annotations[0])


ground_truth_tensor = bounding_boxes_to_grid(
    annotations,
    grid_size=7,
    image_shape=images[0].shape[:2]
)

print(images.shape)
print(ground_truth_tensor.shape)

# Attempt to train with the synthetic dataset
# For simplicity, this is a single training step
model.fit(images, ground_truth_tensor, batch_size=32, epochs=100)


yolo_output = model.predict(np.expand_dims(images[0], axis=0))


def visualize_prediction(image, predictions):
    # Assumes predictions are in [grid_size, grid_size, 5] with [class, x, y, w, h]
    
    grid_size = predictions.shape[1]

    print(grid_size)
    height, width, _ = image.shape
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    
    for i in range(grid_size):
        for j in range(grid_size):
            pred = predictions[0, i, j, :]
            class_id = pred[0]
            center_x = pred[1]
            center_y = pred[2]
            box_w = pred[3]
            box_h = pred[4]
            print(f"{i,j} cell: {pred}")
            # Convert from grid coordinates to absolute coordinates
            x = (j + center_x) / grid_size * width
            y = (i + center_y) / grid_size * height
            w = box_w * width / grid_size
            h = box_h * height / grid_size
            print(x,y, w,h)
            # Convert to rectangle coordinates for plotting

            try:
                rect = plt.Rectangle(
                    (x - 0.5 * w, y - 0.5 * h),
                    w,
                    h,
                    fill=False,
                    color='red',
                    linewidth=2
                )
                print(rect)
                ax.add_patch(rect)
            except:
                pass

    plt.savefig("train.png", bbox_inches='tight', pad_inches=0.1)  # Save with minimal padding


visualize_prediction(images[0], yolo_output)