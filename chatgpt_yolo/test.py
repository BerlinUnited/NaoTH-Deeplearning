import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from main import annotations_to_yolo_format, YOLOLoss, extract_bounding_boxes
import cv2

# Define the YOLO loss function as previously described

# Configuration
S = 7  # Grid size
B = 1  # Number of bounding boxes per grid cell
C = 2  # Number of classes
lambda_coord = 5
lambda_noobj = 0.5
num_samples = 100
input_shape = (60, 80, 3)
output_shape = (S, S, B * 5 + C)  # Proper output shape: (7 x 7 x 30)

# Define a simple YOLO model with the correct output shape
def simple_yolo_model(S, B, C):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(60, 80, 3)),  # Image input
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # Basic convolution
        tf.keras.layers.MaxPooling2D((2, 2)),  # Simple pooling
        tf.keras.layers.Flatten(),  # Flatten the features
        tf.keras.layers.Dense(256, activation='relu'),  # Intermediate dense layer
        tf.keras.layers.Dense(S * S * (B * 5 + C)),  # Final dense layer for YOLO output
        tf.keras.layers.Reshape((S, S, B * 5 + C))  # Reshape to the YOLO output shape
    ])
    return model

# Instantiate YOLO loss
yolo_loss = YOLOLoss(S, B, C, lambda_coord, lambda_noobj)

# Create the model
model = simple_yolo_model(S, B, C)

# Compile the model with the custom YOLO loss
model.compile(optimizer='adam', loss=yolo_loss, metrics=['accuracy'])

# Dummy data generation (for demonstration; replace with real dataset)


# Random input data (representing images)
img = cv2.imread("0005419.png")
print("img shape", img.shape)
X_train = np.repeat(img[np.newaxis,...], 100, axis=0)
#X_train = np.random.rand(num_samples, *input_shape)

# Random output data (representing YOLO outputs)
Y_train = np.random.rand(num_samples, *output_shape)
print(Y_train.shape)
annotations = [
    [0, 0.4919928550720215, 0.5065828959147135, 0.1762082099914551, 0.23383344014485677]
]
yolo_target = annotations_to_yolo_format(annotations, S, B, C)
print(yolo_target.shape)

# Train the model (with a small number of epochs)
model.fit(X_train, Y_train, epochs=100)

img2 = cv2.imread("0005419.png")
yolo_output = model.predict(np.expand_dims(img2, axis=0))
print(yolo_output.shape)



# Extract bounding boxes
bounding_boxes = extract_bounding_boxes(yolo_output, S, B, C, 0.0)

# Visualization function to plot bounding boxes
def plot_bounding_boxes(input_image, bounding_boxes, grid_size):
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(input_image)
    ax = plt.gca()

    # Draw grid lines (optional, for visualization)
    #for i in range(1, grid_size):
    #    plt.axvline(i / grid_size, color='red', linestyle='--')
    #    plt.axhline(i / grid_size, color='red', linestyle='--')

    # Draw bounding boxes
    for box in bounding_boxes:
        x, y, w, h, confidence = box

        # Determine the grid cell this box belongs to
        cell_x = int(x * grid_size)
        cell_y = int(y * grid_size)

        # Calculate the relative position of the bounding box within the grid cell
        # This positions the bounding box's top-left corner within the corresponding grid cell
        x_center = (x - 0.5) / grid_size + cell_x / grid_size
        y_center = (y - 0.5) / grid_size + cell_y / grid_size
        x_min = x_center - w / 2 / grid_size
        y_min = y_center - h / 2 / grid_size
        
        # Create a rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min),
            w / grid_size,
            h / grid_size,
            linewidth=1,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')  # Optionally hide axis
    plt.savefig("test.png", bbox_inches='tight', pad_inches=0.1)  # Save with minimal padding
    plt.close(fig)  # Close the figure to free memory

# Plot bounding boxes
plot_bounding_boxes(img2, bounding_boxes, S)