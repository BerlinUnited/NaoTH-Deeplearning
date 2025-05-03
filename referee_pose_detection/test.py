import numpy as np
import cv2
import tensorflow as tf

# Load the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (192, 192))  # MoveNet expects 192x192 input
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Run inference using the TFLite model
def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

# Visualize keypoints on the image
def visualize_keypoints(image, keypoints):
    image = image.copy()
    height, width, _ = image.shape
    keypoints = keypoints[0, 0]  # Remove batch and person dimensions

    # Define the keypoint connections (e.g., for limbs)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
        (5, 6), (6, 7), (7, 8),  # Left arm
        (9, 10), (10, 11), (11, 12),  # Right leg
        (13, 14), (14, 15), (15, 16),  # Left leg
        (0, 5), (5, 6), (6, 11), (11, 12),  # Torso
    ]

    # Draw keypoints
    for y, x, confidence in keypoints:
        if confidence > 0.3:  # Only draw keypoints with confidence > 30%
            cv2.circle(image, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)

    # Draw connections
    for start, end in connections:
        y1, x1, conf1 = keypoints[start]
        y2, x2, conf2 = keypoints[end]
        if conf1 > 0.3 and conf2 > 0.3:
            cv2.line(image, (int(x1 * width), int(y1 * height)),
                     (int(x2 * width), int(y2 * height)), (255, 0, 0), 2)

    return image

# Main function
def main(image_path, model_path):
    # Load the TFLite model
    interpreter = load_tflite_model(model_path)

    # Preprocess the image
    image = preprocess_image(image_path)

    # Run inference
    keypoints = run_inference(interpreter, image)

    # Load the original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Visualize keypoints
    output_image = visualize_keypoints(original_image, keypoints)

    # Display the result
    cv2.imshow("MoveNet Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    image_path = "test_image.png"  # Replace with your image path
    model_path = "movenet_lightning.tflite"  # Replace with your TFLite model path
    main(image_path, model_path)