import numpy as np
import tensorflow as tf

def annotations_to_yolo_format(annotations, S, B, C):
    # Create a blank output tensor with zeros
    output_shape = (S, S, B * 5 + C)  # Ensure correct shape
    print(output_shape)
    yolo_target = np.zeros(output_shape)

    # Iterate through each annotation
    for annotation in annotations:
        class_id = int(annotation[0])  # Class ID
        center_x = annotation[1]  # Bounding box center x (normalized)
        center_y = annotation[2]  # Bounding box center y (normalized)
        width = annotation[3]  # Bounding box width (normalized)
        height = annotation[4]  # Bounding box height (normalized)

        # Determine the grid cell to which the bounding box belongs
        cell_x = int(center_x * S)
        cell_y = int(center_y * S)

        # Check for available bounding box slots in the grid cell
        found_slot = False
        for b in range(B):
            # Objectness index for this bounding box
            obj_idx = C + b
            # Start index for the bounding box parameters
            start_idx = C + b * 5  # Corrected index for bounding box data in each cell
            print("start_idx", start_idx)
            if yolo_target[cell_y, cell_x, obj_idx] == 0:  # Check if slot is free
                # Set the class probability for the appropriate class
                yolo_target[cell_y, cell_x, class_id] = 1

                # Set the objectness score to 1 (since this is a target, we know there's an object)
                yolo_target[cell_y, cell_x, obj_idx] = 1

                # Calculate the bounding box parameters relative to the grid cell
                box_params = [
                    (center_x * S) - cell_x,  # Local x-center within the cell
                    (center_y * S) - cell_y,  # Local y-center within the cell
                    width * S,  # Local width relative to the grid
                    height * S,  # Local height relative to the grid
                    1  # Confidence (1 for targets)
                ]

                # Assign bounding box parameters to the correct slot
                yolo_target[cell_y, cell_x, start_idx:start_idx + 5] = box_params

                found_slot = True
                break  # Found a slot, no need to check further

        if not found_slot:
            print(f"Warning: No available slot for grid cell ({cell_x}, {cell_y}).")

    return yolo_target

class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, S, B, C, lambda_coord, lambda_noobj):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        # Split predictions into class, objectness, and bounding box predictions
        pred_class = y_pred[..., :self.C]
        pred_obj = y_pred[..., self.C:self.C + self.B]
        pred_box = y_pred[..., self.C + self.B:]

        # Split targets into class, objectness, and bounding box targets
        target_class = y_true[..., :self.C]
        target_obj = y_true[..., self.C:self.C + self.B]
        target_box = y_true[..., self.C + self.B:]
        print(target_obj.shape)
        # Calculate binary cross-entropy loss for objectness
        obj_loss = tf.keras.losses.BinaryCrossentropy()(target_obj, pred_obj)

        # Calculate no-object loss
        noobj_loss = tf.keras.losses.BinaryCrossentropy()(1 - target_obj, 1 - pred_obj)

        # Mean squared error for bounding box coordinates
        box_loss = tf.keras.losses.MeanSquaredError()(pred_box, target_box)

        # Class prediction loss
        class_loss = tf.keras.losses.BinaryCrossentropy()(target_class, pred_class)

        # Total loss with lambda weighting
        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            self.lambda_noobj * noobj_loss +
            class_loss
        )

        return total_loss

# Post-process the output to get bounding boxes and confidence scores
def extract_bounding_boxes(yolo_output, S, B, C, threshold):
    # Get the grid cells
    output_shape = yolo_output.shape
    grid_size = output_shape[1]
    
    # Extract bounding boxes and objectness scores
    bounding_boxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = yolo_output[0, i, j]
            print("cell")
            print(cell)
            class_scores = cell[0] # cell[:C]
            objectness_scores = cell[0] #cell[C:C + B]
            box_data = cell[2:6] # cell[C + B:]
            print(objectness_scores)

            # Only consider boxes with objectness scores above the threshold
            """
            # does not work for only one bounding box prediction
            for b in range(B):
                confidence = objectness_scores[b]
                if confidence > threshold:
                    # Bounding box parameters
                    x = box_data[b * 5 + 0]  # x-center
                    y = box_data[b * 5 + 1]  # y-center
                    w = box_data[b * 5 + 2]  # Width
                    h = box_data[b * 5 + 3]  # Height
                    bounding_boxes.append((x, y, w, h, confidence))
            """
            b = 0
            confidence = objectness_scores
            if confidence > threshold:
                # Bounding box parameters
                x = box_data[b * 5 + 0]  # x-center
                y = box_data[b * 5 + 1]  # y-center
                w = box_data[b * 5 + 2]  # Width
                h = box_data[b * 5 + 3]  # Height
                bounding_boxes.append((x, y, w, h, confidence))

    return bounding_boxes