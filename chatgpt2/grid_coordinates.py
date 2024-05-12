import numpy as np

# Function to map bounding boxes to grid cells
def bounding_boxes_to_grid(batch_bboxes, grid_size=7, image_shape=(60, 80)):
    """
    Args:
    - batch_bboxes: A list of lists containing bounding boxes for each image in the batch.
      Each bounding box is represented as [class_id, center_x, center_y, width, height], where
      center_x and center_y are in absolute image coordinates.
    - grid_size: Number of grid cells along each axis.
    - image_shape: Shape of the input images [height, width].

    Returns:
    - ground_truth: Ground truth tensor with shape [batch_size, grid_size, grid_size, 5].
    """
    batch_size = len(batch_bboxes)

    grid_height, grid_width = grid_size, grid_size
    image_height, image_width = image_shape

    # Initialize the ground truth tensor
    ground_truth = np.zeros((batch_size, grid_height, grid_width, 5))

    for b in range(batch_size):
        for bbox in batch_bboxes[b]:

            conf, center_x, center_y, box_width, box_height, class_id = bbox

            # Determine grid cell indices for the bounding box center
            grid_x = int((center_x / image_width) * grid_width)
            grid_y = int((center_y / image_height) * grid_height)

            # Calculate relative positions and sizes within the grid
            rel_center_x = (center_x % (image_width / grid_width)) / (image_width / grid_width)
            rel_center_y = (center_y % (image_height / grid_height)) / (image_height / grid_height)
            rel_width = box_width / image_width * grid_width
            rel_height = box_height / image_height * grid_height

            # Assign the ground truth values
            ground_truth[b, grid_y, grid_x, :] = [class_id, rel_center_x, rel_center_y, rel_width, rel_height]

    return ground_truth

batch_bounding_boxes = [
    # Annotations for the first image
    [
        [1, 39, 30, 13, 13],
    ],
    [
        [1, 39, 30, 13, 13],
    ],
]

#[[1, 39, 30, 13, 13]]

#output = bounding_boxes_to_grid(batch_bounding_boxes)
#print(output)