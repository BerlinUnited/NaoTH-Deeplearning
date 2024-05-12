import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

# Function to visualize the 10x10 grid on an image and highlight a bounding box
def visualize_grid_and_annotation(image_path, annotation, grid_size=7):
    """
    Visualize a 10x10 grid on top of the image and highlight the bounding box
    and the grid cell that contains the center of the bounding box.

    Args:
    - image_path: Path to the image file to be displayed.
    - annotation: A list representing the bounding box annotation [class_id, center_x, center_y, width, height].
      The coordinates should be in image-level (absolute) coordinates.
    - grid_size: Number of grid cells along each axis (default is 10x10).
    """
    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Draw the 10x10 grid
    for i in range(1, grid_size):
        # Vertical lines
        ax.axvline(x=i * (img_width / grid_size), color='gray', linestyle='--', linewidth=1)
        # Horizontal lines
        ax.axhline(y=i * (img_height / grid_size), color='gray', linestyle='--', linewidth=1)

    # Extract annotation information
    class_id, center_x, center_y, box_width, box_height = annotation
    """
    center_x = center_x * 100
    center_y = center_y * 100
    box_width = box_width * 100
    box_height = box_height * 100
    """
    # Calculate grid indices for the center of the bounding box
    grid_x = int((center_x / img_width) * grid_size)
    grid_y = int((center_y / img_height) * grid_size)
    print("grid index is", grid_x, grid_y)
    # Highlight the grid cell that contains the bounding box center
    grid_cell = patches.Rectangle(
        ((grid_x) * (img_width / grid_size), (grid_y) * (img_height / grid_size)),
        img_width / grid_size,
        img_height / grid_size,
        fill=True,
        color='red',
        alpha=0.5
    )
    ax.add_patch(grid_cell)

    # Draw the bounding box on the image
    bounding_box = patches.Rectangle(
        (center_x - (box_width / 2), center_y - (box_height / 2)),
        box_width,
        box_height,
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    ax.add_patch(bounding_box)

    # Set title with class_id
    ax.set_title(f"Class ID: {class_id}")

    # Show the image with grid and bounding box
    plt.savefig("output.png", bbox_inches='tight', pad_inches=0.1)  # Save with minimal padding

# Example usage
image_path = "0005419.png"  # Path to your image file
#annotation = [0, 0.4919928550720215, 0.5065828959147135, 0.1762082099914551, 0.23383344014485677]  # [class_id, center_x, center_y, width, height]
annotation = [1, 39, 30, 13, 13]
visualize_grid_and_annotation(image_path, annotation)