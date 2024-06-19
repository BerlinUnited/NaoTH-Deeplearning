import os
import cv2


def load_data_from_ultralytics_dataset():
    # Expect that all txt files contain labels
    # Path to the datasets
    dataset_dir = "datasets/ball_only_dataset_80-60"

    # Subdirectories for images and labels
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    # Dictionary to store the mapping of images to their corresponding labels
    image_label_mapping = {}

    # Iterate over the subfolders in the images directory
    for folder_name in os.listdir(images_dir):
        images_subfolder_path = os.path.join(images_dir, folder_name)
        labels_subfolder_path = os.path.join(labels_dir, folder_name)

        # Ensure the corresponding labels subfolder exists
        if os.path.isdir(labels_subfolder_path):
            # Iterate over the image files in the current images subfolder
            for image_file in os.listdir(images_subfolder_path):
                image_file_path = os.path.join(images_subfolder_path, image_file)

                # Create the expected label file path
                label_file_name = os.path.splitext(image_file)[0] + ".txt"
                label_file_path = os.path.join(labels_subfolder_path, label_file_name)

                # Ensure the label file exists
                if os.path.isfile(label_file_path):
                    # Load the image
                    # image = Image.open(image_file_path)
                    image = cv2.imread(str(image_file_path))
                    image = image / 255.0

                    # Read the label data
                    with open(label_file_path, "r") as f:
                        label_data = f.read()

                    # Store the image and its corresponding label in the mapping
                    # TODO I dont think that works well with multiple labels per image
                    image_label_mapping[image_file_path] = {
                        "image": image,
                        "label": label_data,
                    }
    return image_label_mapping
