from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load the YOLOv8 pose estimation model
model = YOLO('yolov8n-pose.pt')

images = Path("/mnt/d/arms-up_01/top/").glob('*.png')
for image_path in images:
    # Run pose detection on an image
    results = model.predict(source=str(image_path), save=False, conf=0.5)

    image = cv2.imread(str(image_path))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib


    # Extract keypoints and plot them
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints in numpy format

        for person in keypoints:
        # Plot keypoints
            for x, y in person:
                if x > 0 and y > 0:  # Filter out invalid keypoints
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dots

            # Draw skeleton (connect keypoints)
            skeleton = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
                (1, 5), (5, 6), (6, 7),  # Left arm
                (1, 8), (8, 9), (9, 10),  # Right leg
                (1, 11), (11, 12), (12, 13)  # Left leg
            ]
            #for start, end in skeleton:
            #    if person[start][0] > 0 and person[start][1] > 0 and person[end][0] > 0 and person[end][1] > 0:
            #        plt.plot([person[start][0], person[end][0]], [person[start][1], person[end][1]], color='blue', linewidth=2)

    # Show the plot
    cv2.imwrite(f'/mnt/d/arms-up_01/top_output/{image_path.name}.jpg', image)