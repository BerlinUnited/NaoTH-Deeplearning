# /lustre/informatik/schlotts/yolo_balls_dataset

from ultralytics import YOLO

# Load a lightweight model like YOLOv8n
model = YOLO("yolov8n.pt")

# Train on your new dataset
model.train(data="/lustre/informatik/schlotts/yolo_balls_dataset/dataset.yaml", epochs=1000, imgsz=640)