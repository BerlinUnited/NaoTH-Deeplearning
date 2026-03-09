from ultralytics import YOLO
new_model = YOLO("best.pt")
results = new_model("image.jpg")