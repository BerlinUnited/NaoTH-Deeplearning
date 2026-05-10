from ultralytics import YOLO
model = YOLO("best.pt") 

results_generator = model("robot_log_fixed.mp4", stream=True, show=True, imgsz=640, conf=0.4, iou=0.9, save=True)


for i, result in enumerate(results_generator):
    print(result)
