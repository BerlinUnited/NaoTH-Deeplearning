from ultralytics import YOLO

# 1. Load a pretrained segmentation model
# 'yolo11n-seg.pt' is the smallest/fastest. Use 'yolo11m-seg.pt' for better accuracy.
model = YOLO('yolo26n-seg.pt')

# 2. Train the model
results = model.train(
    data='dataset.yaml', 
    epochs=500,              # Number of passes through the data
    imgsz=640,               # Image size (must be multiple of 32)
    device="cpu",                # Use GPU '0'. If no GPU, use 'cpu'
    project='my_ls_project', # Name of the output folder
    name='seg_experiment'
)