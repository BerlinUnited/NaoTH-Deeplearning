

from ultralytics import FastSAM
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

results_generator = model("/home/stella/2025-03-15_17-15-00_BerlinUnited_vs_Hulks_half2_Field-B_PiCam.mp4", stream=True, show=True, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

n_frames = 3

for i, result in enumerate(results_generator):
    if i >= n_frames:
        break  # Stop after processing n frames
    # You can do further processing on each 'result' object here if needed
    # For example, you can access the bounding boxes, confidence scores, etc.
    # print(f"Frame {i+1}: {len(result.boxes)} objects detected.")
    result.to_sql()
    quit()
    
print(f"Inference completed for the first {n_frames} frames.")
