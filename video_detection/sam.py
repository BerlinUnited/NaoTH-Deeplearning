

from ultralytics import FastSAM
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

results_generator = model("/mnt/d/logs/2026-03-10-GO26/2026-03-11_11-50-00_Bit-Bots_vs_Berlin United_half1/videos/2026-03-11_11-50-00_Bit-Bots_vs_Berlin United_half1_Field-C_PiCam.mp4", stream=True, show=True, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

n_frames = 3

for i, result in enumerate(results_generator):
    #if i >= n_frames:
    #    break  # Stop after processing n frames
    # You can do further processing on each 'result' object here if needed
    # For example, you can access the bounding boxes, confidence scores, etc.
    # print(f"Frame {i+1}: {len(result.boxes)} objects detected.")
    print(result)
    #quit()
    
print(f"Inference completed for the first {n_frames} frames.")
