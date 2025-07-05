from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")
n_frames = 2


# Run inference on the video with stream=True to get a generator
# This allows you to process frames one by one without loading the whole video into memory
results_generator = model("/home/stella/2025-03-15_17-15-00_BerlinUnited_vs_Hulks_half2_Field-B_PiCam.mp4", stream=True, show=True)

# Iterate through the generator and process the first n frames
for i, result in enumerate(results_generator):
    if i >= n_frames:
        break  # Stop after processing n frames
    # You can do further processing on each 'result' object here if needed
    # For example, you can access the bounding boxes, confidence scores, etc.
    # print(f"Frame {i+1}: {len(result.boxes)} objects detected.")
    #print(result.path)
    break
    
print(f"Inference completed for the first {n_frames} frames.")
