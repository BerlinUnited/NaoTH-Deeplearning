# Label with Videos

Toolchain to efficiently label many images by converting them to a video, labeling the video in LabelStudio, then pushing annotations back to the original image tasks.

## Workflow

```
get.py → [label video in LS] → export.py → upload.py
```

### 1. `get.py` — Download images and upload as video

Downloads images from the source projects (`PROJECT_IDS`), renders them into an MP4 video, and uploads the video to the video labeling project (`LS_TARGET_PROJECT_ID`). Creates a `mapping_project_<pid>.json` with the mapping between frames and task IDs.

### 2. Label in LabelStudio

Open the video project in LabelStudio and draw bounding boxes on the video. LabelStudio's keyframe interpolation handles the frames in between.

### 3. `export.py` — Read and interpolate video annotations

Reads the video annotations from LabelStudio, interpolates the bounding boxes onto each individual frame, and saves them to the mapping file.

### 4. `upload.py` — Push annotations back to images

Takes the interpolated frame annotations and uploads them as individual annotations to the original image tasks.

## Configuration

All in `config.py`:

| Variable | Description |
|----------|-------------|
| `LS_URL` | LabelStudio URL |
| `LS_TOKEN` | API key (also via `LABELSTUDIO_API_KEY` env var) |
| `PROJECT_IDS` | List of source project IDs containing images |
| `LS_TARGET_PROJECT_ID` | Project ID for video labeling |
| `DOWNLOAD_DIR` | Folder for temporary videos and mapping JSONs |

## File Structure

```
label_with_videos/
├── config.py          # Centralized configuration
├── utils.py           # Shared helper functions
├── get.py             # Step 1: images → video
├── export.py          # Step 3: video annotations → frames
├── upload.py          # Step 4: frames → image tasks
├── temp_videos/       # Mapping JSONs and temporary videos
└── annotations_project_*.json  # Upload history (idempotency)
```
