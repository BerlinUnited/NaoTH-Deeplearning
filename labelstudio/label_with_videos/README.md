# Label with Videos

Toolchain to efficiently label many images by converting them to a video, labeling the video in LabelStudio, then pushing annotations back to the original image tasks.

## Workflow

```
get.py → [label video in LS] → export.py → upload.py
```

### 1. `get.py` — Download images and upload as video

Downloads images from LS1 source projects (`PROJECT_IDS`), renders them into an MP4 video, and uploads the video to the video labeling project (`LS_TARGET_PROJECT_ID`). If LS2 is configured, the video goes to LS2. Otherwise it goes to LS1. Creates a `mapping_project_<pid>.json` with the mapping between frames and task IDs.

### 2. Label in LabelStudio

Open the video project in LabelStudio (LS2 if configured, otherwise LS1) and draw bounding boxes on the video. LabelStudio's keyframe interpolation handles the frames in between.

### 3. `export.py` — Read and interpolate video annotations

Reads the video annotations from the video LS (LS2 if configured, otherwise LS1), interpolates the bounding boxes onto each individual frame, and saves them to the mapping file.

### 4. `upload.py` — Push annotations back to images

Takes the interpolated frame annotations and uploads them as individual annotations to the original image tasks on LS1.

## Configuration

All in `config.py`. Two LabelStudio instances are supported:

- **LS1** — source images live here (always required)
- **LS2** — optional, used for video labeling. If not set, LS1 is used for everything.

| Variable | Description |
|----------|-------------|
| `LS1_URL` | LabelStudio URL for source images |
| `LS1_TOKEN` | API key (also via `LABELSTUDIO_API_KEY` env var) |
| `LS2_URL` | Optional second LS for video labeling (env var `LS2_URL`) |
| `LS2_TOKEN` | Token for LS2 (env var `LS2_TOKEN`) |
| `PROJECT_IDS` | List of source project IDs containing images |
| `LS_TARGET_PROJECT_ID` | Project ID for video labeling |
| `DOWNLOAD_DIR` | Folder for temporary videos and mapping JSONs |

## File Structure

```
label_with_videos/
├── config.py          # Centralized configuration (LS1 + optional LS2)
├── utils.py           # Shared helper functions
├── get.py             # Step 1: images → video
├── export.py          # Step 3: video annotations → frames
├── upload.py          # Step 4: frames → image tasks
├── temp_videos/       # Mapping JSONs and temporary videos
└── annotations_project_*.json  # Upload history (idempotency)
```
