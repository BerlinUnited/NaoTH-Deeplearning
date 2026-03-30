# Workflow - classifier

## Preparation Step

Get the BallCandidatePatches from the log-files.

```
uv run log_iteration.py -l 679,683
```

## Step 1 Create Patches

## Step 2 Train model

---

## Testing the Model in the Log Simulator

## 1. Load the TFLite model

Place the `.tflite` file in:

```
naoth-2020/NaoTHSoccer/Config/
```

## 2. Register the classifier (first-time setup only)

If not previously configured, add the classifier path in:

```
naoth-2020/NaoTHSoccer/Source/Cognition/Modules/VisualCortex/BallDetector/CNNBallDetector.cpp
```

## 3. Configure the Log Simulator

In the Log Simulator, open `CNNBallDetector` and:

- Set `classifier` and `classifierClose` to point to your `.tflite` file
- Tune `cnn.threshold` and `cnn.thresholdClose` as needed
