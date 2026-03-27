import os
import cv2
from ultralytics import YOLO
from collections import Counter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set bottom or top")

    args = parser.parse_args()

    print("Das Modell wurde nicht festgelegt. Der Inspektor schaut in seinen Spind...")

    model_dir = f"./data/{args.camera}/autolabel_model"
    
    if not os.path.exists(model_dir):
        print(f"Fehler: Der Ordner {model_dir} existiert noch nicht.")
        exit()
    available_models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not available_models:
        print(f"Fehler: Keine Modelle im Ordner {model_dir} gefunden.")
        exit()
        
    print("\nBitte wähle ein Modell aus:")
    for i, model_name in enumerate(available_models):
        print(f"[{i + 1}] {model_name}")
    
    model = ""
    while True:
        try:
            auswahl = int(input("\nGib die Nummer des gewünschten Modells ein: "))
            if 1 <= auswahl <= len(available_models):
                model = available_models[auswahl - 1]
                print(f"--> Modell '{model}' wurde erfolgreich ausgewählt!\n")
                break 
            else:
                print("Ungültige Nummer. Bitte wähle eine Zahl aus der Liste oben.")
        except ValueError:
            print("Das war keine Zahl. Bitte gib eine gültige Ziffer ein.")

    MODEL_PATH = f"./data/{args.camera}/autolabel_model/{model}/weights/best.pt"
    IMAGES_FOLDER  = 'data/TOP/human_proofed/images/val'
    LABELS_FOLDER  = 'data/TOP/human_proofed/labels/val'
    ERROR_FOLDER   = f'data/TOP/error_{model}'

    # Subfolders for FP-only, FN-only, and mixed
    for sub in ['false_positive', 'false_negative', 'both']:
        os.makedirs(os.path.join(ERROR_FOLDER, sub), exist_ok=True)

    model = YOLO(MODEL_PATH)

    images_checked = 0
    stats = {'fp': 0, 'fn': 0, 'both': 0}

    # Confidence tracking
    all_conf  = []   # every predicted ball confidence
    fp_conf   = []   # confidences in FP-only images
    fn_conf   = []   # confidences in FN-only images (predictions that exist)
    both_conf = []   # confidences in "both" images

    for file_name in sorted(os.listdir(IMAGES_FOLDER)):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        images_checked += 1

        image_path = os.path.join(IMAGES_FOLDER, file_name)
        txt_name   = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(LABELS_FOLDER, txt_name)

        # Ground truth
        true_classes = Counter()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        true_classes[int(parts[0])] += 1

        # Prediction
        prediction = model.predict(
            image_path,
            iou=0.2,
            agnostic_nms=True,
            verbose=False
        )[0]

        predicted_classes = Counter(int(box.cls[0].item()) for box in prediction.boxes)

        # Collect all confidences
        confs = [float(box.conf[0]) for box in prediction.boxes]
        all_conf.extend(confs)

        if true_classes == predicted_classes:
            continue  # Perfect match, skip

        # Determine error type
        has_fp = any(predicted_classes[k] > true_classes[k] for k in predicted_classes)
        has_fn = any(true_classes[k] > predicted_classes[k] for k in true_classes)

        if has_fp and has_fn:
            sub = 'both'
            stats['both'] += 1
            both_conf.extend(confs)
        elif has_fp:
            sub = 'false_positive'
            stats['fp'] += 1
            fp_conf.extend(confs)
        else:
            sub = 'false_negative'
            stats['fn'] += 1
            fn_conf.extend(confs)

        for i, box in enumerate(prediction.boxes):
            print(f"  Box {i}: conf={float(box.conf[0]):.2f} "
                f"xyxy={[round(v) for v in box.xyxy[0].tolist()]}")

        # Annotate image
        image_with_boxes = prediction.plot()

        if os.path.exists(label_path):
            original = cv2.imread(image_path)
            h, w = original.shape[:2]

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)

                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)

                        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            image_with_boxes,
                            "GT",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )

        target_path = os.path.join(ERROR_FOLDER, sub, file_name)

        print(f"  GT boxes : {dict(true_classes)}")
        print(f"  Predicted: {dict(predicted_classes)}")
        print(f"  Label file: {label_path} exists={os.path.exists(label_path)}")

        cv2.imwrite(target_path, image_with_boxes)

        print(f"[{sub.upper():15}] {file_name} | "
            f"GT: {dict(true_classes)} | Model: {dict(predicted_classes)}")

    avg = lambda lst: sum(lst) / len(lst) if lst else float('nan')

    print("-" * 50)
    print(f"Checked: {images_checked} images")
    print(f"  FP only : {stats['fp']}")
    print(f"  FN only : {stats['fn']}")
    print(f"  Both    : {stats['both']}")
    print(f"  Total errors: {sum(stats.values())}")
    print(f"  Avg conf all predictions: {avg(all_conf):.2f} | "
        f"FP: {avg(fp_conf):.2f} | "
        f"FN: {avg(fn_conf):.2f} | "
        f"Both: {avg(both_conf):.2f}")

    if images_checked == 0:
        print("WARNING: 0 images processed — please check paths!")