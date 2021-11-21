"""
    This script should run the best model on the trainings data set, the validation dataset and provide metrics for classification

    Additionally this script should output the image paths of the false positives and negative images
"""
import pickle
import tensorflow.keras as keras
import numpy as np
import cv2
import toml
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(Path(__file__).parent.parent.absolute() / "data_balldetection").resolve()
MODEL_DIR = Path(Path(__file__).parent.absolute() / "models").resolve()


def main(config_name):
    with open('classification.toml', 'r') as f:
        config_dict = toml.load(f)

    cfg = config_dict[config_name]

    data_file = str(DATA_DIR / cfg["trainings_data"])
    with open(data_file, "rb") as f:
        mean = pickle.load(f)
        x = pickle.load(f)  # x are all input images
        y = pickle.load(f)  # y are the trainings target: [r, x,y,1] if task is detection

    model_filename = Path(cfg["model_name"] + "_" + cfg["trainings_data"]).with_suffix(".h5")
    model_path = MODEL_DIR / model_filename

    model = keras.models.load_model(model_path)
    model.summary()

    dot_img_file = 'model_1.png'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


    #

    predictions = model.predict(x)
    predictions = np.squeeze(predictions)
    difference = predictions - y
    nicht_erkannt = list()
    for idx, value in enumerate(difference):
        if value == 0.0:
            #print("cool")
            pass
        elif value == -1.0:
            #print("nicht erkannt")
            nicht_erkannt.append(idx)
        else:
            print(value)

    # show images that were false detected
    vis_images = x + mean
    print(len(nicht_erkannt))
    for idx in nicht_erkannt:
        cv2.imshow('image window', vis_images[idx] / 255.0 )
        print(vis_images[idx])
        k = cv2.waitKey(0)
        if k==27:    # Esc key to stop
            break
        else:
            continue
    cv2.destroyAllWindows()


    result = model.evaluate(x, y)
    for idx in range(0, len(result)):
        print(model.metrics_names[idx] + ":", result[idx])

if __name__ == '__main__':
    main("classification_1")