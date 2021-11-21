"""
    This is code for evaluating synthetic vs. real data in the classification task
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


loss_figure = plt.figure(1)
acc_figure = plt.figure(2)

file_list = list()
file_list.append("models/history_naoth_classification1_bhuman_classification_16x16.pkl")
file_list.append("models/history_naoth_classification2_bhuman_classification.pkl")

for filename in file_list:
    with open(str(filename), 'rb') as f:
        # load trainings history of a single run
        history = pickle.load(f)

        # get loss and acc
        plt.figure(1)
        loss = np.array(history['val_loss'])
        acc = np.array(history['val_accuracy'])

        # plot trainings progress
        plt.plot(history['val_loss'], label=str(Path(filename).name))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')

        plt.figure(2)
        plt.plot(history['val_accuracy'], label=Path(filename).name)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')


plt.show()
