"""
    This is code for evaluating synthetic vs. real data in the classification task
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

file_list = list()
file_list.append("models/history_naoth_classification_16_16_rc19_classification_16_bw_combined.pkl")
file_list.append("models/history_naoth_classification_16_16_rc19_classification_16_bw_bottom.pkl")
file_list.append("models/history_naoth_classification_16_16_rc19_classification_16_bw_top.pkl")


def plot_loss():
    # plot loss
    for filename in file_list:
        with open(str(filename), 'rb') as f:
            # load trainings history of a single run
            history = pickle.load(f)

            # get loss and acc
            plt.figure(1)

            loss = np.array(history['val_loss'])
            new_label = str(Path(filename).name).split("_")[-1]

            # plot trainings progress
            plt.plot(history['val_loss'], label=new_label, linewidth=3)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.title("Validation loss over epochs")

    plt.tight_layout()
    # plt.show()
    plt.savefig('exp_cam_val_loss.png')
    plt.clf()


def plot_val():
    # plot accuraccy
    for filename in file_list:
        with open(str(filename), 'rb') as f:
            # load trainings history of a single run
            history = pickle.load(f)

            # get loss and acc
            plt.figure(1)

            acc = np.array(history['val_accuracy'])
            new_label = str(Path(filename).name).split("_")[-1]

            plt.plot(history['val_accuracy'], label=new_label, linewidth=3)
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('exp_cam_val_acc.png')
    # plt.show()
    plt.clf()


plot_loss()
plot_val()
