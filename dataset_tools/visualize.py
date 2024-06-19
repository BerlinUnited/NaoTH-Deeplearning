"""
 show images from dataset with annotations for different datasets (naoth, b-human, etc)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pickle
from pathlib import Path
import matplotlib.patches as ptc
from common_tools.main import get_data_root

DATA_DIR = Path(get_data_root()) / "data_balldetection"


class BhumanVisualizer(object):
    def __init__(self, index_start):
        self.ind = index_start + 1

        self.get_b_human_data()
        self.init_plot()

    def get_b_human_data(self):
        f = h5py.File(f"{DATA_DIR}/bhuman/b-alls-2019.hdf5", "r")

        # TODO combine the data and
        self.negative_data = np.array(f.get("negatives/data"))
        self.positive_data = np.array(f.get("positives/data"))
        self.negative_labels = np.array(f.get("negatives/labels"))
        self.positive_labels = np.array(f.get("positives/labels"))

    def init_plot(self):
        # Adjust bottom to make room for Buttons
        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        start_value = 0

        self.subplot_list = list()

        for plot_idx, i in enumerate(range(start_value, start_value + columns * rows)):
            img = self.positive_data[i]
            a = fig.add_subplot(rows, columns, plot_idx + 1)
            self.subplot_list.append(a)
            current_label = self.positive_labels[i]

            circle1 = plt.Circle((current_label[1], current_label[2]), current_label[3], color="r", alpha=0.7)
            ax = fig.gca()
            ax.add_patch(circle1)
            plt.imshow(img, cmap="gray")

        # Connect to a "switch" Button, setting its left, top, width, and height
        axswitch = plt.axes([0.40, 0.01, 0.2, 0.05])
        bswitch = Button(axswitch, "Next")
        bswitch.on_clicked(self.next)
        plt.show()

    # This function is called when bswitch is clicked
    def next(self, event):
        for idx, subplot in enumerate(self.subplot_list):
            subplot.cla()
            # get new image data
            img = self.positive_data[self.ind]
            current_label = self.positive_labels[self.ind]
            circle1 = plt.Circle((current_label[1], current_label[2]), current_label[3], color="r", alpha=0.7)

            subplot.add_patch(circle1)
            subplot.imshow(img, cmap="gray")

            self.ind += 1
            plt.draw()


class TK3Visualizer(object):
    def __init__(self, index_start):
        self.ind = index_start + 1

        imgdb_path = str(DATA_DIR / "naoth/tk03_natural_detection.pkl")
        with open(imgdb_path, "rb") as f:
            mean = pickle.load(f)
            self.x = pickle.load(f)
            self.y = pickle.load(f)

        print(sum(self.y[:, -1]))
        print(len(self.y) - sum(self.y[:, -1]))
        self.init_plot()

    def init_plot(self):
        # Adjust bottom to make room for Buttons
        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        start_value = 0

        self.subplot_list = list()

        for plot_idx, i in enumerate(range(start_value, start_value + columns * rows)):
            img = self.x[i]
            a = fig.add_subplot(rows, columns, plot_idx + 1)
            self.subplot_list.append(a)
            current_label = self.y[i]

            # only draw circle if there is actually a circle annotation in the groundtruth
            if current_label[3] == 1.0:
                # format for plotting is (xy), r
                circle1 = plt.Circle(
                    (current_label[1] * 16, current_label[2] * 16), current_label[0] * 16, color="r", alpha=0.5
                )
                ax = fig.gca()
                # ax.add_patch(circle1)
            plt.imshow(img, cmap="gray")

        # Connect to a "switch" Button, setting its left, top, width, and height
        axswitch = plt.axes([0.40, 0.01, 0.2, 0.05])
        bswitch = Button(axswitch, "Next")
        bswitch.on_clicked(self.next)
        plt.show()

    # This function is called when bswitch is clicked
    def next(self, event):
        for idx, subplot in enumerate(self.subplot_list):
            subplot.cla()
            # get new image data
            img = self.x[self.ind]
            current_label = self.y[self.ind]

            # only draw circle if there is actually a circle annotation in the groundtruth
            if current_label[3] == 1.0:
                circle1 = plt.Circle(
                    (current_label[1] * 16, current_label[2] * 16), current_label[0] * 16, color="r", alpha=0.5
                )
                # subplot.add_patch(circle1)

            subplot.imshow(img, cmap="gray")

            self.ind += 1
            plt.draw()


class BhumanVisualizer2(object):
    def __init__(self, index_start):
        self.ind = index_start + 1
        self.patch_size = 32

        imgdb_path = str(DATA_DIR / "bhuman_classification.pkl")
        with open(imgdb_path, "rb") as f:
            mean = pickle.load(f)
            self.x = pickle.load(f)
            self.y = pickle.load(f)

        # set patch size to either 32 or 16
        self.patch_size = self.x[0].shape[0]

        self.init_plot()

    def init_plot(self):
        # Adjust bottom to make room for Buttons
        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        start_value = 0

        self.subplot_list = list()

        for plot_idx, i in enumerate(range(start_value, start_value + columns * rows)):
            img = self.x[i]
            a = fig.add_subplot(rows, columns, plot_idx + 1)
            self.subplot_list.append(a)
            current_label = self.y[i]

            # only draw patch overay if there is actually a positive annotation in the groundtruth
            if current_label == 1:
                rec = ptc.Rectangle((0, 0), width=self.patch_size + 1, height=self.patch_size + 1, alpha=0.3)
                ax = fig.gca()
                ax.add_patch(rec)

            plt.imshow(img, cmap="gray")

        # Connect to a "switch" Button, setting its left, top, width, and height
        axswitch = plt.axes([0.40, 0.07, 0.2, 0.05])
        bswitch = Button(axswitch, "Next")
        bswitch.on_clicked(self.next)
        plt.show()

    # This function is called when bswitch is clicked
    def next(self, event):
        for idx, subplot in enumerate(self.subplot_list):
            subplot.cla()
            # get new image data
            img = self.x[self.ind]
            current_label = self.y[self.ind]

            # only draw patch overay if there is actually a positive annotation in the groundtruth
            if current_label == 1:
                rec = ptc.Rectangle((0, 0), width=self.patch_size + 1, height=self.patch_size + 1, alpha=0.3)
                subplot.add_patch(rec)

            subplot.imshow(img, cmap="gray")

            self.ind += 1
            plt.draw()


if __name__ == "__main__":
    # BhumanVisualizer(0)
    TK3Visualizer(0)

    # bhuman visualizer2 which actually uses the created pkl files for visualization
    # BhumanVisualizer2(100)
