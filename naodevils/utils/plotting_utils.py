import numpy as np
import pickle
from enum import Enum

import matplotlib
from matplotlib.figure import figaspect
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from textwrap import wrap

from matplotlib import rc

from utils import *

if DEBUGGING:
    # matplotlib.use('Qt5Agg')
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')


PLOT_FORMAT = "png"

COLOR_LABEL = {
    "robot": "tab:blue",
    "ball": "tab:orange",
    "penalty cross": "limegreen",
    "robot reference": "darkblue",
    "ball reference": "sienna",
    "penalty cross reference": "darkgreen",
}

MARKER_LABEL = {
    "robot": ".",
    "ball": ".",
    "penalty cross": ".",
    "robot reference": "x",
    "ball reference": "x",
    "penalty cross reference": "x",
}

MARKER_VALUE = {
    "robot": "o",
    "ball": "o",
    "penalty cross": "o",
    "robot reference": "x",
    "ball reference": "x",
    "penalty cross reference": "x",
}

class TYPE(Enum):
    ACC = 0
    DISTANCE = 1
    ROC = 2

def scale_axis(ax, ratio):
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()

    if ax.get_xscale() == "log":
        xrange = np.log(xvals[1]) - np.log(xvals[0])
    else:
        xrange = xvals[1] - xvals[0]

    if ax.get_yscale() == "log":
        yrange = np.log(yvals[1]) - np.log(yvals[0])
    else:
        yrange = yvals[1] - yvals[0]

    if ax.get_xscale() != "log" and ax.get_yscale() != "log":
        try:
            ax.set_aspect(ratio * abs(xrange / yrange)) #, adjustable='box')
        except Exception as e:
            print(str(e))

def set_ticks_and_locator(axis, axis_name="x", major_locator=0.2, minor_locator=0.05, limits_x=[-0.03, 1.03], limits_y=[-0.03, 1.03]):
    if axis_name == "x":
        ax = axis.xaxis
        axis.set_xlim(limits_x)
    else:
        ax = axis.yaxis
        axis.set_ylim(limits_y)

    ax.set_major_locator(MultipleLocator(major_locator))
    if major_locator * 10 > 1:
        ax.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_minor_locator(MultipleLocator(minor_locator))

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def plot_pr_curve_subplot(data, filename, save=True):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    fig.subplots_adjust(top=0.2, left=0.7)
    ax2.axis('off')

    fig.suptitle("\n".join(wrap(os.path.basename(filename), 60)))
    ax1.set_xlim([0.0, 1.1])
    ax1.set_ylim([0.0, 1.1])
    ax1.set_aspect("equal", adjustable='box')
    ax1.set_xticks(np.arange(0, 1.1, step=0.1), minor=False)
    ax1.set_yticks(np.arange(0, 1.1, step=0.1), minor=False)
    ax1.set(xlabel='Precision', ylabel='Recall')
    ax1.grid(True)

    f_score_text = ""
    for label_idx, data_for_label in enumerate(data):
        ax1.plot(data_for_label["precisions"], data_for_label["recalls"], marker='.', label=data_for_label["label"])
        f_score_text += "" if f_score_text == "" else "\n\n"
        f_score_text += "%s:\n  F%s-score(micro): %.3f\n  F%s-score(macro): %.3f" % (data_for_label["label"], data_for_label["f_score_beta"], data_for_label["avg_f_score_micro"], data_for_label["f_score_beta"], data_for_label["avg_f_score_macro"])
    ax1.legend(loc="upper right")
    fig.text(0.63, 0.55, f_score_text, bbox=dict(facecolor='white', alpha=1.0), verticalalignment='center', horizontalalignment='left')

    if save:
        fig.tight_layout()
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
    else:
        fig.show()
    plt.close(fig)

def plot_confusion_matrix(data, filename, title="", latex=False, iou_threshold=0.5, save=True, label=""):
    f_beta = FLAGS.f_beta
    axis3_0_type = TYPE.ACC
    size = 45
    color_map = COLOR_LABEL

    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(0.837)
    fig, axs = plt.subplots(4, 3, figsize=(w, h), dpi=300, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1, 1, 1]})
    fig.subplots_adjust(left=0.085, bottom=0.05, right=0.99, top=0.95, wspace=0.59, hspace=0.30)

    plot_true_positives(axs[0, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    plot_false_positives(axs[0, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    plot_precision(axs[0, 2], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)

    plot_false_negatives(axs[1, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    legends = plot_true_negatives(axs[1, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    #axs[1, 1].legend(handles=legends, loc='lower right', handlelength=1.0, handletextpad=0.5)
    plot_mAP(axs[1, 2], data, legends, position=[0.47, 0.45], label=label, latex=latex, size=size, alpha=1.0, color_map=color_map, iou_threshold=0.5)

    plot_recalls(axs[2, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    plot_avg_iou(axs[2, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    plot_pr_curve(axs[2, 2], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)

    if axis3_0_type == TYPE.ACC:
        plot_accuracy(axs[3, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    elif axis3_0_type == TYPE.ROC:
        plot_roc(axs[3, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    elif axis3_0_type == TYPE.DISTANCE:
        plot_distances(axs[3, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
    plot_f_beta(axs[3, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map, f_beta=f_beta)
    plot_mcc(axs[3, 2], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)

    ratio = 1.0
    for ax in axs.flat:
        ax.grid(True, which='major')
        scale_axis(ax, ratio)

    loc_title = os.path.basename(filename)[:-4] if title == "" else title
    if latex:
        loc_title = loc_title.replace("_", r"\_")
        fig.text(0.53, 0.97, r"\textbf{" + loc_title + r"}", ha='center', size=17)
    else:
        loc_title = loc_title.replace("_", " ")
        fig.text(0.53, 0.97, loc_title, ha='center', size=17)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT)
        plt.close(fig)
    else:
        fig.show()

def plot_comparison(data, filename, title="", latex=False, iou_threshold=0.5, save=True, label=""):
    f_beta = FLAGS.f_beta
    size = 45
    color_map = COLOR_LABEL

    if len(data) == 6:
        matplotlib.rcParams["figure.figsize"] = [9, 12]
        h, w = figaspect(0.63)
    else:
        matplotlib.rcParams["figure.figsize"] = [10, 10]
        h, w = figaspect(0.84)
    fig, axs = plt.subplots(len(data), 3, figsize=(w, h), dpi=300, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1 for _ in range(len(data))]})
    if len(data) == 6:
        fig.subplots_adjust(left=0.06, bottom=0.04, right=0.97, top=0.97, wspace=0.45, hspace=0.30)
    else:
        fig.subplots_adjust(left=0.08, bottom=0.04, right=0.97, top=0.97, wspace=0.45, hspace=0.20)

    for label_idx in range(int(len(data)/2)):
        label = data[label_idx]["label"]

        plot_recalls(axs[label_idx*2, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
        legends = plot_precision(axs[label_idx*2, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
        plot_mAP(axs[label_idx*2, 2], data, legends, position=[0.5, 0.5], label=label, latex=latex, size=size, alpha=1.0, color_map=color_map, iou_threshold=0.5)
        #plot_pr_curve(axs[2, 2], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
        plot_accuracy(axs[label_idx*2 + 1, 0], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)
        plot_f_beta(axs[label_idx*2 + 1, 1], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map, f_beta=f_beta)
        plot_mcc(axs[label_idx*2 + 1, 2], data, label=label, latex=latex, size=size, alpha=1.0, color_map=color_map)

    ratio = 1.0
    for ax in axs.flat:
        ax.grid(True, which='major')
        scale_axis(ax, ratio)

    loc_title = os.path.basename(filename)[:-4] if title == "" else title
    if latex:
        loc_title = loc_title.replace("_", r"\_")
        fig.text(0.53, 0.97, r"\textbf{" + loc_title + r"}", ha='center', size=17)
    else:
        loc_title = loc_title.replace("_", " ")
        fig.text(0.50, 0.98, loc_title, ha='center', size=17)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT)
        plt.close(fig)
    else:
        fig.show()

##########################################################################################################
##########################################################################################################
##########################################################################################################

def plot_true_positives(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    if latex:
        axis.set(xlabel='Threshold', ylabel=r'\#True Positives')
    else:
        axis.set(xlabel='Threshold', ylabel='#True Positives')
    axis.text(0.5, 0.5, 'TP',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        line, = axis.plot(data_for_label["thresholds"], data_for_label["tp"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_false_positives(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    if latex:
        axis.set(xlabel='Threshold', ylabel=r'\#False Positives')
    else:
        axis.set(xlabel='Threshold', ylabel='#False Positives')
    axis.text(0.5, 0.5, 'FP',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        line, = axis.semilogy(data_for_label["thresholds"], (np.array(data_for_label["fp"]) + 1), marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        #line, = axis.plot(data_for_label["thresholds"], (np.array(data_for_label["fp"]) + 1), marker=MARKER_MAP[index], label=data_for_label["label"], color=color_map[index], alpha=alpha)
        legends.append(line)

    if np.array(data_for_label["fp"]).max() > 5000:
        axis.set_yticks([1, 11, 101, 1001, 10001])
        axis.set_yticklabels([0, 10, 100, 1000, 10000])
        axis.set_yticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 20000], minor=True)
        axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        axis.set_yticks([1, 11, 101, 1001])
        axis.set_yticklabels([0, 10, 100, 1000])
        axis.set_yticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000, 4000], minor=True)
        axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # axis.minorticks_on()
    # axis.tick_params(which='both')
    # locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=3)
    # axis.yaxis.set_minor_locator(locmin)
    # axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # # axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return legends

def plot_true_negatives(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    if latex:
        axis.set(xlabel='Threshold', ylabel=r'\#True Negatives')
    else:
        axis.set(xlabel='Threshold', ylabel='#True Negatives')
    axis.text(0.5, 0.5, 'TN',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    #axis.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        line, = axis.plot(data_for_label["thresholds"], data_for_label["tn"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_false_negatives(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    if latex:
        axis.set(xlabel='Threshold', ylabel=r'\#False Negatives')
    else:
        axis.set(xlabel='Threshold', ylabel='#False Negatives')
    axis.text(0.5, 0.5, 'FN',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        line, = axis.plot(data_for_label["thresholds"], data_for_label["fn"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_precision(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='Precision')
    axis.text(0.5, 0.5, 'Pre',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        line, = axis.plot(data_for_label["thresholds"], data_for_label["precisions"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_recalls(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='Recall')
    axis.text(0.5, 0.5, 'Rec',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue

        line, = axis.plot(data_for_label["thresholds"], data_for_label["recalls"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_pr_curve(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set_title('PR-Curve')
    axis.set(xlabel='Recall', ylabel='Precision')
    axis.text(0.5, 0.5, 'P-R',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue

        r = data_for_label["recalls"]
        r.insert(0, data_for_label["recalls"][0])
        r.append(0.0)
        p = data_for_label["precisions"]
        p.insert(0, 0.0)
        p.append(data_for_label["precisions"][-1])

        line, = axis.plot(r, p, marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_avg_iou(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='Avg. IoU')
    axis.text(0.5, 0.5, 'IoU',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y", limits_y=[0.485, 1.015], major_locator=0.1, minor_locator=0.05)
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        #line, = axis.errorbar(data_for_label["thresholds"], data_for_label["average_IoUs"], yerr=data_for_label["std_IoUs"], marker=MARKER_MAP[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], capsize=2, elinewidth=1, alpha=alpha)
        line, = axis.plot(data_for_label["thresholds"], data_for_label["average_IoUs"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_roc(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set_title('ROC')
    axis.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    axis.text(0.5, 0.5, 'ROC',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        fpr = []
        tpr = data_for_label["recalls"].copy()
        for i, t in enumerate(data_for_label["thresholds"]):
            fpr.append(data_for_label["fp"][i] / max((data_for_label["fp"][i] + data_for_label["tn"][i]), 1e-8))
        tpr.insert(0, 1.0)
        tpr.append(0.0)
        fpr.insert(0, 1.0)
        fpr.append(0.0)

        line, = axis.plot(fpr, tpr, marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        legends.append(line)
    return legends

def plot_distances(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='Avg. Distance')
    axis.text(0.5, 0.5, 'Dist',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y", major_locator=2000, minor_locator=1000, limits_y=[-300, 8300])
    ################################################################
    legends = []
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        baseline = []
        baseline.extend(data_for_label["detected_distances"][0])
        baseline.extend(data_for_label["not_detected_distances"][0])
        baseline = np.array(baseline)
        baseline_means = []
        baseline_std = []
        detected_distances_means = []
        detected_distances_std = []
        for i, t in enumerate(data_for_label["thresholds"]):
            detected_distances = []
            detected_distances.extend(data_for_label["detected_distances"][i])
            detected_distances = np.array(detected_distances)
            detected_distances_means.append(detected_distances.mean())
            detected_distances_std.append(detected_distances.std())
            baseline_means.append(baseline.mean())
            baseline_std.append(baseline.std())

        detected_distances_means = np.array(detected_distances_means)
        detected_distances_std = np.array(detected_distances_std)
        baseline_means = np.array(baseline_means)
        baseline_std = np.array(baseline_std)
        # axs[3, 0].plot(data_for_label["thresholds"], baseline_means, marker='.', label=data_for_label["label"], color='black')
        line, = axis.plot(data_for_label["thresholds"], detected_distances_means, marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        axis.fill_between(data_for_label["thresholds"], detected_distances_means - detected_distances_std, detected_distances_means + detected_distances_std, color=color_map[data_for_label["label"]], alpha=0.2)

        legends.append(line)

    return legends

def plot_accuracy(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='Accuracy')
    axis.text(0.5, 0.5, 'Acc',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y", major_locator=0.02, minor_locator=0.002, limits_y=[0.897, 1.003])
    ################################################################
    legends = []
    acc_legends = []
    mean_acc_score = 0
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue
        max_acc_score_index = np.array(data_for_label["acc_scores"]).argmax(axis=0)
        mean_acc_score += np.array(data_for_label["acc_scores"]).mean()
        max_acc_score = (data_for_label["thresholds"][max_acc_score_index], data_for_label["acc_scores"][max_acc_score_index])

        line, = axis.plot(data_for_label["thresholds"], data_for_label["acc_scores"], marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        axis.plot(max_acc_score[0], max_acc_score[1], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none',  markersize=12, label=data_for_label["label"], color=color_map[data_for_label["label"]])

        #patch = mpatches.Patch(color=color_map[data_for_label["label"]], label="%.4f" % (max_acc_score[1]))
        patch = Line2D([0], [0], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none',  markersize=12, color='w', markeredgecolor=color_map[data_for_label["label"]], label="%.4f" % (max_acc_score[1]))
        acc_legends.append(patch)
        legends.append(line)

    if mean_acc_score / 2.0 < 0.5:
        axis.legend(handles=acc_legends, loc='upper center')
    else:
        axis.legend(handles=acc_legends, loc='lower center')

    return legends

def plot_f_beta(axis, data, f_beta=0.5, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    # axis.set(xlabel='Threshold', ylabel='F%s-Score' % data[0]["f_score_beta"])
    # axis.text(0.5, 0.5, 'F%s' % data[0]["f_score_beta"],
    axis.set(xlabel='Threshold', ylabel='F%.3f-Score' % f_beta)
    axis.text(0.5, 0.5, 'F%.3f' % f_beta,
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    f_beta_legends = []
    mean_f_beta_score = 0
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue

        f_beta_score = []
        for i, t in enumerate(data_for_label["thresholds"]):
            oben = (1 + f_beta ** 2) * data_for_label["recalls"][i] * data_for_label["precisions"][i]
            unten = data_for_label["recalls"][i] + (f_beta ** 2 * data_for_label["precisions"][i])
            f_beta_score.append(oben / max(unten, 1e-8))
        max_f_beta_score_index = np.array(f_beta_score).argmax(axis=0)
        mean_f_beta_score += np.array(f_beta_score).mean()
        max_f_beta_score = (data_for_label["thresholds"][max_f_beta_score_index], f_beta_score[max_f_beta_score_index])

        line, = axis.plot(data_for_label["thresholds"], f_beta_score, marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        axis.plot(max_f_beta_score[0], max_f_beta_score[1], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none',  markersize=12, label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)

        #patch = mpatches.Patch(color=color_map[data_for_label["label"]], label="%.4f" % (max_f_beta_score[1]))
        patch = Line2D([0], [0], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none',  markersize=12, color='w', markeredgecolor=color_map[data_for_label["label"]], label="%.4f" % (max_f_beta_score[1]))
        f_beta_legends.append(patch)
        legends.append(line)

    if mean_f_beta_score / 2.0 < 0.5:
        axis.legend(handles=f_beta_legends, loc='upper center')
    else:
        axis.legend(handles=f_beta_legends, loc='lower center')

    return legends

def plot_mcc(axis, data, label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set(xlabel='Threshold', ylabel='MCC')
    axis.text(0.5, 0.5, 'MCC',
                   ha='center', va='center',
                   fontsize=size, color='gray',
                   alpha=0.2, transform=axis.transAxes)
    set_ticks_and_locator(axis, axis_name="x", minor_locator=0.1)
    set_ticks_and_locator(axis, axis_name="y")
    ################################################################
    legends = []
    mcc_legends = []
    mean_mcc_score = 0
    for label_idx, data_for_label in enumerate(data):
        if label != "" and not data_for_label["label"].startswith(label):
            continue

        mcc_score = np.array(data_for_label["mcc_scores"])
        max_mcc_score_index = mcc_score.argmax(axis=0)
        mean_mcc_score += mcc_score.mean()
        max_mcc_score = (data_for_label["thresholds"][max_mcc_score_index], mcc_score[max_mcc_score_index])

        line, = axis.plot(data_for_label["thresholds"], mcc_score, marker=MARKER_LABEL[data_for_label["label"]], label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)
        axis.plot(max_mcc_score[0], max_mcc_score[1], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none', markersize=12, label=data_for_label["label"], color=color_map[data_for_label["label"]], alpha=alpha)

        #patch = mpatches.Patch(color=color_map[data_for_label["label"]], label="%.4f" % (max_mcc_score[1]))
        patch = Line2D([0], [0], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none', markersize=12, color='w', markeredgecolor=color_map[data_for_label["label"]], label="%.4f" % (max_mcc_score[1]))
        mcc_legends.append(patch)
        legends.append(line)

    if mean_mcc_score / 2.0 < 0.5:
        axis.legend(handles=mcc_legends, loc='upper center')
    else:
        axis.legend(handles=mcc_legends, loc='lower center')

    return legends

def plot_mAP(axis, data, legends, iou_threshold=0.5, position=[0.5, 0.5], label="", latex=False, size=45, alpha=1.0, color_map=COLOR_LABEL):
    ################################################################
    axis.set_title('')
    axis.axis('off')
    ################################################################
    score_text = r""
    mAP = []

    full_legends = []
    for legend in legends:
        for label_idx, data_for_label in enumerate(data):
            if data_for_label["label"] == legend._label:
                l = Line2D([0], [0])
                l.update_from(legend)
                l._label = l._label + ":"
                full_legends.append(l)
                # rectangle = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"]))
                # full_legends.append(rectangle)
                line = Line2D([0], [0], marker=MARKER_VALUE[data_for_label["label"]], fillstyle='none', markersize=12, color='w', markeredgecolor=color_map[data_for_label["label"]], label=r"    AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"]))
                full_legends.append(line)
                mAP.append(data_for_label["avp"])
            else:
                continue

    if label == "":
        rectangle = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        full_legends.append(rectangle)
        rectangle = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=r"mAP@%.2f: %.3f" % (iou_threshold, np.mean(mAP)))
        full_legends.append(rectangle)

    axis.legend(handles=full_legends, loc='center', handlelength=1.0, handletextpad=0.5, bbox_to_anchor=(position[0], position[1]))

    # for label_idx, data_for_label in enumerate(data):
    #     if label != "" and not data_for_label["label"].startswith(label):
    #         continue
    #
    #     if score_text != r"":
    #         score_text += "\n"
    #
    #     # if label != "":
    #     #     if latex:
    #     #         score_text += r"\textbf{%s}:" % (data_for_label["label"][len(label) + 1:])
    #     #         score_text += "\n"
    #     #         score_text += r"AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"])
    #     #     else:
    #     #         score_text += "%s:" % (data_for_label["label"][len(label) + 1:])
    #     #         score_text += "\n"
    #     #         score_text += "AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"])
    #     # else:
    #     if latex:
    #         score_text += r"\textbf{%s}:" % (data_for_label["label"])
    #         score_text += "\n"
    #         score_text += r"  AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"])
    #     else:
    #         score_text += "%s:" % (data_for_label["label"])
    #         score_text += "\n"
    #         score_text += "  AP@%.2f: %.3f" % (iou_threshold, data_for_label["avp"])
    #     mAP.append(data_for_label["avp"])

    # if label == "":
    #     if latex:
    #         score_text += "\n"
    #         score_text += r"\_\_\_\_"
    #         score_text += "\n"
    #         score_text += r"mAP@%.2f: %.3f" % (iou_threshold, np.mean(mAP))
    #     else:
    #         score_text += "\n"
    #         score_text += "\n"
    #         score_text += "mAP@%.2f: %.3f" % (iou_threshold, np.mean(mAP))
    #
    # # axis.text(0.5, 0.9, r"" + f, ha='center', va='center', transform=legend_axis.transAxes)
    # axis.text(position[0], position[1], score_text, bbox=dict(facecolor='white', alpha=1.0, ), ha='center', va='center', ma='left', transform=axis.transAxes)

##########################################################################################################
##########################################################################################################
##########################################################################################################

def plot_distance_histogram(dataset, filename="dataset_distances.png", title=None, latex=False, left="Training", right="Validation", side_by_side=True, save=True):
    dataset_train = dataset["train"]
    dataset_valid = dataset["valid"]
    n_bins = 15
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(1.8)
    fig, axs = plt.subplots(1, 2, figsize=(w, h), sharey=True, dpi=150, tight_layout=True)
    ###################################################################################################
    axs[0].set_title(left)
    axs[0].set(ylabel='Frequency')
    ###################################################################################################
    axs[1].set_title(right)
    # axs[1].set(xlabel='Distance [m]')
    ###################################################################################################
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(True, which="both")

    training_hist = []
    validation_hist = []
    minimum = np.inf
    maximum = 0
    colors = []
    for i, label in enumerate(dataset["label_names"]):
        training_hist.append(dataset_train[label][:,8]/1000.0)
        validation_hist.append(dataset_valid[label][:,8]/1000.0)
        colors.append(COLOR_LABEL[label])
        minimum = min(minimum, training_hist[-1].min(), validation_hist[-1].min())
        maximum = max(maximum, training_hist[-1].max(), validation_hist[-1].max())

    set_ticks_and_locator(axs[0], axis_name="x", major_locator=1, minor_locator=0.5, limits_x=[-0.25, np.ceil(maximum)+0.25])
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    set_ticks_and_locator(axs[1], axis_name="x", major_locator=1, minor_locator=0.5, limits_x=[-0.25, np.ceil(maximum)+0.25])
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    args = {}
    args["bins"] = np.arange(0.0, np.ceil(maximum+0.5),1.5) #n_bins
    args["range"] = (0.0, np.ceil(maximum))
    args["log"] = True
    args["align"] = "right"

    if side_by_side:
        args["rwidth"] = 1.5
        args["alpha"] = 0.9
        axs[0].hist(training_hist, label=dataset["label_names"], color=colors, **args)
        axs[1].hist(validation_hist, label=dataset["label_names"], color=colors, **args)
    else:
        args["alpha"] = 0.9
        for i in range(len(training_hist)):
            axs[0].hist(training_hist[i], label=dataset["label_names"][i], color=colors[i], **args)
            axs[1].hist(validation_hist[i], label=dataset["label_names"][i], color=colors[i], **args)

    #axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    fig.text(0.53, -0.01, 'Distance to NAO [m]', ha='center')
    if title:
        if latex:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=18)
        else:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=20)

    ratio = 0.4
    for ax in axs.flat:
        scale_axis(ax, ratio)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
        plt.close(fig)
    else:
        # fig.tight_layout()
        fig.show()

def plot_visibility_histogram(dataset, filename="dataset_visibility.png", title=None, latex=False, left="Training", right="Validation", side_by_side=True, save=True):
    dataset_train = dataset["train"]
    dataset_valid = dataset["valid"]
    n_bins = 6
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(1.0)

    if latex:
        vis_xticks = [r'', r'100\%', r'$>$75\%', r'$>$50\%', r'$>$25\%', r'$>$0\%', r'0\%']
    else:
        vis_xticks = [r'', r'100%', r'$>$75%', r'$>$50%', r'$>$25%', r'$>$0%', r'0%']

    xticks = [-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    fig, axs = plt.subplots(2, 2, figsize=(w, h), sharey=True, dpi=150, tight_layout=True)
    ####################################################################################################
    axs[0,0].set_title(left)
    axs[0,0].set(ylabel='Frequency')
    set_ticks_and_locator(axs[0,0], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[0].yaxis.set_minor_locator(MultipleLocator(100))
    #axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[0,0].set_xticklabels(vis_xticks)
    axs[0,0].set_xticks(xticks)
    axs[0,0].xaxis.set_tick_params(rotation=45)
    ####################################################################################################
    axs[0,1].set_title(right)
    # axs[1].set(xlabel='Visibility Level')
    set_ticks_and_locator(axs[0,1], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[1].yaxis.set_minor_locator(MultipleLocator(50))
    #axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[0,1].set_xticklabels(vis_xticks)
    axs[0, 1].set_xticks(xticks)
    axs[0,1].xaxis.set_tick_params(rotation=45)
    ####################################################################################################
    axs[1,0].set(ylabel='Frequency')
    set_ticks_and_locator(axs[1,1], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[0].yaxis.set_minor_locator(MultipleLocator(100))
    #axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[1,0].set_xticklabels(vis_xticks)
    axs[1, 0].set_xticks(xticks)
    axs[1,0].xaxis.set_tick_params(rotation=45)
    ####################################################################################################
    # axs[1].set(xlabel='Visibility Level')
    set_ticks_and_locator(axs[1,1], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[1].yaxis.set_minor_locator(MultipleLocator(50))
    #axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[1,1].set_xticklabels(vis_xticks)
    axs[1, 1].set_xticks(xticks)
    axs[1,1].xaxis.set_tick_params(rotation=45)
    ####################################################################################################
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(True, which="both")

    training_hist = []
    training_hist_concealed = []
    validation_hist = []
    validation_hist_concealed = []
    colors = []
    colors_concealed = []
    for i, label in enumerate(dataset["label_names"]):
        train_concealed = dataset_train[label][:, 5][dataset_train[label][:, 6] == 1.0]
        train_not_concealed = dataset_train[label][:, 5][dataset_train[label][:, 6] == 0.0]
        training_hist_concealed.append(train_concealed)
        training_hist.append(train_not_concealed)

        valid_concealed = dataset_valid[label][:, 5][dataset_valid[label][:, 6] == 1.0]
        valid_not_concealed = dataset_valid[label][:, 5][dataset_valid[label][:, 6] == 0.0]
        validation_hist_concealed.append(valid_concealed)
        validation_hist.append(valid_not_concealed)

        # validation_hist.append(dataset_valid[label][:, 5])
        # validation_hist.append([l for l in true_positives[label_index][vis_lvl] if l[5] > threshold])
        colors.append(COLOR_LABEL[label])

    args = {}
    args["bins"] = n_bins
    args["range"] = (0, n_bins)
    args["log"] = True
    args["align"] = "left"

    res_train = None
    res_valid = None

    args["alpha"] = 0.9
    res_train = axs[0,0].hist(training_hist, label=dataset["label_names"], color=colors, **args)
    res_train = axs[1,0].hist(training_hist_concealed, label=["concealed " + s for s in dataset["label_names"]], color=colors, **args)
    res_valid = axs[0,1].hist(validation_hist, label=dataset["label_names"], color=colors, **args)
    res_valid = axs[1,1].hist(validation_hist_concealed, label=["concealed " + s for s in dataset["label_names"]], color=colors, **args)

    # if side_by_side:
    #     args["rwidth"] = 1.5
    #     args["alpha"] = 0.9
    #     res_train = axs[0].hist(training_hist, label=dataset["label_names"], color=colors, **args)
    #     res_valid = axs[1].hist(validation_hist, label=dataset["label_names"], color=colors, **args)
    # else:
    #     args["alpha"] = 0.9
    #     for i in range(len(training_hist)):
    #         res_train = axs[0].hist(training_hist[i], label=dataset["label_names"][i], color=colors[i], **args)
    #         res_valid = axs[1].hist(validation_hist[i], label=dataset["label_names"][i], color=colors[i], **args)

    #axs[0].legend(loc='upper right')
    axs[0, 1].legend(loc='upper right')
    axs[1, 1].legend(loc='upper right')
    fig.text(0.53, 0.0, 'Visibility', ha='center')
    if title:
        if latex:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=18)
        else:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=20)

    ratio = 2.0
    for ax in axs.flat:
        scale_axis(ax, ratio)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
        plt.close(fig)
    else:
        # fig.tight_layout()
        fig.show()

def plot_blurriness_histogram(dataset, filename="dataset_blurriness.png", title=None, latex=False, left="Training", right="Validation", side_by_side=True, save=True):
    dataset_train = dataset["train"]
    dataset_valid = dataset["valid"]
    n_bins = 10
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(1.8)
    fig, axs = plt.subplots(1, 2, figsize=(w, h), sharey=True, dpi=150, tight_layout=True)
    ####################################################################################################
    axs[0].set_title(left)
    axs[0].set(ylabel='Frequency')
    ####################################################################################################
    axs[1].set_title(right)
    # axs[1].set(xlabel='Blurriness')
    ####################################################################################################
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(True, which="both")

    training_hist = []
    validation_hist = []
    minimum = np.inf
    maximum = 0
    colors = []
    for i, label in enumerate(dataset["label_names"]):
        training_hist.append(dataset_train[label][:, 7])
        validation_hist.append(dataset_valid[label][:, 7])
        colors.append(COLOR_LABEL[label])
        minimum = min(minimum, training_hist[-1].min(), validation_hist[-1].min())
        maximum = max(maximum, training_hist[-1].max(), validation_hist[-1].max())

    set_ticks_and_locator(axs[0], axis_name="x", major_locator=400, minor_locator=200, limits_x=[-100, np.ceil(maximum)+100])
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    set_ticks_and_locator(axs[1], axis_name="x", major_locator=400, minor_locator=200, limits_x=[-100, np.ceil(maximum)+100])
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    args = {}
    args["bins"] = np.arange(0.0, np.ceil(maximum),200) #n_bins#n_bins
    args["range"] = (minimum, maximum)
    args["log"] = True
    args["align"] = "left"

    if side_by_side:
        args["rwidth"] = 1.5
        args["alpha"] = 0.9
        axs[0].hist(training_hist, label=dataset["label_names"], color=colors, **args)
        axs[1].hist(validation_hist, label=dataset["label_names"], color=colors, **args)
    else:
        args["alpha"] = 0.9
        for i in range(len(training_hist)):
            axs[0].hist(training_hist[i], label=dataset["label_names"][i], color=colors[i], **args)
            axs[1].hist(validation_hist[i], label=dataset["label_names"][i], color=colors[i], **args)

    #axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    fig.text(0.53, -0.01, 'Blurriness', ha='center')
    if title:
        if latex:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=18)
        else:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=20)

    ratio = 0.4
    for ax in axs.flat:
        scale_axis(ax, ratio)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
        plt.close(fig)
    else:
        # fig.tight_layout()
        fig.show()

def plot_size_histogram(dataset, image_size=(480,640), filename="dataset_size.png", latex=False, title=None, left="Training", right="Validation", side_by_side=True, save=True):
    dataset_train = dataset["train"]
    dataset_valid = dataset["valid"]
    n_bins = 10
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(1.8)
    fig, axs = plt.subplots(1, 2, figsize=(w, h), sharey=True, dpi=150, tight_layout=True)
    ####################################################################################################
    axs[0].set_title(left)
    axs[0].set(ylabel='Frequency')
    #set_ticks_and_locator(axs[0], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[0].yaxis.set_minor_locator(MultipleLocator(100))
    #axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ####################################################################################################
    axs[1].set_title(right)
    # axs[1].set(xlabel='Area [% of image]')
    #set_ticks_and_locator(axs[1], axis_name="x", major_locator=1, minor_locator=1, limits_x=[-0.5, (n_bins-0.5)])
    #axs[1].yaxis.set_minor_locator(MultipleLocator(50))
    #axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ####################################################################################################
    for ax in axs.flat:
        ax.set_axisbelow(True)
        ax.grid(True, which="both")
        ax.set_xscale('log')

    training_hist = []
    validation_hist = []
    minimum = np.inf
    maximum = 0
    colors = []
    for i, label in enumerate(dataset["label_names"]):
        training_hist.append(((dataset_train[label][:, 2] - dataset_train[label][:, 0]) * (dataset_train[label][:, 3] - dataset_train[label][:, 1])) * 100.0 / (image_size[0]*image_size[1]))
        validation_hist.append(((dataset_valid[label][:, 2] - dataset_valid[label][:, 0]) * (dataset_valid[label][:, 3] - dataset_valid[label][:, 1])) * 100.0 / (image_size[0]*image_size[1]))
        colors.append(COLOR_LABEL[label])
        minimum = min(minimum, training_hist[-1].min(), validation_hist[-1].min())
        maximum = max(maximum, training_hist[-1].max(), validation_hist[-1].max())

    #hist, bins = np.histogram(training_hist[0], bins=n_bins, range=(minimum, maximum))
    #logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    logbins = np.logspace(np.log10(0.05), np.log10(100), n_bins)
    # logbins = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    #            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    #            1, 2, 3, 4, 5, 6, 7, 8, 9,
    #            10, 20, 30, 40, 50, 60, 70, 80, 90,
    #            100]

    args = {}
    args["bins"] = logbins
    args["range"] = (logbins[0], logbins[-2])
    args["log"] = True
    args["align"] = "right"
    #args["histtype"] = 'barstacked'

    res_train = None
    res_valid = None

    if side_by_side:
        args["rwidth"] = 1.5
        args["alpha"] = 0.90
        res_train = axs[0].hist(training_hist, label=dataset["label_names"], color=colors, **args)
        res_valid = axs[1].hist(validation_hist, label=dataset["label_names"], color=colors, **args)
    else:
        args["alpha"] = 0.9
        for i in range(len(training_hist)):
            res_train = axs[0].hist(training_hist[i], label=dataset["label_names"][i], color=colors[i], **args)
            res_valid = axs[1].hist(validation_hist[i], label=dataset["label_names"][i], color=colors[i], **args)

    #axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    if latex:
        fig.text(0.53, -0.01, r'Area [\% of image]', ha='center')
        if title:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=18)
    else:
        fig.text(0.53, -0.01, 'Area [% of image]', ha='center')
        if title:
            fig.text(0.53, 1.0, title, ha='center', weight="bold", size=20)

    ratio = 0.5
    for ax in axs.flat:
        scale_axis(ax, ratio)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
        plt.close(fig)
    else:
        # fig.tight_layout()
        fig.show()

def plot_anchors(filename="anchors.png", latex=False, title=None, save=True):
    anchors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    upper_anchor_avg_iou = [0.3477, 0.5083, 0.5881, 0.6182, 0.6546, 0.6784, 0.7020, 0.7197, 0.7376, 0.7445]
    lower_anchor_avg_iou = [0.4231, 0.5843, 0.6546, 0.6784, 0.6956, 0.7264, 0.7478, 0.7658, 0.7790, 0.7873]

    matplotlib.rcParams["figure.figsize"] = [10, 10]
    h, w = figaspect(2.0)
    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=150, tight_layout=True)
    ####################################################################################################
    ax.set_title(title)
    ax.set(ylabel='Avg IoU', xlabel='# Anchors')
    set_ticks_and_locator(ax, axis_name="x", major_locator=1, minor_locator=1, limits_x=[0.7, 10.3])
    set_ticks_and_locator(ax, axis_name="y", major_locator=0.1, minor_locator=0.05, limits_y=[0.33, 0.80])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_axisbelow(True)
    ax.grid(True, which="both")

    ax.axvline(x=3, color='gray', linestyle='--', zorder=2, linewidth=2.5)
    ax.plot(anchors, upper_anchor_avg_iou, marker='.', label="upper", color="tab:blue")
    ax.plot(anchors, lower_anchor_avg_iou, marker='x', label="lower", color="tab:orange")
    #ax.vlines(x=3, ymin=0, ymax=lower_anchor_avg_iou[2], color='gray', zorder=2, linestyle='--')
    ax.legend(loc='lower right')

    # if latex:
    #     fig.text(0.53, -0.01, r'Area [\% of image]', ha='center')
    #     if title:
    #         fig.text(0.53, 1.0, title, ha='center', weight="bold", size=18)
    # else:
    #     fig.text(0.53, -0.01, 'Area [% of image]', ha='center')
    #     if title:
    #         fig.text(0.53, 1.0, title, ha='center', weight="bold", size=20)
    #
    ratio = 1.0
    scale_axis(ax, ratio)

    if save:
        fig.savefig(filename, dpi=fig._get_dpi(), format=PLOT_FORMAT, bbox_inches='tight')
        plt.close(fig)
    else:
        # fig.tight_layout()
        fig.show()

def plot_dataset():
    side_by_side = True
    dataset = pickle.load(open("c:\\NDevils\\YOLO\\DevilsKerasNeuralNetwork\\dataset_upper.pkl", "rb"))

    if use_latex:
        left = r"Upper Camera \textbf{Training} Dataset"
        right = r"Upper Camera \textbf{Validation} Dataset"
        distance_distribution = r"\textbf{Distance Distribution}"
        visibility_distribution = r"\textbf{Visibility Distribution}"
        sharpness_distribution = r"\textbf{Blurriness Distribution}"
        size_distribution = r"\textbf{Size Distribution}"
    else:
        left = r"Training Dataset"
        right = "Validation Dataset"
        # distance_distribution = "Distance Distribution for the Upper Camera"
        # visibility_distribution = "Visibility Distribution for the Upper Camera"
        # sharpness_distribution = "Blurriness Distribution for the Upper Camera"
        # size_distribution = "Size Distribution for the Upper Camera"
        distance_distribution = None
        visibility_distribution = None
        sharpness_distribution = None
        size_distribution = None

    plot_distance_histogram(dataset, title=distance_distribution, latex=use_latex, left=left, right=right,
                            side_by_side=side_by_side, filename="dataset_distances_upper." + PLOT_FORMAT)
    plot_visibility_histogram(dataset, title=visibility_distribution, latex=use_latex, left=left, right=right,
                              side_by_side=side_by_side, filename="dataset_visibility_upper." + PLOT_FORMAT)
    plot_blurriness_histogram(dataset, title=sharpness_distribution, latex=use_latex, left=left, right=right,
                              side_by_side=side_by_side, filename="dataset_blurriness_upper." + PLOT_FORMAT)
    plot_size_histogram(dataset, title=size_distribution, latex=use_latex, left=left, right=right,
                        side_by_side=side_by_side, image_size=(480, 640), filename="dataset_size_upper." + PLOT_FORMAT)

    dataset = pickle.load(open("c:\\NDevils\\YOLO\\DevilsKerasNeuralNetwork\\dataset_lower.pkl", "rb"))
    if use_latex:
        left = r"Lower Camera \textbf{Training} Dataset"
        right = r"Lower Camera \textbf{Validation} Dataset"
    else:
        left = "Training Dataset"
        right = "Validation Dataset"
        # distance_distribution = "Distance Distribution for the Lower Camera"
        # visibility_distribution = "Visibility Distribution for the Lower Camera"
        # sharpness_distribution = "Blurriness Distribution for the Lower Camera"
        # size_distribution = "Size Distribution for the Lower Camera"
        distance_distribution = None
        visibility_distribution = None
        sharpness_distribution = None
        size_distribution = None

    plot_distance_histogram(dataset, title=distance_distribution, latex=use_latex, left=left, right=right,
                            side_by_side=side_by_side, filename="dataset_distances_lower." + PLOT_FORMAT)
    plot_visibility_histogram(dataset, title=visibility_distribution, latex=use_latex, left=left, right=right,
                              side_by_side=side_by_side, filename="dataset_visibility_lower." + PLOT_FORMAT)
    plot_blurriness_histogram(dataset, title=sharpness_distribution, latex=use_latex, left=left, right=right,
                              side_by_side=side_by_side, filename="dataset_blurriness_lower." + PLOT_FORMAT)
    plot_size_histogram(dataset, title=size_distribution, latex=use_latex, left=left, right=right,
                        side_by_side=side_by_side, image_size=(240, 320), filename="dataset_size_lower." + PLOT_FORMAT)

if __name__ == '__main__':
    from tkinter import filedialog
    from tkinter import simpledialog
    from tkinter import *

    def ask_filename(init_dir=None):
        root = Tk()
        root.withdraw()
        if not init_dir:
            initial_dir = r"C:\NDevils\YOLO\DevilsKerasNeuralNetwork\models"
        else:
            initial_dir = init_dir

        root.filename = filedialog.askopenfilename(initialdir=initial_dir,
                                                   title="Select pkl files",
                                                   filetypes=(("pkl files", "*.pkl"), ("all files", "*.*")))
        root.destroy()
        print(root.filename)
        return root.filename

    def ask_title():
        root = Tk()
        root.update_idletasks()
        root.focus_force()
        answer = simpledialog.askstring("Title", "Type in the title:?",
                                        parent=root)
        return answer

    use_latex = False
    PLOT_FORMAT = "svg"
    if use_latex:
        rc('font', **{
            'family': 'serif',
            'sans-serif': ['Computer Modern Roman'],
            'size': '16',
            'style': 'normal', #'normal', 'italic', 'oblique'
            'variant': 'normal', #'small-caps'
            'stretch': 'expanded', # 0-1000, 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'
        })
        rc('text', usetex=True)
    else:
        rc('font', **{
            'family': 'serif',
            'serif': ['cmr10'],
            'size': '16',
            'style': 'normal',  # 'normal', 'italic', 'oblique'
            'variant': 'normal',  # 'small-caps'
            'stretch': 'expanded',  # 0-1000, 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'
        })
        rc('mathtext', **{
            'fontset': 'custom',
            'rm': 'Arial'
        })

