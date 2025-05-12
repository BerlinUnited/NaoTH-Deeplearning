from datetime import datetime
from psutil import virtual_memory
import png
from tkinter import Tk, filedialog
import numpy as np
import cv2
import pickle

from utils import *
from utils.tqdm_utils import tqdm, TQDM_BAR_FORMAT, TQDM_UPDATE_INTERVAL

from protobuf.protobuf import clearProtobufAnotations, ImageLabelData, encode_data, write_label_chunk

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def get_curr_time():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def get_size(images):
    total_size = 0
    for image in images:
        total_size += os.path.getsize(image["filename"])
    return total_size

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def find_nth_back(haystack, needle, n):
    end = haystack.rfind(needle)
    while end >= 0 and n > 1:
        end = haystack.rfind(needle, 0, end-len(needle))
        n -= 1
    return end

def print_memory_available():
    print("RAM available: {:.2f} Gb ({:.1f}% free)".format(virtual_memory().available / (1024 * 1024 * 1024), 100 - virtual_memory().percent))

def check_if_images_fit_ram(images):
    avail = virtual_memory().available
    needed = get_size(images) * 1.14
    print("Need {:.2f} Gb of Ram for the images".format(needed / (1024 * 1024 * 1024)))
    return avail > needed, np.ceil(needed / (1024 * 1024))

def bbox_iou(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union

def interval_overlap(interval_a, interval_b):
    """
    returns the overlap between two 1D-intervals
    :param interval_a:
    :param interval_b:
    :return:
    """
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def freeze_all(model, frozen=True):
    from utils.setup_tensorflow_utils import keras
    model.trainable = not frozen
    if isinstance(model, keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def chunks(lst, n):
    """Return successive n-sized chunks from lst."""
    result = []
    for i in range(0, len(lst), n):
        result.append(lst[i:i + n])
    return result

def _sigmoid(x):
    return 1. / ((1. + np.exp(-x)) + np.finfo(np.float32).eps)

def _inv_sigmoid(x):
    return -np.log((1 / (x + np.finfo(np.float32).eps)) - 1)

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def get_label_text(box, labels):
    if box.label < len(labels):
        label_text = labels[box.label]
    else:
        label_text = "unknown"

    return label_text

def draw_boxes(image, boxes, labels, print_classname=False):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        if box.label == 0:
            rgb_color = (0, 255, 0)
        elif box.label == 1:
            rgb_color = (255, 0, 0)
        elif box.label == 2:
            rgb_color = (0, 0, 255)
        elif box.label == 3:
            rgb_color = (255, 255, 255)
        else:
            rgb_color = (0, 0, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rgb_color, 2)

        if print_classname:
            image_text = "{0:4.2f}_{1:s}".format(box.get_score() * 100., get_label_text(box, labels))
        else:
            image_text = "{0:4.2f}".format(box.get_score() * 100.)

        if (0 + ymin) > (image_h - ymax):
            cv2.putText(image,
                        image_text,
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h,
                        rgb_color, 2)
        else:
            cv2.putText(image,
                        image_text,
                        (xmin, ymax + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h,
                        rgb_color, 2)

    return image

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_visibility(a, b):
    """
    Parameters
    ----------
    a: (4) array of float
    b: (4) array of float
    Returns
    -------
    overlaps: value of how much of a is in b
    """
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])
    ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = area_a + area_b - iw * ih

    return ((1 - ((ua-area_a) / np.maximum(area_b, np.finfo(float).eps))), area_a, area_b)

def compute_ap(recall, precision, use_07_metric=True):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    if not isAscending(recall):
        recall.reverse()
    if not isDescending(precision):
        precision.reverse()

    # # 11 point metric
    # ap11 = 0.
    # recall = np.array(recall)
    # precision = np.array(precision)
    # for t in np.arange(0., 1.1, 0.1):
    #     if np.sum(recall >= t) == 0:
    #         p = 0
    #     else:
    #         p = np.max(precision[recall >= t])
    #     ap11 = ap11 + p / 11.

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    #mrec = mrec[::-1]
    #mpre = mpre[::-1]
    #average_precision = -np.sum(np.diff(mrec) * np.array(mpre)[:-1])

    return ap

def isAscending(list):
    previous = list[0]
    no_of_outliers = 0
    max_no_of_outliers = int(0.25 * len(list))
    for number in list:
        if number < previous:
            no_of_outliers += 1
        if no_of_outliers > max_no_of_outliers:
            return False
        previous = number
    return True

def isDescending(list):
    previous = list[0]
    no_of_outliers = 0
    max_no_of_outliers = int(0.25 * len(list))
    for number in list:
        if number > previous:
            no_of_outliers += 1
        if no_of_outliers > max_no_of_outliers:
            return False
        previous = number
    return True

def compute_mAP(all_detections, all_annotations, iou_threshold):
    # compute mAP by comparing all detections and all annotations
    average_precisions = {}
    recalls = {}
    precisions = {}
    for label in range(len(FLAGS.label_names)):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in tqdm(range(len(all_detections)), total=len(all_detections), desc="Calculate IOUs and mAP for {}".format(FLAGS.label_names[label]), file=sys.stdout, bar_format=TQDM_BAR_FORMAT, mininterval=TQDM_UPDATE_INTERVAL):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
        recalls[label] = recall
        precisions[label] = precision

    return average_precisions, recalls, precisions

def bound_to_image_size(xmin, xmax, ymin, ymax, width, height):
    if xmin < 0:
        #xmax -= xmin
        xmin = 0
    elif xmax > width:
        #xmin -= xmax - width
        xmax = width
    if ymin < 0:
        #ymax -= ymin
        ymin = 0
    elif ymax > height:
        #ymin -= ymax - height
        ymax = height
    return xmin, xmax, ymin, ymax

def calculate_hash_list(true_positives, false_negatives, false_positives, true_negatives, filename=""):
    tp_hashes = []
    fn_hashes = []
    fp_hashes = []
    cp = []
    cn = []
    for label_index, label in enumerate(FLAGS.label_names):
        tp_hashes.append(list())
        fn_hashes.append(list())
        fp_hashes.append(list())
        for vis_lvl in range(len(true_positives[label_index])):
            if len(true_positives[label_index][vis_lvl]) > 0:
                tp_hashes[label_index].extend([(str(l[np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])]), l[5]) for l in true_positives[label_index][vis_lvl]])
            if len(false_negatives[label_index][vis_lvl]) > 0:
                fn_hashes[label_index].extend([(str(l[np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])]), l[5]) for l in false_negatives[label_index][vis_lvl]])
        for l in false_positives[label_index]:
            fp_hashes[label_index].append((str(l[np.array([0, 1, 2, 3, 4])]), l[5]))
        cp.append(len(tp_hashes[label_index]) + len(fn_hashes[label_index]))
        cn.append(len(false_positives[label_index]) + len(true_negatives[label_index]))

    data = {}
    data['cp'] = cp
    data['cn'] = cn
    data['tp_hashes'] = tp_hashes
    data['fn_hashes'] = fn_hashes
    data['fp_hashes'] = fp_hashes

    if filename != "":
        with open(filename + "_hashes.pkl", 'wb') as f:
            pickle.dump(data, f)

def save_patch(l, image, patch_filename, visibility, area_ratio, decoded_protobuf, det_bbox, ann_bbox, extra_folder=False, width_height_factor=1.0, size=32, teamcolor=None):
    det_xmin = det_bbox[0]
    det_ymin = det_bbox[1]
    det_xmax = det_bbox[2]
    det_ymax = det_bbox[3]

    ann_xmin = ann_bbox[0]
    ann_ymin = ann_bbox[1]
    ann_xmax = ann_bbox[2]
    ann_ymax = ann_bbox[3]

    patchdir = os.path.join(os.getcwd(), "data", "patches", str(FLAGS.label_names[l]))

    if DEBUGGING:
        plot_img = image / 255.0
        plot_img = cv2.rectangle(plot_img, (det_xmin, det_ymin), (det_xmax, det_ymax), (255, 0, 0), 2)
        plot_img = cv2.rectangle(plot_img, (int(ann_xmin), int(ann_ymin)), (int(ann_xmax), int(ann_ymax)), (0, 255, 0), 2)
        # plt.imshow(plot_img)

    label_visible = False
    if visibility >= 0.75 and area_ratio > (1/2):
        patchdir = os.path.join(patchdir, "1.00")
        label_visible = True
    elif visibility >= 0.75 and area_ratio > (1/4):
        patchdir = os.path.join(patchdir, "0.90")
        label_visible = True
    elif visibility >= 0.50:
        patchdir = os.path.join(patchdir, "0.75")
        label_visible = True
    elif visibility >= 0.10:
        patchdir = os.path.join(patchdir, "0.50")
        label_visible = True
    elif visibility > 0.00:
        patchdir = os.path.join(patchdir, "0.10")
        label_visible = True
    else:
        patchdir = os.path.join(patchdir, "0.00")

    if teamcolor is not None:
        patchdir = os.path.join(patchdir, str(teamcolor))

    if extra_folder:
        patchdir = os.path.join(patchdir, "modified")

    resize_factor = (det_xmax - det_xmin) / size
    try:
        width = det_xmax-det_xmin
        height = det_ymax-det_ymin
        if width > 0 and height > 0:
            clipped_xmin, clipping_patch_xmin = (0, -det_xmin) if det_xmin < 0 else (det_xmin, 0)
            clipped_xmax, clipping_patch_xmax = (image.shape[1], -(det_xmax - image.shape[1])) if det_xmax > image.shape[1] else (det_xmax, width)
            clipped_ymin, clipping_patch_ymin = (0, -det_ymin) if det_ymin < 0 else (det_ymin, 0)
            clipped_ymax, clipping_patch_ymax = (image.shape[0], -(det_ymax - image.shape[0])) if det_ymax > image.shape[0] else (det_ymax, height)

            image_patch = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
            image_patch[clipping_patch_ymin:clipping_patch_ymax, clipping_patch_xmin:clipping_patch_xmax] = image[clipped_ymin:clipped_ymax, clipped_xmin:clipped_xmax]
            patch = cv2.resize(image_patch, (int(size), int(size/width_height_factor)), interpolation=cv2.INTER_NEAREST)
        else:
            return
    except Exception as e:
        print(str(e))
        return

    if not os.path.exists(patchdir):
        os.makedirs(patchdir)
    patchname = os.path.join(patchdir, patch_filename)
    png.from_array(np.reshape(patch, (-1, patch.shape[1] * patch.shape[2])), 'RGB').save(patchname)

    decoded_protobuf = clearProtobufAnotations(decoded_protobuf)

    decoded_protobuf.patchBoundingBox.upperLeft.x = det_xmin
    decoded_protobuf.patchBoundingBox.upperLeft.y = det_ymin
    decoded_protobuf.patchBoundingBox.lowerRight.x = det_xmax
    decoded_protobuf.patchBoundingBox.lowerRight.y = det_ymax

    if label_visible:
        if l == 0:
            object = decoded_protobuf.robots.add()
            if teamcolor:
                object.teamcolor = teamcolor
        elif l == 1:
            object = decoded_protobuf.balls.add()
        elif l == 2:
            object = decoded_protobuf.penaltyCrosses.add()

        object = object.label

        object.boundingBox.upperLeft.x = int(round((ann_xmin - det_xmin) / resize_factor))
        object.boundingBox.upperLeft.y = int(round((ann_ymin - det_ymin) / resize_factor))
        object.boundingBox.lowerRight.x = int(round((ann_xmax - det_xmin) / resize_factor))
        object.boundingBox.lowerRight.y = int(round((ann_ymax - det_ymin) / resize_factor))

        object.concealed = False
        object.blurriness = 0
        if visibility == 1.0:
            object.visibilityLevel = ImageLabelData.Label.FULL
        elif visibility > 0.75:
            object.visibilityLevel = ImageLabelData.Label.THREEQUARTER_TO_FULL
        elif visibility > 0.50:
            object.visibilityLevel = ImageLabelData.Label.HALF_TO_THREEQUARTER
        elif visibility > 0.25:
            object.visibilityLevel = ImageLabelData.Label.ONEQUARTER_TO_HALF
        elif visibility > 0.0:
            object.visibilityLevel = ImageLabelData.Label.ZERO_TO_ONEQUARTER
        elif visibility == 0.0:
            object.visibilityLevel = ImageLabelData.Label.HIDDEN

        object.person = "DevilsKerasNeuralNetwork"

        today = datetime.now()
        object.date.day = today.day
        object.date.month = today.month
        object.date.year = today.year
    encoded_protobuf = encode_data(decoded_protobuf)
    write_label_chunk(patchname, encoded_protobuf)

def ask_folder(initial_dir=""):
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    root.focus_force()
    if initial_dir == "":
        initial_dir = os.getcwd()
    root.filenames = filedialog.askdirectory(initialdir=initial_dir, title="Select image folder")
    root.destroy()
    return root.filenames

def ask_file(initial_dir="", initialfile="", filetypes=[("All Files", ".*")]):
    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    root.focus_force()
    if initial_dir == "":
        initial_dir = os.getcwd()
    root.filenames = filedialog.askopenfilename(filetypes=filetypes, initialdir=initial_dir, initialfile=initialfile, title="Select feature model")
    root.destroy()
    return root.filenames

