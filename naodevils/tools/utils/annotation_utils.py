import threading
from functools import partial
from multiprocessing.pool import ThreadPool
import numpy as np
import imagesize

from protobuf.protobuf import read_bboxes, decode_data, encode_data, write_label_chunk

from utils import *
from utils.tqdm_utils import tqdm, TQDM_BAR_FORMAT, TQDM_UPDATE_INTERVAL

from utils.decorator_utils import exception_catcher

c = threading.Condition()

def parse_annotation_protobuf(img_dir, labels=None, search_subfolders=True, folder_filter_list=None, load_all=False,
                              use_multithreading=True):
    """
    walks through a dir, lists all images and then read the protobuf annotations
    :param img_dir: folder to cycle through
    :param labels: labels to scan for
    :param search_subfolders: search in subfolders instead of only in the root foler
    :param folder_filter_list: only enter subfolders with this names
    :return: list of dict, which contains the image path and the parsed annotations and a list of countings for each label
    """
    all_imgs = []
    seen_labels = {}

    image_dict = {}
    print("Loading Protobuf Annotations from '{}'...".format(img_dir))
    exclude = ["out"]

    for root, dirs, files in os.walk(img_dir, topdown=False):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if len(folder_filter_list) > 0:
                if not os.path.basename(root) in folder_filter_list:
                    break
            elif file.endswith('.png'):
                image_dict[file[0:-4]] = os.path.join(root, file)

        if not search_subfolders:
            break

    for label in labels:
        seen_labels[label] = 0

    func = partial(parse_single_annotation_protobuf, image_dict, labels, seen_labels, all_imgs, load_all)
    if use_multithreading:
        with ThreadPool(FLAGS.num_cores) as p:
            list(tqdm(p.imap(func, image_dict.keys()), total=len(image_dict.keys()), desc="Loading Annotations",
                      file=sys.stdout, bar_format=TQDM_BAR_FORMAT, mininterval=TQDM_UPDATE_INTERVAL))
    else:
        for key in tqdm(image_dict.keys(), total=len(image_dict.keys()), desc="Loading Annotations", file=sys.stdout,
                        mininterval=TQDM_UPDATE_INTERVAL):
            func(key)

    if img_dir == FLAGS.val_dir:
        prefix = "Validation "
    else:
        prefix = "Training "

    print("Number of valid Images: " + str(len(all_imgs)))
    print("Seen labels:")
    for k, v in seen_labels.items():
        print("\t" + str(k) + ": " + str(v))

    all_imgs = sorted(all_imgs, key=lambda img: img['filename'])
    return all_imgs


@exception_catcher
def parse_single_annotation_protobuf(image_dict, labels, seen_labels, all_imgs, load_all, key):
    img = {'object': []}
    img['filename'] = image_dict[key]
    if 'hard_examples' in img['filename'] or 'manual' in img['filename']:
        img['hard'] = True
    else:
        img['hard'] = False
    try:
        with open(img['filename'], "rb") as f:
            decoded_protobuf = decode_data(f)

        img['width'], img['height'] = imagesize.get(img['filename'])
        img['depth'] = decoded_protobuf.imageInfos.imagechannel

        try:
            if decoded_protobuf.timestamp == 0:
                img['timestamp'] = int(str(os.path.basename(img['filename'])).split("_")[0].split(".")[0])
            else:
                img['timestamp'] = decoded_protobuf.timestamp
        except Exception as e:
            img['timestamp'] = -1
            eprint(str(e))

        try:
            img['datasetName'] = decoded_protobuf.datasetName
        except Exception as e:
            img['datasetName'] = ""
            eprint(str(e))

        try:
            rotationMatrix = decoded_protobuf.cameraInformation.rotation
            img["rotation"] = np.array([
                [rotationMatrix.m00, rotationMatrix.m01, rotationMatrix.m02],
                [rotationMatrix.m10, rotationMatrix.m11, rotationMatrix.m12],
                [rotationMatrix.m20, rotationMatrix.m21, rotationMatrix.m22],
            ])
        except Exception as e:
            eprint(str(e))

        try:
            translation = decoded_protobuf.cameraInformation.translation
            img["translation"] = np.array([translation.x, translation.y, translation.z])
        except Exception as e:
            eprint(str(e))

        try:
            intrinsics = decoded_protobuf.cameraInformation.cameraIntrinsics
            img["openingAngleHeight"] = intrinsics.openingAngleHeight
            img["openingAngleWidth"] = intrinsics.openingAngleWidth
            img['full_width'] = decoded_protobuf.cameraInformation.resolution.width
            img['full_height'] = decoded_protobuf.cameraInformation.resolution.height
            img["opticalCenter"] = np.array([intrinsics.opticalCenter.x * img['full_width'], intrinsics.opticalCenter.y * img['height']])
            img["focalLength"] = img['full_width'] / (2.0 * np.tan(np.radians(img["openingAngleWidth"]) / 2.0) + np.finfo(float).eps)
            img["invFocalLength"] = 1 / (img["focalLength"] + np.finfo(float).eps)
        except Exception as e:
            eprint(str(e))

        try:
            img["patch_xmin"] = decoded_protobuf.patchBoundingBox.upperLeft.x
            img["patch_xmax"] = decoded_protobuf.patchBoundingBox.lowerRight.x
            img["patch_ymin"] = decoded_protobuf.patchBoundingBox.upperLeft.y
            img["patch_ymax"] = decoded_protobuf.patchBoundingBox.lowerRight.y
        except Exception as e:
            eprint(str(e))

        global_updated = False

        if "robot" in labels:
            updated, bboxes = read_bboxes("robot", decoded_protobuf.robots, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["robot"] += len(bboxes)

        if "obstacle" in labels:
            updated, bboxes = read_bboxes("obstacle", decoded_protobuf.obstacles, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["obstacle"] += len(bboxes)

        if "ball" in labels:
            updated, bboxes = read_bboxes("ball", decoded_protobuf.balls, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["ball"] += len(bboxes)

        if "goal post" in labels:
            updated, bboxes = read_bboxes("goal post", decoded_protobuf.goalposts, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["goal post"] += len(bboxes)

        if "penalty cross" in labels:
            updated, bboxes1 = read_bboxes("penalty cross", decoded_protobuf.DEPRECATED_penaltyCrosses, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes1
            updated, bboxes2 = read_bboxes("penalty cross", decoded_protobuf.penaltyCrosses, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes2
            with c:
                seen_labels["penalty cross"] += len(bboxes1)
                seen_labels["penalty cross"] += len(bboxes2)

        if "line crossing" in labels:
            updated, bboxes = read_bboxes("line crossing", decoded_protobuf.lineCrossing, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["line crossing"] += len(bboxes)

        if "center circle" in labels:
            updated, bboxes = read_bboxes("center circle", decoded_protobuf.centerCircle, filename=img['filename'], width=img['width'], height=img['height'])
            if updated == True:
                global_updated = True
            img['object'] += bboxes
            with c:
                seen_labels["center circle"] += len(bboxes)

        if global_updated:
            try:
                encoded_protobuf = encode_data(decoded_protobuf)
                write_label_chunk(img['filename'], encoded_protobuf)
                print("Updated: " + str(img['filename']))
            except Exception as e:
                eprint(e)

    except Exception as e:
        eprint(str(e))
        if DEBUGGING:
            import traceback
            print_seperator()
            eprint("Error in parse_annotation_protobuf of " + str(img['filename']))
            traceback.print_exc(file=sys.stdout)
            print_seperator()

    with c:
        if len(img['object']) > 0 or load_all:
            all_imgs += [img]

def generate_batches_from_sequences(name, images, verbose=False):
    sequences = []
    images = sorted(images, key=lambda x: (x["filename"]))
    startindex = 0
    endindex = startindex + 1
    current_timestamp = images[startindex]["timestamp"]
    current_dataset_name = images[startindex]["datasetName"]
    while endindex < len(images):
        if "timestamp" not in images[endindex]:
            assert False, "Image " + str(images[endindex]['filename']) + " produces an error!"
        time_diff = images[endindex]["timestamp"] - current_timestamp
        skipped_frames = round(time_diff / 33.0) - 1
        if current_dataset_name == images[endindex]["datasetName"] and skipped_frames < 60:
            if skipped_frames > 0 and verbose:
                print("[" + name + "] " + os.path.basename(images[endindex]["filename"]) + " skipped " + str(skipped_frames) + " Frames!!!")
            current_timestamp = images[endindex]["timestamp"]
            endindex += 1
        else:
            sequences.append((startindex, endindex - 1, endindex - startindex))
            startindex = endindex
            endindex = startindex + 1
            if startindex < len(images):
                current_timestamp = images[startindex]["timestamp"]
                current_dataset_name = images[startindex]["datasetName"]
            else:
                break
    sequences.append((startindex, endindex - 1, endindex - startindex))
    minimal_sequence = min(sequences, key=lambda x: x[2])
    minimal_sequence_length = minimal_sequence[2]
    return images, sequences, minimal_sequence_length