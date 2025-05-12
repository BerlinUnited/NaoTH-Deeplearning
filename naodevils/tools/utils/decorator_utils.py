import timeit
import threading
import numpy as np
import cv2
from utils import *

def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            import traceback
            print_seperator()
            print("Error in " + str(func.__name__))
            traceback.print_exc(file=sys.stdout)
            print_seperator()
            exit(-1)
    return wrapper

thread_mapping_list = []

def time_it(func):
    def wrapper(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id not in thread_mapping_list:
            thread_mapping_list.append(thread_id)

        start_time = timeit.default_timer()
        results = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time

        print('Thread {thread}: Function "{name}" took {time:.06f} seconds to complete.'.format(thread=thread_mapping_list.index(thread_id), name=func.__name__, time=elapsed))
        return results
    return wrapper

def save_grid(images, filename, cols=8):
    grid_image = None
    horizontal_row = None
    for i, img in enumerate(images):
        rest = i % cols
        if rest == 0:
            if i == cols:
                grid_image = horizontal_row
            elif i > cols:
                grid_image = np.vstack((grid_image, horizontal_row))
            horizontal_row = img
        else:
            horizontal_row = np.hstack((horizontal_row, img))

    if horizontal_row.shape[1] != grid_image.shape[1]:
        horizontal_row = np.hstack((horizontal_row, np.zeros(
            [horizontal_row.shape[0], grid_image.shape[1] - horizontal_row.shape[1], horizontal_row.shape[2]],
            dtype=int)))
    grid_image = np.vstack((grid_image, horizontal_row))
    grid_image = np.flip(grid_image, 2)
    if os.path.isfile(filename):
        os.remove(filename)
    cv2.imwrite(filename, grid_image)

class plot_patch_batch(object):
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            (x_batch, xywh_batch), y_batch = func(*args, **kwargs)

            if not DEBUGGING:
                all_imgs = []
                for i, img in enumerate(x_batch):
                    sanity_img = img.copy()
                    sanity_img *= 255
                    bbox = y_batch[i]
                    x = bbox[1]
                    y = bbox[2]
                    width_half = (bbox[3]/2.)
                    height_half = (bbox[4]/2.)
                    upper_left = (
                        int(x - width_half),
                        int(y - height_half * 2)
                    )
                    lower_right = (
                        int(x + width_half),
                        int(y + 0)
                    )
                    if upper_left != lower_right:
                        cv2.rectangle(sanity_img, upper_left, lower_right, (255, 255, 255), 1)

                    all_imgs.append(sanity_img)

                cols = int(np.floor(np.sqrt(len(x_batch))))
                save_grid(all_imgs, os.path.join("data", "augmentations", args[0].NAME + "_batch_{:04d}.png".format(args[1])), cols=cols)

            return (x_batch, xywh_batch), y_batch
        return wrapper
