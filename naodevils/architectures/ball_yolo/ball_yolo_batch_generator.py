import cv2
import random
import numpy as np
from imgaug import augmenters as iaa
from functools import cache
from utils import *
from utils.decorator_utils import save_grid
from architectures import BatchGenerator


def make_generator_config_ball(validation=False):
    generator_config = {
        'IMAGE_H': FLAGS.height,
        'IMAGE_W': FLAGS.width,
        'OUTPUT_SIZE': 3,
        'LABELS': FLAGS.label_names,
        'CLASS': len(FLAGS.label_names),
        'BATCH_SIZE': FLAGS.batch_size,
        'DEBUG': DEBUGGING,
        'MULTIPROCESSING': FLAGS.use_multiprocessing,
        'BOTTOM_AS_CENTER': FLAGS.use_bottom_as_center,
        'THRESHOLD': FLAGS.iou_threshold,
    }
    if validation and FLAGS.validation_batch_size > 0:
        generator_config['BATCH_SIZE'] = FLAGS.validation_batch_size

    return generator_config

class plot_ball_batch(object):
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            x_batch, y_batch = func(*args, **kwargs)

            if DEBUGGING:
                all_imgs = []
                for i, img in enumerate(x_batch):
                    sanity_img = img.copy()
                    sanity_img *= 255
                    bbox = y_batch[i]
                    x = bbox[1]
                    y = bbox[2]
                    width_half = (bbox[3]/2.)
                    height_half = (bbox[4]/2.)
                    color = (0, 255, 0)
                    if bbox[0] > FLAGS.iou_threshold:
                        cv2.drawMarker(sanity_img, (int(round(x)), int(round(y))), color, cv2.MARKER_CROSS, 2, 1)
                        cv2.circle(sanity_img, (int(round(x)), int(round(y))), int(round((width_half + height_half) / 2.0)), color, 1)
                    all_imgs.append(sanity_img)

                cols = int(np.floor(np.sqrt(len(x_batch))))
                save_grid(all_imgs, os.path.join("data", "augmentations", args[0].NAME + "_batch_ball_{:04d}.png".format(args[1])), cols=cols)

            return x_batch, y_batch
        return wrapper

class BatchGeneratorBall(BatchGenerator):
    NAME = ""

    def __init__(self, images,
                 config,
                 shuffle=True,
                 grayscale=False,
                 norm=True,
                 augmentation_threshold=0.0,
                 name="training",
                 num_cores=4,
                 use_multithreading=False):
        super(BatchGeneratorBall, self).__init__(images=images,
                                             config=config,
                                             shuffle=shuffle,
                                             grayscale=grayscale,
                                             norm=norm,
                                             augmentation_threshold=augmentation_threshold,
                                             name=name,
                                             num_cores=num_cores,
                                             use_multithreading=use_multithreading)

    def shuffle_images(self):
        random.shuffle(self.images)
        eprint("Shuffled " + str(self.NAME) + " generator!")

    def set_augmentation_pipeline(self, augmentation_threshold):
        if augmentation_threshold != self.augmentation_threshold:
            self.augmentation_threshold = augmentation_threshold

        if self.NAME.startswith("predict") or self.NAME.endswith("validation"):
            self.aug_pipe = iaa.Identity()
            return

        deviation_percent = 8 / max(self.config['IMAGE_H'], self.config['IMAGE_W'])
        deviation_percent *= self.augmentation_threshold

        self.aug_pipe = iaa.Sequential(iaa.SomeOf((1, 3),
                                       [
                                           iaa.Affine(
                                               translate_percent={"x": (-deviation_percent, +deviation_percent), "y": (-deviation_percent, +deviation_percent)},
                                               mode="reflect",
                                               order=0),
                                           iaa.Affine(
                                               scale={"x": (1 - deviation_percent, 1 + deviation_percent), "y": (1 - (deviation_percent), 1 + (deviation_percent))},
                                               mode="reflect",
                                               order=0),
                                           iaa.Affine(rotate=(-15.0 * self.augmentation_threshold,
                                                              15.0 * self.augmentation_threshold),
                                                      mode="reflect",
                                                      order=0),
                                           iaa.Affine(shear=(-5.0 * self.augmentation_threshold,
                                                             5.0 * self.augmentation_threshold),
                                                      mode="reflect",
                                                      order=0),
                                           iaa.Fliplr(p=0.5),
                                           iaa.Flipud(p=0.33)
                                       ], random_order=True))

        if self.augmentation_threshold > 0.75:
            self.aug_pipe.append(iaa.SomeOf((1, 2),
                                 [
                                     iaa.MotionBlur(k=max(3, int(round(self.augmentation_threshold * 10))), angle=[-90, -75, -45, 45, 75, 90]),
                                     iaa.OneOf([
                                         iaa.Add((int(self.augmentation_threshold * -10),
                                                  int(self.augmentation_threshold * 10))),
                                         iaa.AddElementwise((int(self.augmentation_threshold * -10),
                                                             int(self.augmentation_threshold * 10)), per_channel=0.5),
                                         iaa.Multiply((1 - self.augmentation_threshold * 0.5,
                                                       1 + self.augmentation_threshold * 0.5)),
                                     ]),
                                     iaa.LinearContrast((1 - self.augmentation_threshold * 0.5,
                                                         1 + self.augmentation_threshold * 0.5)),
                                     iaa.Grayscale(alpha=(0.0, self.augmentation_threshold * 0.30)),
                                     iaa.JpegCompression(compression=(int(round(self.augmentation_threshold * 60)), 60)),
                                 ], random_order=True))

    def filter_bboxes(self, bounding_boxes, filter_too_small=True, min_factor=0.2):
        return super(BatchGeneratorBall, self).filter_bboxes(bounding_boxes=bounding_boxes,
                                                             filter_too_small=filter_too_small,
                                                             min_factor=min_factor)

    def on_epoch_end(self):
        return super(BatchGenerator, self).on_epoch_end()

    @cache
    def interpolate_blurriness(self, blurriness):
        return np.interp(blurriness, [500, 5000], [self.config["THRESHOLD"] + 0.001, 1.0])

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        image_ids = self.determine_batch_bounds(idx)
        channel = 3
        assert len(image_ids) == self.config['BATCH_SIZE']

        x_batch = np.zeros((len(image_ids), self.config['IMAGE_H'], self.config['IMAGE_W'], 1 if self.grayscale else channel), dtype=np.float32 if self.norm else np.uint8)  # input images
        y_batch = np.zeros((len(image_ids), self.config['OUTPUT_SIZE'] + 5), dtype=np.float32)  # desired network output

        images, bboxes_on_images = self.load_images_and_bboxes_from_batch(image_ids, use_multithreading=True)
        images_aug, bboxes_on_images_aug = self.augment_images_and_bboxes(images, bboxes_on_images, self.augmentation_threshold, use_multithreading=True)

        instance_count = 0
        for i in range(len(images_aug)):
            # assign input image to x_batch
            img = images_aug[i]
            if self.norm:
                x_batch[instance_count, :, :, :3] = img / 255.
            else:
                x_batch[instance_count, :, :, :3] = img

            bboxes = bboxes_on_images_aug[i].bounding_boxes

            if len(bboxes) > 0:
                center_x = bboxes[0].center_x
                center_y = bboxes[0].center_y

                if center_x > self.config['IMAGE_W']:
                    center_x = self.config['IMAGE_W']
                elif center_x < 0:
                    center_x = 0.0

                if center_y > self.config['IMAGE_H']:
                    center_y = self.config['IMAGE_H']
                elif center_y < 0:
                    center_y = 0.0

                blurriness = bboxes_on_images_aug[i][0].label["blurriness"]
                visibility = bboxes_on_images_aug[i][0].label["visibility"]
                concealed = bboxes_on_images_aug[i][0].label["concealed"]
                if blurriness <= 0:
                    assert False

                y_batch[instance_count, 0] = self.interpolate_blurriness(blurriness)

                y_batch[instance_count, 1] = center_x
                y_batch[instance_count, 2] = center_y
                y_batch[instance_count, 3] = bboxes[0].width
                y_batch[instance_count, 4] = bboxes[0].height
                y_batch[instance_count, 5] = blurriness

                factor = (1. - self.config["THRESHOLD"]) / 5
                if visibility > 4:
                    y_batch[instance_count, 6] = factor * 1
                elif visibility == 4:
                    y_batch[instance_count, 6] = self.config["THRESHOLD"] + factor * 1
                elif visibility == 3:
                    y_batch[instance_count, 6] = self.config["THRESHOLD"] + factor * 2
                elif visibility == 2:
                    y_batch[instance_count, 6] = self.config["THRESHOLD"] + factor * 3
                elif visibility == 1:
                    y_batch[instance_count, 6] = self.config["THRESHOLD"] + factor * 4
                else:
                    y_batch[instance_count, 6] = self.config["THRESHOLD"] + factor * 5

                if concealed:
                    y_batch[instance_count, 6] -= factor

                # Confidence as minimum of visibility and blurriness
                y_batch[instance_count, 0] = min(y_batch[instance_count, 0], y_batch[instance_count, 6])

            else:
                y_batch[instance_count, 0] = 0.
                y_batch[instance_count, 1] = 0.
                y_batch[instance_count, 2] = 0.
                y_batch[instance_count, 3] = 0.
                y_batch[instance_count, 4] = 0.
                y_batch[instance_count, 5] = self.images[image_ids[i]]["imageBlurriness"]
                y_batch[instance_count, 6] = 1.

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch