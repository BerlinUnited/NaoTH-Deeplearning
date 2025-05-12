import cv2
import numpy as np
from utils import *
from utils.decorator_utils import save_grid
from architectures.ball_yolo.ball_yolo_neural_network import BatchGeneratorBall

def make_generator_config_multi_stage_ball(validation=False):
    generator_config = {
        'IMAGE_H': FLAGS.height,
        'IMAGE_W': FLAGS.width,
        'OUTPUT_SIZE': [1, 3],
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

class plot_multi_stage_ball_batch(object):
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
                        cv2.drawMarker(sanity_img, (x, y), color, cv2.MARKER_CROSS, 2, 2)
                        cv2.circle(sanity_img, (x, y), round((width_half + height_half) / 2.0), color, 1)
                    all_imgs.append(sanity_img)

                cols = int(np.floor(np.sqrt(len(x_batch))))
                save_grid(all_imgs, os.path.join("data", "augmentations", args[0].NAME + "_batch_ball_{:04d}.png".format(args[1])), cols=cols)

            return x_batch, y_batch
        return wrapper

class BatchGeneratorMultiStageBall(BatchGeneratorBall):
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
        super(BatchGeneratorMultiStageBall, self).__init__(images=images,
                                             config=config,
                                             shuffle=shuffle,
                                             grayscale=grayscale,
                                             norm=norm,
                                             augmentation_threshold=augmentation_threshold,
                                             name=name,
                                             num_cores=num_cores,
                                             use_multithreading=use_multithreading)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        image_ids = self.determine_batch_bounds(idx)
        channel = 3
        assert len(image_ids) == self.config['BATCH_SIZE']

        x_batch = np.zeros((len(image_ids), self.config['IMAGE_H'], self.config['IMAGE_W'], 1 if self.grayscale else channel), dtype=np.float32 if self.norm else np.uint8)  # input images
        y_confidence = np.zeros((len(image_ids), self.config['OUTPUT_SIZE'][0] + 3), dtype=np.float32)  # desired first network output
        y_position = np.zeros((len(image_ids), self.config['OUTPUT_SIZE'][1] + 3), dtype=np.float32)  # desired second network output

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

                factor = (1. - self.config["THRESHOLD"]) / 5
                if visibility > 4:
                    visibility_factor = factor * 1
                elif visibility == 4:
                    visibility_factor = self.config["THRESHOLD"] + factor * 1
                elif visibility == 3:
                    visibility_factor = self.config["THRESHOLD"] + factor * 2
                elif visibility == 2:
                    visibility_factor = self.config["THRESHOLD"] + factor * 3
                elif visibility == 1:
                    visibility_factor = self.config["THRESHOLD"] + factor * 4
                else:
                    visibility_factor = self.config["THRESHOLD"] + factor * 5
                if concealed:
                    visibility_factor -= factor

                y_confidence[instance_count, 0] = self.interpolate_blurriness(blurriness)
                y_confidence[instance_count, 1] = blurriness
                y_confidence[instance_count, 2] = visibility_factor
                # Confidence as minimum of visibility and blurriness
                y_confidence[instance_count, 0] = min(y_confidence[instance_count, 0], y_confidence[instance_count, 2])

                y_position[instance_count, 0] = y_confidence[instance_count, 0]
                y_position[instance_count, 1] = center_x
                y_position[instance_count, 2] = center_y
                y_position[instance_count, 3] = bboxes[0].width
                y_position[instance_count, 4] = bboxes[0].height
                y_position[instance_count, 5] = y_confidence[instance_count, 2]

            else:
                y_confidence[instance_count, 0] = 0.
                y_confidence[instance_count, 1] = self.images[image_ids[i]]["imageBlurriness"]
                y_confidence[instance_count, 2] = 0.
                y_position[instance_count, 0] = y_confidence[instance_count, 0]
                y_position[instance_count, 1] = 0.
                y_position[instance_count, 2] = 0.
                y_position[instance_count, 3] = 0.
                y_position[instance_count, 4] = 0.
                y_position[instance_count, 5] = y_confidence[instance_count, 2]

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, [y_position, y_confidence] #[y_confidence, y_position]