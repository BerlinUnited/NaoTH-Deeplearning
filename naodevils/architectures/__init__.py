import cv2
import random
import pickle
import cProfile
import numpy as np
from functools import partial
from functools import cache
from multiprocessing.pool import ThreadPool
import imgaug

from utils import *
from utils.setup_tensorflow_utils import *
from utils.tqdm_utils import tqdm, TQDM_BAR_FORMAT, TQDM_SMOOTHING
from utils.hardware_utils import set_gpu_and_cpu_usage
from utils.util_functions import get_curr_time, chunks
from utils.augmenter_utils import ResizeWithCameraMatrix

from utils.callback_utils import DevilsProgbarLogger, DevilsStatistics
from utils.optimizer_utils import DevilsLookahead, DevilsSGD, DevilsRMSprop, DevilsAdam, DevilsAdamW, DevilsAdadelta, DevilsAdagrad, \
    DevilsLAMB, DevilsRAdam, DevilsAMSGrad, DevilsAdamax, DevilsNadam
from utils.tensorflow_utils import load_model_and_meta_params_from_json, save_keras_model_to_file, get_flops, \
    load_checkpoint, load_train_and_validation_images, make_train_and_validation_generator, convert_to_tflite

from utils.seeding_utils import *

class NeuralNetwork(object):
    _OPTIMIZERS = dict({
        'sgd': partial(DevilsSGD, nesterov=True),
        'rmsprop': DevilsRMSprop,
        'adadelta': DevilsAdadelta,
        'adagrad': DevilsAdagrad,
        'adam': partial(DevilsAdam, epsilon=0.1),
        'adamw': partial(DevilsAdamW, weight_decay=1e-2),
        'lamb': DevilsLAMB,
        'radam': DevilsRAdam,
        'amsgrad': partial(DevilsAMSGrad, amsgrad=True, name='AMSGrad', epsilon=0.1),
        'adamax': DevilsAdamax,
        'nadam': DevilsNadam
    })

    def __init__(self, generator_function, generator_config_function, name="", checkpoint=None, create_dot=False, load_all=False, pickleDataset=""):
        self.generator_function = generator_function
        self.generator_config_function = generator_config_function
        self.pickleDataset = pickleDataset
        setSeed(SEED)
        set_gpu_and_cpu_usage()

        ###########################
        # Load and make the model #
        ###########################
        self.json_model = load_model_and_meta_params_from_json()

        FLAGS.norm = self.json_model.input.dtype == tf.float32
        print("Model input " + str(self.json_model.input.dtype) + " Norm:" + str(FLAGS.norm))

        self.train_generator, self.train_imgs, \
        self.valid_generator, self.valid_imgs = self.load_images_and_create_generators(generator_function=self.generator_function,
                                                     generator_config_function=self.generator_config_function,
                                                     load_all=load_all,
                                                     hard_multiplier=FLAGS.hard_multiplier)
        self.model_name = self.generate_model_name()

        self.model, self.val_model = self.load_or_create_model(checkpoint=checkpoint)

        # Save the model as graph and as json
        self.checkpoint_filepath = str(
            os.path.join(
                FLAGS.trained_models_dir,
                name,
                self.model_name,  # Folder
                self.model_name,  # Beginning of Model name
            )
        )
        if not os.path.exists(os.path.dirname(self.checkpoint_filepath)):
            os.makedirs(os.path.dirname(self.checkpoint_filepath))

        printable_model = self.model if self.func_model is None else self.func_model

        if create_dot:
            try:
                keras.utils.plot_model(printable_model, show_shapes=True, expand_nested=True, to_file=self.checkpoint_filepath + '.png')
            except Exception as e:
                print_seperator(True)
                eprint(str(e))
                print_seperator(True)
        save_keras_model_to_file(printable_model, self.checkpoint_filepath + '.json')

        # print a summary of the whole model
        print_seperator()
        print("Model:")
        printable_model.summary(line_length=150)
        if DEBUGGING:
            print("Whole Model:")
            self.model.summary(line_length=150)
        self.flops = get_flops(printable_model, os.path.basename(FLAGS.model_file)[:-4], debug=False)

        self.current_summary_dir = os.path.join(FLAGS.summary_dir, self.model_name, get_curr_time())

        print_seperator()
        print("Loaded" + (" epoch " + str(FLAGS.initial_epoch) if FLAGS.initial_epoch > 0 else " NO epoch"))
        print("Used Labels:" + str(FLAGS.label_names))

    def generate_model_name(self):
        return os.path.basename(FLAGS.model_file)[:-4] + "_bs%03d" % (FLAGS.batch_size)

    def load_or_create_model(self, checkpoint):
        self.loaded_model, self.loaded_optimizer_weights, self.callback_state = load_checkpoint(self.model_name, self.json_model, checkpoint)
        self.func_model = None
        return self.create_model(), self.create_val_model()

    def load_images_and_create_generators(self, generator_function, generator_config_function, load_all, hard_multiplier):
        train_imgs, valid_imgs = load_train_and_validation_images(load_all, self.pickleDataset, hardMultiplier=hard_multiplier)
        # np.random.shuffle(train_imgs)
        # np.random.shuffle(valid_imgs)
        self.generate_validation_parameters(valid_imgs)
        train_generator, valid_generator = make_train_and_validation_generator(train_imgs=train_imgs,
                                                                               valid_imgs=valid_imgs,
                                                                               generator_config_function=generator_config_function,
                                                                               generator_function=generator_function)
        return train_generator, train_imgs, valid_generator, valid_imgs

    def generate_validation_parameters(self, valid_imgs):
        pass

    def create_model(self):
        raise NotImplemented("Please implement create_model")

    def create_val_model(self):
        return None

    def create_optimizer(self):
        return DevilsLookahead(optimizer=self._OPTIMIZERS[FLAGS.trainer](learning_rate=FLAGS.learning_rate))

    # @with_tensorboard
    def train(self, loss, metrics, loss_weights=None, individual_callbacks=None):
        ############################################
        # Make a few callbacks
        ############################################
        callbacks = individual_callbacks

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.current_summary_dir,
            histogram_freq=5,
            write_graph=False,
            write_images=True,
            update_freq='batch',
            profile_batch=0,
        )
        callbacks.append(tensorboard)

        devilsStatistics = DevilsStatistics(
            train_generator=self.train_generator,
            valid_generator=self.valid_generator,
            verbose=int(DEBUGGING)
        )
        callbacks.append(devilsStatistics)

        ############################################
        # Start the training process
        ############################################
        optimizer = self.create_optimizer()

        if self.loaded_optimizer_weights is not None:
            try:
                grad_vars = self.model.trainable_weights
                zero_grads = [tf.zeros_like(w) for w in grad_vars]

                # Apply gradients which don't do nothing with Adam
                optimizer.apply_gradients(zip(zero_grads, grad_vars))

                # Set the weights of the optimizer
                optimizer.set_weights(self.loaded_optimizer_weights)
                print("Loaded optimizer state!")
            except Exception as e:
                eprint("Could not load optimizer state due to: " + str(e))
        else:
            eprint("Could not load optimizer state!")

        ## Load callback state ##
        if self.callback_state:
            for state in self.callback_state:
                for cb in callbacks:
                    if state['string_repr'] in cb.__str__():
                        cb.load_config(state)
                        break

        print_seperator()
        print("Start Training" + (" at epoch " + str(FLAGS.initial_epoch + 1) if FLAGS.initial_epoch > 0 else " from the beginning") + " with a learning rate of " + str(optimizer.learning_rate.numpy()) +
              " and optimizer " + "Lookahead(" + optimizer._optimizer._name + ")" if optimizer._name == "Lookahead" else optimizer._name + ":")

        try:
            self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, run_eagerly=DEBUGGING, metrics=metrics)
            if self.val_model:
                self.val_model.compile(loss=loss, loss_weights=loss_weights, optimizer=self.create_optimizer(), run_eagerly=DEBUGGING, metrics=metrics)

            max_queue_size = max(4, FLAGS.worker * 2) #min(10, int(np.ceil(len(self.train_generator) * FLAGS.validate_every_nth / max(FLAGS.worker, 1))))
            print(f"Using {FLAGS.worker} worker and a queue size of {max_queue_size}")

            progbar = DevilsProgbarLogger(count_mode='steps', validation_steps=len(self.valid_generator))
            callbacks.append(progbar)

            self.model.fit(
                x=self.train_generator,
                steps_per_epoch=np.ceil(len(self.train_generator) * FLAGS.validate_every_nth),
                epochs=FLAGS.warmup_epochs + FLAGS.epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=self.valid_generator,
                validation_steps=len(self.valid_generator),
                validation_freq=1,
                class_weight=None,
                max_queue_size=max_queue_size,
                workers=FLAGS.worker,
                use_multiprocessing=FLAGS.use_multiprocessing,
                shuffle=True,
                initial_epoch=FLAGS.initial_epoch
            )
        except AttributeError as e:
            import traceback
            print_seperator()
            print(str(e))
            traceback.print_exc(file=sys.stdout)
            print_seperator()
            exit(-1)

    def prepare_model(self):
        if not self.valid_generator:
            self.train_generator, self.train_imgs, \
            self.valid_generator, self.valid_imgs = self.load_images_and_create_generators(generator_function=self.generator_function,
                                                                                           generator_config_function=self.generator_config_function,
                                                                                           load_all=True,
                                                                                           hard_multiplier=0)
        if not self.model or (not self.loaded_optimizer_weights and not self.callback_state):
            self.model, self.val_model = self.load_or_create_model(checkpoint=None)

    def convert_to_tflite(self, quantization=QUANTIZATION.NONE):
        # model = keras.models.load_model(file, compile=False)
        # model.summary()
        # feature_model = keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        # feature_model.summary()

        FLAGS.batch_size = 1
        self.model = None
        self.prepare_model()
        first_model = keras.Model(inputs=self.model.input,
                                  outputs=[self.model.outputs[0], self.model.get_layer('pre1').input])
        second_model = keras.Model(inputs=self.model.get_layer(first_model.output[1].name.split("/")[0]).output,
                                   outputs=[self.model.outputs[1]])
        self.model.summary(line_length=200)
        get_flops(self.model, "Model")
        first_model.summary(line_length=200)
        get_flops(first_model, "First")
        second_model.summary(line_length=200)
        get_flops(second_model, "second")
        try:
            convert_to_tflite(self.model if self.func_model is None else self.func_model, self.checkpoint_filepath, self.train_generator, quantization=quantization)
            convert_to_tflite(first_model, self.checkpoint_filepath + str("_1"), self.train_generator, quantization=quantization)
            convert_to_tflite(second_model, self.checkpoint_filepath + str("_2"), self.train_generator, quantization=quantization)
        except Exception as e:
            eprint(str(e))

    def create_individual_dirs(self):
        pass

    def save_single_prediction(self, filenames, batch_images, batch_predictions, i):
        raise NotImplemented("Please create save_single_prediction")

    def batch_prediction(self, idx, predict_batch_function, model, valid_generator, obj_threshold, nms_threshold, use_multithreading, training, resize_factor, t):
        batch_predictions, batch_images = predict_batch_function(model=model,
                                                                 valid_generator=valid_generator,
                                                                 idx=idx,
                                                                 obj_threshold=obj_threshold,
                                                                 nms_threshold=nms_threshold,
                                                                 draw=True,
                                                                 force=True,
                                                                 resize_factor=resize_factor)
        filenames = valid_generator.get_filenames(idx, basename=True)
        for i, filename in enumerate(filenames):
            if training:
                filenames[i] = os.path.join("training", filenames[i])
            else:
                filenames[i] = os.path.join("validation", filenames[i])

        assert len(filenames) == len(batch_images) == len(batch_predictions)

        func = partial(self.save_single_prediction, filenames, batch_images, batch_predictions)
        if use_multithreading:
            with ThreadPool(FLAGS.num_cores) as p:
                for _ in p.imap_unordered(func, range(len(filenames))):
                    t.update(1)
        else:
            for i in range(len(filenames)):
                func(i)
                t.update(1)
        return batch_predictions

    def predict_folder_batch(self, predict_batch_function, obj_threshold=0.5, nms_threshold=0.1, use_multithreading=True, resize_factor=1, predict_training=False):
        self.prepare_model()
        self.valid_generator.NAME = "predict_validation"
        self.valid_generator.set_augmentation_threshold(0.0)
        self.train_generator.NAME = "predict_training"
        self.train_generator.set_augmentation_threshold(0.0)
        self.train_generator.config["BATCH_SIZE"] = self.valid_generator.config["BATCH_SIZE"]

        self.out_dir = os.path.join("..",
                                    "predict_folder",
                                    self.model_name + "_" + str(FLAGS.load_checkpoint),
                                    str(int(obj_threshold[0] * 100)) if isinstance(obj_threshold, list) else str(int(obj_threshold * 100)))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.create_individual_dirs()
        print("Write prediction for images to: {}".format(self.out_dir))
        if predict_training:
            training_results = []
            with tqdm(total=len(self.train_generator)*self.train_generator.config["BATCH_SIZE"], desc="Predict images batches training", file=sys.stdout,
                      bar_format=TQDM_BAR_FORMAT, mininterval=1.0, smoothing=TQDM_SMOOTHING) as t:
                prediction_func = partial(self.batch_prediction,
                                          predict_batch_function=predict_batch_function,
                                          model=self.model,
                                          valid_generator=self.train_generator,
                                          obj_threshold=obj_threshold,
                                          nms_threshold=nms_threshold,
                                          use_multithreading=False,
                                          training=True,
                                          resize_factor=resize_factor,
                                          t=t)
                if use_multithreading and not DEBUGGING:
                    with ThreadPool(max(FLAGS.worker, 1)) as p:
                        for res in p.imap(prediction_func, range(len(self.train_generator))):
                            training_results.extend(res)
                else:
                    for idx in range(len(self.train_generator)):
                        res = prediction_func(idx)
                        training_results.extend(res)

            with open(os.path.join(self.out_dir, "training.pkl"), 'wb') as f:
                pickle.dump(training_results, f)

        validation_results = []
        with tqdm(total=len(self.valid_generator)*self.valid_generator.config["BATCH_SIZE"], desc="Predict images batches validation", file=sys.stdout,
                  bar_format=TQDM_BAR_FORMAT, mininterval=1.0, smoothing=TQDM_SMOOTHING) as t:
            prediction_func = partial(self.batch_prediction,
                                      predict_batch_function=predict_batch_function,
                                      model=self.model,
                                      valid_generator=self.valid_generator,
                                      obj_threshold=obj_threshold,
                                      nms_threshold=nms_threshold,
                                      use_multithreading=False,
                                      training=False,
                                      resize_factor=resize_factor,
                                      t=t)
            if use_multithreading and not DEBUGGING:
                with ThreadPool(max(FLAGS.worker, 1)) as p:
                    for res in p.imap(prediction_func, range(len(self.valid_generator))):
                        validation_results.extend(res)
            else:
                for idx in range(len(self.valid_generator)):
                    res = prediction_func(idx)
                    validation_results.extend(res)

        with open(os.path.join(self.out_dir, "validation.pkl"), 'wb') as f:
            pickle.dump(validation_results, f)

class BatchGenerator(keras.utils.Sequence):
    NAME = ""
    REMOVE_HORIZON = False

    LABEL_TO_SIZE = {
        "robot": (550, 275),
        "ball": (100, 100),
        "penalty cross": (100, 100),
    }

    def __init__(self, images,
                 config,
                 shuffle=True,
                 grayscale=False,
                 norm=True,
                 augmentation_threshold=0.0,
                 name="training",
                 num_cores=4,
                 use_multithreading=False):
        self.images = images
        self.images_cached = False
        self.config = config
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.norm = norm
        self.NAME = name
        self.num_cores = num_cores
        self.use_multithreading = use_multithreading

        self.set_augmentation_threshold(augmentation_threshold)

        if "REMOVE_HORIZON" in self.config:
            self.REMOVE_HORIZON = self.config["REMOVE_HORIZON"]

        if self.REMOVE_HORIZON:
            self.aug_pipe_scale = ResizeWithCameraMatrix({"height": self.config['IMAGE_H'], "width": self.config['IMAGE_W']})
        else:
            self.aug_pipe_scale = imgaug.augmenters.Resize({"height": self.config['IMAGE_H'], "width": self.config['IMAGE_W']}, interpolation="nearest")

        #self.generator = None

        if self.shuffle and len(images) > 0:
            self.shuffle_images()

    def set_augmentation_threshold(self, augmentation_threshold):
        self.augmentation_threshold = augmentation_threshold
        self.set_augmentation_pipeline(augmentation_threshold)
        self.aug_pool = self.aug_pipe.pool(processes=self.num_cores, seed=SEED, maxtasksperchild=1000)

    def shuffle_images(self):
        raise NotImplementedError("Please implement shuffle_images!!!")

    def shuffle_list(self, *ls):
        l = list(zip(*ls))
        random.shuffle(l)
        return (list(a) for a in zip(*l))

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_images()

    def __del__(self):
        pass

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def get_filename(self, i, basename=False):
        if basename:
            return os.path.basename(self.images[i]['filename'])
        else:
            return self.images[i]['filename']

    def get_filenames(self, idx, basename=False):
        filenames = []
        image_ids = self.determine_batch_bounds(idx)
        for i in image_ids:
            filenames.append(self.get_filename(i, basename=basename))
        return filenames

    def determine_batch_bounds(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        image_ids = []
        for i in range(l_bound, r_bound):
            image_ids.append(i)
        return image_ids

    def check_scale(self, height, width):
        if (height != self.config['IMAGE_H']) or (width != self.config['IMAGE_W']):
            return True
        return False

    def check_image_blurriness(self, image):
        blurriness = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / (image.shape[0] * image.shape[1])
        return blurriness

    def load_image(self, i):
        if self.grayscale:
            image = np.expand_dims(cv2.imread(self.images[i]['filename'], flags=0), -1)
        else:
            image = cv2.imread(self.images[i]['filename'])

        if image is None:
            eprint("Image could not be loaded!!! Check pickle Dataset!")
            assert(image is not None)

        return image[:, :, ::-1], self.check_scale(height=image.shape[0],
                                                   width=image.shape[1])

    def check_annotation_blurriness(self, i, xmin, ymin, xmax, ymax, image=None):
        if image is None:
            image, _ = self.load_image(i)
        cropped_image = image[int(max(0.0, ymin)): int(min(image.shape[1], ymax)),
                              int(max(0.0, xmin)): int(min(image.shape[0], xmax))]
        if cropped_image.shape[0] > 2 and cropped_image.shape[1] > 2:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            blurriness = cv2.Laplacian(gray, cv2.CV_64F).var() / (len(image) / len(cropped_image))
            return blurriness
        return 0

    def load_bounding_boxes_on_image(self, i, image=None):
        bboxes = []
        for obj in self.images[i]['object']:
            label_dict = {"label": obj['name'],
                          "visibility": obj['visibilityLevel'],
                          "concealed": obj["concealed"],
                          "blurriness": obj['blurriness']}
            if True: #obj['blurriness'] == 0:
                blurriness = self.check_annotation_blurriness(i, obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"], image)
                label_dict["blurriness"] = blurriness
            if 'teamColor' in obj:
                label_dict["teamcolor"] = obj['teamColor']
            bboxes.append(imgaug.BoundingBox(obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"], label=label_dict))

        return imgaug.BoundingBoxesOnImage(bboxes, shape=(self.images[i]['height'], self.images[i]['width'], 1 if self.grayscale else 3)), \
               self.check_scale(height=self.images[i]['height'], width=self.images[i]['width']) #TODO self.images[i]['width']

    def scale_image_and_bbox(self, aug_pipe_scale, image, bboxes_on_image, camera_intrinsics=None):
        if self.REMOVE_HORIZON:
            return aug_pipe_scale(image=image, bounding_boxes=bboxes_on_image, camera_intrinsics=camera_intrinsics)
        else:
            return aug_pipe_scale(image=image, bounding_boxes=bboxes_on_image)

    @cache
    def load_image_and_bbox_training(self, i):
        return self.load_image_and_bbox(i)

    @cache
    def load_image_and_bbox_validation(self, i):
        return self.load_image_and_bbox(i)

    def load_image_and_bbox(self, i):
        image, image_scale = self.load_image(i)
        bboxes_on_image, bboxes_on_image_scale = self.load_bounding_boxes_on_image(i, image)

        if image_scale is not None or bboxes_on_image_scale is not None:
            if image_scale is True or bboxes_on_image_scale is True:
                aug_pipe_scale = self.aug_pipe_scale.to_deterministic()
                try:
                    camera_intrinsics = {}
                    camera_intrinsics["rotation"] = np.array(self.images[i]["rotation"])
                    camera_intrinsics["translation"] = np.array(self.images[i]["translation"])
                    camera_intrinsics["focal_length"] = self.images[i]["focalLength"]
                    camera_intrinsics["optical_center"] = np.array(self.images[i]["opticalCenter"])
                    image, bboxes_on_image = self.scale_image_and_bbox(aug_pipe_scale, image, bboxes_on_image, camera_intrinsics=camera_intrinsics)
                except Exception as e:
                    print(str(e))
                    image, bboxes_on_image = self.scale_image_and_bbox(aug_pipe_scale, image, bboxes_on_image, camera_intrinsics=None)

            if not "imageBlurriness" in self.images[i] or self.images[i]["imageBlurriness"] <= 0:
                self.images[i]["imageBlurriness"] = self.check_image_blurriness(image)

        return image, bboxes_on_image

    #@profile
    def load_images_and_bboxes_from_batch(self, image_ids, use_multithreading=False):
        images = []
        bboxes_on_images = []

        func = self.load_image_and_bbox_training if self.NAME.endswith("training") else self.load_image_and_bbox_validation
        if use_multithreading and self.num_cores > 1 and not self.images_cached:
            with ThreadPool(self.num_cores) as p:
                for r in p.imap(func, image_ids):
                    images.append(r[0])
                    bboxes_on_images.append(r[1])
            if self.NAME.endswith("training"):
                self.images_cached = (self.load_image_and_bbox_training.cache_info().hits >= 1)
            else:
                self.images_cached = (self.load_image_and_bbox_validation.cache_info().hits >= 1)
        else:
            for i in image_ids:
                image, bbox = func(i)
                images.append(image)
                bboxes_on_images.append(bbox)

        return images, bboxes_on_images

    def load_information_dict(self, i):
        return self.images[i]

    def augment_images_and_bboxes_iterable(self, images, bboxes_on_images, i):
        img, bbox = self.aug_pipe(image=images[i], bounding_boxes=bboxes_on_images[i])
        return img, bbox

    def augment_images_and_bboxes(self, images, bboxes_on_images, augmentation_threshold, use_multithreading=False):
        if augmentation_threshold > 0.0 and self.images_cached:
            assert(len(images) == len(bboxes_on_images))

            probs = np.random.rand(len(images))
            selected_indizes = np.where(probs < augmentation_threshold)[0]

            selected_images = []
            selected_bboxes_on_images = []
            for i in selected_indizes:
                selected_images.append(images[i])
                selected_bboxes_on_images.append(bboxes_on_images[i])

            batch_size = int(np.floor(len(selected_images) / self.num_cores))
            if batch_size < 5:
                selected_images_batches = [selected_images]
                selected_bboxes_on_images_batches = [selected_bboxes_on_images]
            else:
                selected_images_batches = chunks(selected_images, batch_size)
                selected_bboxes_on_images_batches = chunks(selected_bboxes_on_images, batch_size)

            batches = [imgaug.UnnormalizedBatch(images=selected_images_batches[i], bounding_boxes=selected_bboxes_on_images_batches[i]) for i in range(len(selected_images_batches))]

            selected_images_aug = []
            selected_bboxes_on_images_aug = []
            if use_multithreading and self.num_cores > 0:
                batches_aug = self.aug_pool.map_batches(batches)
                for batch_aug in batches_aug:
                    selected_images_aug.extend(batch_aug.images_aug)
                    selected_bboxes_on_images_aug.extend(batch_aug.bounding_boxes_aug)
            else:
                for batch in batches:
                    batch_aug = self.aug_pipe.augment_batch_(batch)
                    selected_images_aug.extend(batch_aug.images_aug)
                    selected_bboxes_on_images_aug.extend(batch_aug.bounding_boxes_aug)

            images_aug = images
            bboxes_on_images_aug = bboxes_on_images
            for i in range(len(selected_images_aug)):
                index = selected_indizes[i]
                images_aug[index] = selected_images_aug[i]
                bboxes_on_images_aug[index] = selected_bboxes_on_images_aug[i]

            assert (len(images) == len(images_aug))
            assert (len(bboxes_on_images) == len(bboxes_on_images_aug))
            return images_aug, bboxes_on_images_aug
        else:
            return images, bboxes_on_images

    def filter_bboxes(self, bounding_boxes, filter_too_small=True, min_factor=0.02):
        bboxes = bounding_boxes
        number_of_filtered_annotations = 0

        min_pixel_width_for_bbox = max(3, round(self.config['IMAGE_W'] * min_factor))
        min_pixel_height_for_bbox = max(3, round(self.config['IMAGE_H'] * min_factor))

        for i, bbox in enumerate(list(bboxes)):
            temp_bbox = bbox.clip_out_of_image((self.config['IMAGE_W'], self.config['IMAGE_H']))
            remove = False
            label = temp_bbox.label["label"]
            vis_lvl = temp_bbox.label["visibility"]
            concealed = temp_bbox.label["concealed"]
            ### Check label ###
            if label not in self.config['LABELS']:
                remove = True
            ### Check visibility level ###
            elif vis_lvl >= 5:
                remove = True
            ### Check concealed ###
            elif concealed and vis_lvl >= 4:
                remove = True
            ### Check too small bounding box ###
            elif filter_too_small and (np.ceil(temp_bbox.width) < min_pixel_width_for_bbox or np.ceil(temp_bbox.height) < min_pixel_height_for_bbox):
                remove = True
            if remove:
                bboxes.remove(bbox)
                number_of_filtered_annotations += 1

        if self.config["DEBUG"] and number_of_filtered_annotations > 0:
            print("Removed {} annotations".format(number_of_filtered_annotations))
        return bboxes

    def set_augmentation_pipeline(self, augmentation_threshold):
        raise NotImplementedError("Please implement an augmentation_pipeline!!!")

    def __getitem__(self, idx):
        raise NotImplementedError("Please implement __getitem__!!!")
