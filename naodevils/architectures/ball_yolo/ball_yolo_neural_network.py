from matplotlib import pyplot as plt
import itertools
import shutil

from utils import *
from utils.callback_utils import IncreaseAugmentationOnPlateau, DevilsCallback

from architectures import *
from architectures.ball_yolo.ball_yolo_batch_generator import BatchGeneratorBall, make_generator_config_ball
from architectures.ball_yolo.ball_yolo_loss import BallYoloLoss
from architectures.ball_yolo.ball_yolo_model_utils import predict_batch_ball

class BALL_CNN(NeuralNetwork):

    def __init__(self, checkpoint=None, create_dot=False, pickleDataset=""):
        load_all = True
        generator_function = BatchGeneratorBall
        generator_config_function = make_generator_config_ball
        super(BALL_CNN, self).__init__(generator_function=generator_function,
                                 generator_config_function=generator_config_function,
                                 name="ball",
                                 checkpoint=checkpoint,
                                 create_dot=create_dot,
                                 load_all=load_all,
                                 pickleDataset=pickleDataset)

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def create_model(self):
        if self.loaded_model is None:
            return self.json_model
        else:
            return self.loaded_model

    def train(self):
        callbacks = []
        if FLAGS.augmentation <= 0.0:
            iaop = IncreaseAugmentationOnPlateau(train_generator=self.train_generator,
                                                 monitor=FLAGS.monitor_1_metric,
                                                 mode='max',
                                                 increment=0.01,
                                                 patience=25,
                                                 min_delta=0.001,
                                                 cooldown=5,
                                                 verbose=1)
            callbacks.append(iaop)

        checkpoint = DevilsCallback(
            evaluate=None,
            checkpoint_filepath=self.checkpoint_filepath,
            file_name="{epoch_string}-loss_{loss:.3f}_-_fscore_{fscore:.3f}_-_deviation_{deviation:.3f}_-_val_loss_{val_loss:.3f}_-_val_fscore_{val_fscore:.3f}_-_val_deviation_{val_deviation:.3f}.h5",
            monitor_1=FLAGS.monitor_1_metric,
            mode_1='max',
            monitor_2=FLAGS.monitor_2_metric,
            mode_2='min',
            verbose=1,
            initial_step=FLAGS.initial_epoch * len(self.train_generator),
            save_best_only=True,
            save_weights_only=False,
            period=1,
            log_dir=self.current_summary_dir,
            warmup_epochs=FLAGS.warmup_epochs,
            callbacks_to_save=callbacks
        )
        callbacks.append(checkpoint)

        loss = BallYoloLoss(width=FLAGS.width, height=FLAGS.height,
                            threshold=FLAGS.iou_threshold,
                            f_beta=FLAGS.f_beta,
                            coord_scale=FLAGS.coord_scale,
                            object_scale=FLAGS.object_scale)
        metrics = [loss.fscore, loss.deviation, loss.precision, loss.recall, loss.loss_conf, loss.loss_xy]

        super(BALL_CNN, self).train(loss=loss.ball_cnn_loss,
                              metrics=metrics,
                              individual_callbacks=callbacks)

    def create_individual_dirs(self):
        self.out_dir_false = os.path.join(self.out_dir, "0")
        self.out_dir_false_negative = os.path.join(self.out_dir_false, "FN")
        self.out_dir_false_negative_blurred = os.path.join(self.out_dir_false_negative, "Blurred")
        self.out_dir_false_negative_deviation = os.path.join(self.out_dir_false_negative, "Deviation")

        self.out_dir_true = os.path.join(self.out_dir, "1")
        self.out_dir_blurred = os.path.join(self.out_dir_true, "Blurred")
        self.out_dir_false_positive = os.path.join(self.out_dir_true, "FP")
        self.out_dir_false_positive_deviation = os.path.join(self.out_dir_true, "Deviation")

        if os.path.exists(self.out_dir):
            try:
                folder_to_delete = [x[0] for x in os.walk(self.out_dir, topdown=False)]
                with tqdm(total=len(folder_to_delete), desc='Deleting existing directory:', file=sys.stdout) as t:
                    for dir in folder_to_delete:
                        t.set_postfix({'current_dir': str(dir.replace(self.out_dir, ""))})
                        if os.path.exists(dir):
                            shutil.rmtree(dir, ignore_errors=True, onerror=handleError)
                        t.update()
            except:
                eprint('Error while deleting directory')

        mode = 0o770
        exist_ok = True
        os.makedirs(self.out_dir, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_true, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_blurred, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false_positive, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false_positive_deviation, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false_negative, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false_negative_blurred, mode=mode, exist_ok=exist_ok)
        os.makedirs(self.out_dir_false_negative_deviation, mode=mode, exist_ok=exist_ok)

    def predict_folder_batch(self, predict_batch_function=None, obj_threshold=0.5, nms_threshold=0.1, use_multithreading=True, resize_factor=1, predict_training=False):
        predict_batch_function = predict_batch_ball
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.deviations = []
        super(BALL_CNN, self).predict_folder_batch(predict_batch_function=predict_batch_function,
                                                   obj_threshold=obj_threshold,
                                                   nms_threshold=nms_threshold,
                                                   use_multithreading=use_multithreading,
                                                   resize_factor=resize_factor,
                                                   predict_training=predict_training)

        self.plot_confusion_matrix(cm=np.array([[self.tp, self.fn],
                                                [self.fp, self.tn]]),
                                   normalize=False,
                                   target_names=['1', '0'],
                                   title=self.model_name + "_" + str(FLAGS.load_checkpoint))

    def save_single_prediction(self, filenames, batch_images, batch_predictions, i):
        extension = ".jpg"
        predicted_filename = "-_" + os.path.basename(filenames[i][:-4]) + extension
        add_blurriness = False
        confidence = batch_predictions[i][2] * 100.0
        deviation = batch_predictions[i][3]
        blurriness = batch_predictions[i][6]
        loss = batch_predictions[i][7]
        max_deviation = 1.0
        min_blurriness = 1000
        if batch_predictions[i][0] == 1:
            saving_dir = self.out_dir_true
            add_deviation = True
            if batch_predictions[i][0] != batch_predictions[i][1]:
                saving_dir = self.out_dir_false_positive
                add_deviation = False
            elif deviation >= max_deviation:
                saving_dir = self.out_dir_false_positive_deviation
            elif blurriness < min_blurriness:
                saving_dir = self.out_dir_blurred
                add_blurriness = True

        else:
            saving_dir = self.out_dir_false
            add_deviation = False
            if batch_predictions[i][1] == 1.0:
                saving_dir = self.out_dir_false_negative
                add_deviation = True
                if deviation >= max_deviation:
                    saving_dir = self.out_dir_false_negative_deviation
                elif blurriness < min_blurriness:
                    saving_dir = self.out_dir_false_negative_blurred
                    add_blurriness = True


        if add_blurriness:
            predicted_filename = '%08.2f' % (blurriness) + "b_" + predicted_filename

        predicted_filename = '%06.2f' % (confidence) + "%_" + predicted_filename

        predicted_filename = '%06.3f' % (loss) + "l_" + predicted_filename

        if add_deviation:
            predicted_filename = '%06.2f' % (deviation) + "d_" + predicted_filename

        predicted_filename = os.path.join(saving_dir, predicted_filename)
        pred_image = batch_images[i]

        ret = cv2.imwrite(predicted_filename, cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR))
        if not ret:
            eprint(predicted_filename + " could NOT be written!!!")

        if deviation > 0.0:
            self.deviations.append(deviation)

        if saving_dir == self.out_dir_true or \
           saving_dir == self.out_dir_false_positive_deviation or \
           saving_dir == self.out_dir_blurred:
            self.tp += 1
        elif saving_dir == self.out_dir_false:
            self.tn += 1
        elif saving_dir == self.out_dir_false_negative or \
             saving_dir == self.out_dir_false_negative_blurred or \
             saving_dir == self.out_dir_false_negative_deviation:
            self.fn += 1
        elif saving_dir == self.out_dir_false_positive:
            self.fp += 1

    def plot_confusion_matrix(self,
                              cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        # np.array([[self.tp, self.fn],
        #           [self.fp, self.tn]]
        precision = cm[0][0] / (cm[0][0] + cm[1][0])
        recall = cm[0][0] / (cm[0][0] + cm[0][1])

        deviations = np.asarray(self.deviations)
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.6f}; misclass={:0.6f}\nprecision={:0.6f}; recall={:0.6f}\ndeviation(mean)={:0.6f}; deviation(std)={:0.6f}'.format(accuracy, misclass, precision, recall, mean_deviation, std_deviation))
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, title))