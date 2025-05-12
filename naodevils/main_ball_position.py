if __name__ == "__main__":
    from absl import app
    from utils import *
    from utils.hardware_utils import available_cpu_count
    from utils.setup_tensorflow_utils import *
    from utils.tensorflow_utils import QUANTIZATION

    from architectures.ball_yolo.ball_yolo_neural_network import BALL_CNN

    def main(_argv):
        if DEBUGGING:
            FLAGS.num_cores = 1
            FLAGS.use_multiprocessing = False
            FLAGS.worker = 0
            FLAGS.gpu_usage = 0.0
        else:
            FLAGS.worker = 2
            FLAGS.num_cores = int(available_cpu_count() / FLAGS.worker)
            FLAGS.use_multiprocessing = False
            FLAGS.gpu_usage = 1.0

        FLAGS.validate_every_nth = 1.0
        FLAGS.augmentation = -1.0
        FLAGS.learning_rate = 0.001
        FLAGS.epochs = 10000
        FLAGS.f_beta = 1 / 15
        FLAGS.warmup_epochs = 0
        FLAGS.batch_size = 256

        FLAGS.img_dir = os.path.join(os.getcwd(), "data", "ball", "patches", "training")
        FLAGS.val_dir = os.path.join(os.getcwd(), "data", "ball", "patches", "validation")
        FLAGS.training_validation_rate = 0.0

        FLAGS.monitor_1_metric = "val_fscore"
        FLAGS.monitor_2_metric = "val_loss"

        FLAGS.iou_threshold = 0.25
        model_files = []
        model_files.append(("ball_position_v23.cfg", 5348, FLAGS.iou_threshold))

        #os.remove(os.path.join(FLAGS.img_dir, "dataset.pkl"))
        FLAGS.hard_multiplier = 0
        for model_file in model_files:
            FLAGS.model_file = os.path.join("configs", "ball", model_file[0])
            if model_file[1]: FLAGS.load_checkpoint = model_file[1]

            ball_position = BALL_CNN(pickleDataset=os.path.join(FLAGS.img_dir, "dataset.pkl"))
            ball_position.train()
            # ball_position.predict_folder_batch(predict_training=True, obj_threshold=model_file[2], resize_factor=5, use_multithreading=True)

            tf.keras.backend.clear_session()

    ########################################
    ############# Start App ################
    ########################################
    try:
        app.run(main)
    except SystemExit:
        pass
    except Exception as e:
        import traceback
        print_seperator()
        print(str(e))
        traceback.print_exc(file=sys.stdout)
        print_seperator()