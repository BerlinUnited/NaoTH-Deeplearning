import os
from absl import flags

aD = os.path.dirname(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)))
#############################
### files and directories ###
#############################
flags.DEFINE_string('model_file', os.path.join(aD, "configs", "sample_model.cfg"), 'file, from which the keras model is loaded')
flags.DEFINE_string('img_dir', os.path.join(aD, "data", "training") + os.path.sep, 'folder, where the training images (and annotations) are located, they can be located in sub-directories')
flags.DEFINE_string('val_dir', os.path.join(aD, "data", "validation") + os.path.sep, 'folder, where the validation images (and annotations) are located, they can be located in sub-directories')
flags.DEFINE_string('trained_models_dir', os.path.join(aD, "models") + os.path.sep, 'folder, where the trained models are located')
flags.DEFINE_string('summary_dir', os.path.join(aD, "models", "summary") + os.path.sep, 'folder, where the summary for tensorboard is saved')

######################
### hyperparameter ###
######################
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('validation_batch_size', -1, 'validation batch size')
flags.DEFINE_integer('epochs', 5000, 'number of epochs to train')
flags.DEFINE_integer('warmup_epochs', 3, 'number of initial epochs during which the sizes of the bounding boxes in each cell is forced to match the sizes of the anchors (> 0 speeds up the early training process)')
flags.DEFINE_integer('validate_every_nth', 1, 'validate every nth epoch, multiplies the epoch length for training')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('f_beta', 1/4, 'F-beta Score')
flags.DEFINE_float('training_validation_rate', 0.8, 'how many percent (0-1) of the images are used for training and for validation (this has only an effect if val_dir is empty)')
flags.DEFINE_float('augmentation', 0.0, 'how many percent (0-1) of the images will be augmented (!!!SLOWER!!!)')
flags.DEFINE_float('iou_threshold', 0.5, 'IoU threshold to determine a detection')

##################################
### multi-threading/processing ###
##################################
flags.DEFINE_boolean('use_multiprocessing', True, 'use the keras multiprocessing for the batches')
flags.DEFINE_integer('worker', 3, 'defines the number of threads/processes to spawn for the batch generator')
flags.DEFINE_integer('num_cores', 4, 'number of physical cpu cores')
flags.DEFINE_float('gpu_usage', 1.0, 'gpu usage in percent, 1.0 enables automatic memory growth')

########################
### other parameters ###
########################
flags.DEFINE_integer('load_checkpoint', 0, 'loads the nearest checkpoint around the specified epoch number (use 0 for loading the latest checkpoint, -1 for not loading a checkpoint)')
flags.DEFINE_integer('hard_multiplier', 5, 'defines how many times hard examples are added to the training dataset')
flags.DEFINE_integer('model_evaluation_start_epoch', 5, 'determines at which epoch the mAP evaluation is triggered if the iou acc increases')
flags.DEFINE_boolean('search_subfolders', True, 'search for checkpoints in subfolders')
flags.DEFINE_boolean('remove_horizon', False, 'remove horizon from scaled images')
flags.DEFINE_boolean('use_bottom_as_center', False, 'use the bottom instead of the center of the bbox (y-value)')
flags.DEFINE_enum('annotation_format', 'protobuf', ['protobuf', 'xml'], 'format for the annotations')
flags.DEFINE_enum('monitor_1_metric', 'val_loss', ['loss', 'avg_iou', 'mean_avg_precision', 'fscore', 'val_loss', 'val_avg_iou', 'val_mean_avg_precision', 'val_fscore'], 'first monitored metric for checkpoints (callbacks)')
flags.DEFINE_enum('monitor_2_metric', 'val_fscore', ['loss', 'avg_iou', 'mean_avg_precision', 'fscore', 'val_loss', 'val_avg_iou', 'val_mean_avg_precision', 'val_fscore'], 'second monitored metric for checkpoints (callbacks)')
flags.DEFINE_multi_string('val_dir_folder_filter_list', [], 'use only this folders in val_dir')
flags.DEFINE_multi_string('img_dir_folder_filter_list', [], 'use only this folders in img_dir')
flags.DEFINE_multi_float('thresholds', [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], 'used thresholds for the pr-curve and everything related')

###############################
### possible flags from cfg ###
###############################
flags.DEFINE_integer('initial_epoch', 0, '!!!internal!!! initial epoch to start/resume the training')
flags.DEFINE_enum('color_model', 'rgb', ['rgb', 'y'], 'color model')
flags.DEFINE_boolean('norm', True, 'should the pixels be normalized in the range between 0.0 and 1.0')
flags.DEFINE_multi_float('anchors', [], 'anchors to use for the bounding boxes')
flags.DEFINE_integer('num_bounding_boxes', 1, 'number of bounding boxes')
flags.DEFINE_enum('trainer', 'adam', ['adam', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'sgd'], 'the optimizer/trainer to use')
flags.DEFINE_multi_string('label_names', [], 'labes to use')
flags.DEFINE_float('object_scale', 0.0, '')
flags.DEFINE_float('noobject_scale', 0.0, '')
flags.DEFINE_float('class_scale', 0.0, '')
flags.DEFINE_float('coord_scale', 0.0, '')
flags.DEFINE_float('wh_scale', 0.0, '')

flags.DEFINE_enum('model_type', 'Sequential', ['Sequential', 'Model'], 'keras model type')
flags.DEFINE_integer('height', 480, 'image input height')
flags.DEFINE_integer('width', 640, 'image input width')
flags.DEFINE_integer('channels', 3, 'image input channels')
flags.DEFINE_integer('grid_height', 7, 'image input height')
flags.DEFINE_integer('grid_width', 10, 'image input width')

##################
### Validators ###
##################
flags.register_validator('model_file',
                         lambda filename: filename.endswith(".cfg"),
                         message='--model_file must end with .cfg')

def safe_flags_to_file(file="utils/flags.saved"):
    with open(file, "w") as flagsSaveFile:
        flagsSaveFile.write(get_non_default_flags())

def get_non_default_flags():
    flags_string = ""
    for unused_module_name, fls in flags.FLAGS.flags_by_module_dict().items():
        for f in fls:
            if f.value != f.default:
                flags_string += f.serialize() + '\n'
    return flags_string[:-1]

def get_flags_as_dict():
    flags_dict = {}
    for flag in flags.FLAGS.flags_by_module_dict()['utils.flags']:
        flags_dict[flag.name] = flag.value
    return flags_dict
