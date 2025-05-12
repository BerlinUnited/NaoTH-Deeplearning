import enum
from utils import *

###########################
### Setup Logging Level ###
###########################
class LOGGING_LEVEL(enum.Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    FATAL = 50

tf_level = LOGGING_LEVEL.ERROR
tf_level = tf_level.value

###################################
### Setup Environment Variables ###
###################################
os.environ['GCS_READ_CACHE_DISABLED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_level)
os.environ['AUTOGRAPH_VERBOSITY'] = str(tf_level)

os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '256'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

#########################
### Import Tensorflow ###
#########################
eprint("Starting tensorflow import, please wait!")
import tensorflow as tf
tf_version = tf.__version__
if int(tf_version.split(".")[0]) != 2 or int(tf_version.split(".")[1]) < 10:
    eprint("This framework requires Tensorflow version 2.10. or higher")
    exit(-1)
tf.get_logger().setLevel(tf_level)
tf.autograph.set_verbosity(tf_level)
import tensorflow_addons
keras = tf.keras

#####################
### Check for GPU ###
#####################
gpus = tf.config.experimental.list_physical_devices('GPU')
eprint(f'Using TensorFlow {tf_version}, GPUs available? : {len(gpus)}')
if not gpus or len(gpus) < 1:
    correct = input("You are not using any GPU is this correct? [Y/n]")
    if correct.lower() == "n":
        exit(-1)

if DEBUGGING:
    tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()
    tf.autograph.experimental.do_not_convert()
else:
    tf.config.run_functions_eagerly(False)

#################
### Decorator ###
#################
def with_tensorboard(func):
    from tensorboard import program
    from tensorboard.util import tb_logging
    import threading, logging

    def wrapper(*args, **kwargs):
        if DEBUGGING:
            result = func(*args, **kwargs)
        else:
            eprint('Starting Tensorboard on ' + args[0].current_summary_dir + ':\n\n')
            tb_logging.get_logger().setLevel(tf_level)
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', args[0].current_summary_dir, '--host', '0.0.0.0'])
            # url = tb.launch()

            server = tb._make_server()
            thread = threading.Thread(target=server.serve_forever, name='TensorBoard')
            thread.start()
            url = server.get_url()
            eprint('TensorBoard running at %s \n' % url.replace('0.0.0.0', 'localhost'))

            logging.getLogger('werkzeug').setLevel(tf_level)
            # print([logging.getLogger(name) for name in logging.root.manager.loggerDict])

            result = func(*args, **kwargs)
            server.shutdown()
            thread.join()
            if not thread.is_alive():
                eprint('TensorBoard stopped!\n')
        return result

    return wrapper

####################
### Quantization ###
####################
class QUANTIZATION(enum.Enum):
    NONE = 0
    FP16 = 1
    DYNAMIC_INT8 = 2
    INT8_FLOAT_FALLBACK = 3
    INT8 = 4
    MODEL = 5