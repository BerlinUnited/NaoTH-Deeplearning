from utils import *

#######################
### Seed everything ###
#######################
SEED = 1337

def setSeed(seed):
    imported_modules = [module.__name__ for module in sys.modules.values() if module]
    imported_modules = sorted(imported_modules)
    # if DEBUGGING:
    #     print('The list of imported Python modules are :', imported_modules)

    if "os" in imported_modules:
        # 1. Set `PYTHONHASHSEED` and other environment variables at a fixed value
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    if "random" in imported_modules:
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        import random
        random.seed(seed)
    if "numpy" in imported_modules:
        # 3. Set `numpy` pseudo-random generator at a fixed value
        import numpy as np
        np.random.seed(seed)
    if "imgaug" in imported_modules:
        # 4. Set `imgaug` pseudo-random generator at a fixed value
        import imgaug
        imgaug.seed(seed)
    if "tensorflow" in imported_modules:
        # 5. Set `tensorflow` pseudo-random generator at a fixed value
        import tensorflow as tf
        tf.random.set_seed(seed)
    # Debug print
    eprint("Set seed to " + str(seed))