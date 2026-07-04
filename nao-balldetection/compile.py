import pickle
from pathlib import Path
import tensorflow as tf
from train  import WeightedBinaryCrossentropy
from tensorflow.keras.models import load_model
#import utility_functions.metrics as metrics_module
from onbcg import NaoTHCompiler  # can throw linter warnings, but python3 can handle imports like that
from inspect import isclass, isfunction

DATA_DIR = Path(Path(__file__).parent.absolute() / "data").resolve()

def main(config_name):
    images = {}
    data_file = str(DATA_DIR / "training.pkl")
    with open(data_file, "rb") as f:
        images["mean"] = pickle.load(f)
        images["images"] = pickle.load(f)
        images["y"] = pickle.load(f)
    print(images["mean"])
    model = tf.keras.models.load_model('rc26_classification_color_32_training.h5')
    #model = load_model(model_path)

    compiler = NaoTHCompiler(images, model, code_path="test2.cpp", unroll_level=2, arch="sse3",
                                 test_binary=False)
    compiler.keras_compile()

if __name__ == '__main__':
    main("stella_config")
