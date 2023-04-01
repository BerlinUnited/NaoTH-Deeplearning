import tensorflow.keras as keras
from pathlib import Path

from onbcg import NaoTHCompiler  # can throw linter warnings, but python3 can handle imports like that


def convert_to_naoth(input_file_path):
    model = keras.models.load_model(input_file_path)

    output_path = Path(input_file_path).with_suffix(".cpp")
    compiler = NaoTHCompiler(model, code_path=str(output_path), unroll_level=2, arch="sse3",
                             test_binary=False)
    compiler.keras_compile()


if __name__ == '__main__':
    convert_to_naoth("dummy_model2.h5")
