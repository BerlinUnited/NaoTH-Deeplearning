import json
from utils import *
from utils.util_functions import ask_file
from utils.setup_tensorflow_utils import tf

last_folder = os.getcwd()
try:
  while True:
    filename = ask_file(initial_dir=last_folder, filetypes=[("h5 files", "*.h5")])
    if filename == "":
      break
    last_folder = os.path.basename(filename)

    # Convert the model.
    model = tf.keras.models.load_model(filename, compile=False)

    json_dict = json.loads(model.to_json())
    input_shapes = []
    input_layers = json_dict["config"]["input_layers"]
    for input_layer in input_layers:
      for layer in json_dict["config"]["layers"]:
        if layer['name'] == input_layer[0]:
          input_shape = layer["config"]["batch_input_shape"]
          input_shape[0] = 1
          input_shapes.append(input_shape)
          break
    json_model = json.dumps(json_dict, indent=2)

    new_model = tf.keras.models.model_from_json(json_model)
    new_model.set_weights(model.get_weights())
    new_model.summary()
    new_model.save(filename[:-3] + "_tflite.h5", overwrite=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TF Lite model.
    tflite_name = filename[:-3] + '.tflite'
    with tf.io.gfile.GFile(tflite_name, 'wb') as f:
      f.write(tflite_model)
    print("Converted model to tflite: " + tflite_name)
except KeyboardInterrupt:
  exit(0)
except Exception as e:
  eprint(e)