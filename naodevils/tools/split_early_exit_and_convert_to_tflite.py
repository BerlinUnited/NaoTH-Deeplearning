import json
from utils import *
from utils.util_functions import ask_file
from utils.setup_tensorflow_utils import tf

last_folder = os.path.join(os.getcwd(), "..", "model")
try:
  filename = ask_file(initial_dir=last_folder, filetypes=[("h5 files", "*.h5")])
  if filename != "":
    # Convert the model.
    model = tf.keras.models.load_model(filename, compile=False)
    if model.__class__ == tf.keras.models.Sequential:
      model = tf.keras.Model(model.inputs, model.outputs)

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
    new_model.summary(line_length=200)

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TF Lite model.
    tflite_name = filename[:-3] + ".tflite"
    with tf.io.gfile.GFile(tflite_name, 'wb') as f:
      f.write(tflite_model)
    print("Converted original model to tflite: " + tflite_name)

    x = input('Enter layername to split: ') # 02_LeakyReLU

    clone = tf.keras.models.clone_model(new_model)
    clone.set_weights(new_model.get_weights())
    cut_layer = clone.get_layer(name=x).output

    early_exit_model = tf.keras.Model(inputs=clone.input, outputs=[clone.outputs[1], cut_layer])
    early_exit_model.save(filename[:-3] + "_early_exit.h5", overwrite=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(early_exit_model)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TF Lite model.
    tflite_name = filename[:-3] + "_early_exit.tflite"
    with tf.io.gfile.GFile(tflite_name, 'wb') as f:
      f.write(tflite_model)
    print("Converted early_exit model to tflite: " + tflite_name)

    ####################################################################################################

    clone = tf.keras.models.clone_model(new_model)
    clone.set_weights(new_model.get_weights())
    cut_layer = clone.get_layer(name=x).output

    remaining_model = tf.keras.Model(inputs=cut_layer, outputs=clone.outputs[0])
    remaining_model.save(filename[:-3] + "_remaining.h5", overwrite=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(remaining_model)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TF Lite model.
    tflite_name = filename[:-3] + "_remaining.tflite"
    with tf.io.gfile.GFile(tflite_name, 'wb') as f:
      f.write(tflite_model)
    print("Converted remaining model to tflite: " + tflite_name)

except KeyboardInterrupt:
  exit(0)
except Exception as e:
  eprint(e)