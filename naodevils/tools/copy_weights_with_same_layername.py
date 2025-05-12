from utils import *
from utils.util_functions import ask_file
from utils.setup_tensorflow_utils import tf

last_folder = os.path.join(os.getcwd(), "..", "models")
try:
    filename_orig = ask_file(initial_dir=last_folder, filetypes=[("h5 files", "*.h5")])
    # last_folder = os.path.basename(filename_orig)
    while True:
        filename_new = ask_file(initial_dir=last_folder, filetypes=[("h5 files", "*.h5")])

        # Convert the model.
        model_orig = tf.keras.models.load_model(filename_orig, compile=False)
        model_new = tf.keras.models.load_model(filename_new, compile=False)

        for layer_orig in model_orig.layers:
          for layer_new in model_new.layers:
            if layer_orig.name == layer_new.name:
                if layer_orig.output_shape[1:] == layer_new.output_shape[1:]:
                    w = layer_orig.get_weights()
                    layer_new.set_weights(w)
                    print("Transferred weights to " + str(layer_new.name))
                    break
                else:
                    print("Skipped weights transfer to " + str(layer_new.name))
        model_new.save(filename_new, overwrite=True)

except KeyboardInterrupt:
  exit(0)
except Exception as e:
  eprint(e)