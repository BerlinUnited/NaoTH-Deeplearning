import json
import numpy as np
import pickle
from utils import *
from utils.flags import get_non_default_flags
from utils.setup_tensorflow_utils import *
from utils.util_functions import compute_visibility, find_nth
from utils.annotation_utils import parse_annotation_protobuf

custom_objects = {}

PATCHES_DICT = {
    0: {
        "zoom_out_factor": 1.0,
        "object_heigth": 550,
        "object_width": 275,
        "use_bottom": True,
    },
    1: {
        "zoom_out_factor": 2.0,
        "object_heigth": 100,
        "object_width": 100,
        "use_bottom": False,
    },
    2: {
        "zoom_out_factor": 1.5,
        "object_heigth": 100,
        "object_width": 100,
        "use_bottom": False,
    }
}


def infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


def calculate_visibility_and_area_ratio(anotation, det_bbox, image_height, image_width, width_height_factor=1.0, initial_visibility=0):
    extra_folder = False

    ann_width_half = (anotation[2] - anotation[0]) / 2.0
    ann_height_half = (anotation[3] - anotation[1]) / 2.0
    ann_center_x = anotation[0] + ann_width_half
    ann_center_y = anotation[1] + ann_height_half

    ann_width_half = round(ann_width_half)
    ann_height_half = round(ann_height_half)
    ann_center_x = round(ann_center_x)
    ann_center_y = round(ann_center_y)

    if ann_height_half == 0.0 or ann_width_half == 0.0:
        return -1.0, 0.0, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float), extra_folder

    width_height_ratio = ann_width_half / ann_height_half

    ann_xmin = int(ann_center_x - ann_width_half)
    ann_xmax = int(ann_center_x + ann_width_half)
    ann_ymin = int(ann_center_y - ann_height_half)
    ann_ymax = int(ann_center_y + ann_height_half)

    if width_height_ratio != width_height_factor:
        if anotation[0] == 0:
            calculated_width = (ann_height_half * 2.0) * width_height_factor
            remaining_width = calculated_width - (ann_width_half * 2.0)
            ann_xmin -= remaining_width
            extra_folder = True
        elif anotation[2] == image_width:
            calculated_width = (ann_height_half * 2.0) * width_height_factor
            remaining_width = calculated_width - (ann_width_half * 2.0)
            ann_xmax += remaining_width
            extra_folder = True
        elif anotation[1] == 0:
            calculated_height = (ann_width_half * 2.0) / width_height_factor
            remaining_height = calculated_height - (ann_height_half * 2.0)
            ann_ymin -= remaining_height
            extra_folder = True
        elif anotation[3] == image_height:
            calculated_height = (ann_width_half * 2.0) / width_height_factor
            remaining_height = calculated_height - (ann_height_half * 2.0)
            ann_ymax += remaining_height
            extra_folder = True
        else:
            if ann_width_half > ann_height_half:
                ann_ymin = int(ann_ymax - round(ann_width_half * 2.0 / width_height_factor))
                #ann_ymax = int(ann_center_y + np.ceil(ann_width_half / width_height_factor))
            else:
                ann_xmin = int(ann_center_x - np.floor(ann_height_half * width_height_factor))
                ann_xmax = int(ann_center_x + np.ceil(ann_height_half * width_height_factor))
    ratio_ann = (ann_xmax - ann_xmin) / np.maximum((ann_ymax - ann_ymin), np.finfo(float).eps)
    assert ratio_ann == width_height_factor

    # overlap = compute_overlap(np.array([[det_xmin, det_ymin, det_xmax, det_ymax]]), np.array([[ann_xmin, ann_ymin, ann_xmax, ann_ymax]]))[0][0]
    ann_bbox = np.array([ann_xmin, ann_ymin, ann_xmax, ann_ymax], dtype=np.float)
    visibility, area_det, area_ann = compute_visibility(det_bbox, ann_bbox)

    if initial_visibility == 0:
        initial_visibility = 1.0
    elif initial_visibility == 1:
        initial_visibility = 0.9
    elif initial_visibility == 2:
        initial_visibility = 0.75
    elif initial_visibility == 3:
        initial_visibility = 0.5
    elif initial_visibility == 4:
        initial_visibility = 0.25
    elif initial_visibility == 5:
        initial_visibility = 0.0

    visibility = visibility * initial_visibility
    area_ratio = (visibility * area_ann) / np.maximum(area_det, np.finfo(float).eps)

    return visibility, area_ratio, ann_bbox, extra_folder


def load_model_and_meta_params_from_json(validation=False):
    """
    creates a keras model from the given config file (json) and loads the flags specified in the "meta" object
    :return: a keras model
    """
    with open(FLAGS.model_file, 'r') as mf:
        loaded_model_file = mf.read()

    params = json.loads(loaded_model_file)
    meta_params = params["meta"]
    meta_params["model_type"] = params["class_name"]

    # Legacy: Convert Sequential Model to Functional Model
    if meta_params["model_type"] == "Sequential":
        seq_model = keras.models.model_from_json(json.dumps(params), custom_objects=custom_objects)
        func_model = keras.Model(seq_model.inputs[0], seq_model.outputs[0])
        func_model.summary()
        func_model_json = func_model.to_json()
        func_model_dict = json.loads(func_model_json)
        func_model_dict.update({"meta": meta_params})

        func_model_json = json.dumps(func_model_dict, indent=2)
        with open(FLAGS.model_file, 'w') as my_data_file:
            my_data_file.write(func_model_json)
        eprint("Converted Sequential Model to Functional Model -> Please Restart!!!")
        exit(0)

    input_shapes = []
    input_layers = params["config"]["input_layers"]
    for input_layer in input_layers:
        for layer in params["config"]["layers"]:
            if layer['name'] == input_layer[0]:
                input_shape = layer["config"]["batch_input_shape"]
                if validation:
                    input_shape[0] = FLAGS.validation_batch_sizee
                else:
                    input_shape[0] = FLAGS.batch_size
                input_shapes.append(input_shape)
                break

    meta_params["height"] = input_shapes[0][-3]
    meta_params["width"] = input_shapes[0][-2]
    meta_params["channels"] = input_shapes[0][-1]

    keras_model = keras.models.model_from_json(json.dumps(params), custom_objects=custom_objects)

    if not validation:
        if DEBUGGING:
            print_seperator()
            print("[load_model_and_meta_params_from_json] Flags before:")
            print(get_non_default_flags())

        for key, value in meta_params.items():
            FLAGS.__setattr__(key, value)

        if DEBUGGING:
            print_seperator()
            print("[load_model_and_meta_params_from_json] Flags after:")
            print(get_non_default_flags())
            print_seperator()

        print("Loaded model from json-file {}".format(FLAGS.model_file))

    return keras_model


def save_keras_model_to_file(model, model_name):
    model_json = model.to_json()
    model_dict = json.loads(model_json)
    model_json = json.dumps(model_dict, indent=2)
    with open(model_name, 'w') as my_data_file:
        my_data_file.write(model_json)


def get_flops(model, model_name="", debug=False):
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential.
    """
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    batch_size = 1
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(batch_size,) + model.input_shape[1:])])
    options = ProfileOptionBuilder.float_operation()
    options['order_by'] = 'name' # name|depth|bytes|peak_bytes|residual_bytes|output_bytes|micros|accelerator_micros|cpu_micros|params|float_ops|occurrence

    if not debug:
        options['output'] = "none"
    graph_info = profile(forward_pass.get_concrete_function().graph, options=options)

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    macs = graph_info.total_float_ops // 2
    # print(f"{model_name}: {flops / 10 ** 6} MAC")
    print(f"{model_name}: {macs:,d} #MAC")
    return macs


def find_all_checkpoints(model_filename):
    """
    finds the in load_checkpoint specified or else latest checkpoint file
    :return: checkpoint filename, number of epochs trained
    """
    epoch_list = []
    for root, dirs, files in os.walk(FLAGS.trained_models_dir, topdown=True):
        for file in files:
            if file.startswith(model_filename + "-") and file.endswith(".h5") and (root == FLAGS.trained_models_dir or os.path.basename(root) == model_filename):
                epoch = extract_epoch_number(model_filename, file)
                if epoch:
                    epoch_list.append((os.path.join(root, file), epoch))
        if not FLAGS.search_subfolders:
            break

    epoch_list.sort(key=lambda x: x[1], reverse=True)
    return epoch_list


def extract_epoch_number(model_filename, checkpoint_filename):
    first_minus = len(model_filename) + find_nth(checkpoint_filename[len(model_filename):], "-", 1)
    second_minus = len(model_filename) + find_nth(checkpoint_filename[len(model_filename):], "-", 2)
    try:
        return int(checkpoint_filename[first_minus + 1:second_minus])
    except ValueError as ve:
        eprint("Could not parse int: " + str(ve) + " of file " + str(checkpoint_filename))
        return None


def load_checkpoint(model_filename, model, checkpoint=None):
    """
    loads the in load_checkpoint specified or else latest checkpoint or initializes the weights of the detection layer
    :return:
    """
    # trained_model_file_name, initial_epoch = find_all_checkpoints(model_filename)
    if FLAGS.load_checkpoint >= 0:
        if not checkpoint:
            epoch_list = find_all_checkpoints(model_filename)
        else:
            epoch = extract_epoch_number("", os.path.basename(checkpoint))
            epoch_list = [(checkpoint, epoch)]
        start_index = 0
        if FLAGS.load_checkpoint > 0:
            for i, epoch in enumerate(epoch_list):
                if epoch[1] == FLAGS.load_checkpoint:
                    start_index = i
                    break

        if len(epoch_list) > 0:
            for i in range(3):
                try:
                    trained_model_file_name, initial_epoch = epoch_list[start_index + i][0], epoch_list[start_index + i][1]
                    try:
                        functional_model = model
                        functional_model.load_weights(trained_model_file_name)
                    except Exception as e:
                        eprint(e)
                        functional_model = keras.models.load_model(trained_model_file_name, custom_objects=custom_objects, compile=False)
                    print_seperator()
                    print("Loaded last checkpoint for model {}! Resuming epoch {}".format(model_filename, initial_epoch))
                    FLAGS.initial_epoch = initial_epoch

                    callback_file = os.path.join(os.path.dirname(trained_model_file_name), model_filename + "-" + "{epoch:04d}".format(epoch=initial_epoch) + "-callbacks.pkl")
                    optimizer_file = os.path.join(os.path.dirname(trained_model_file_name), model_filename + "-" + "{epoch:04d}".format(epoch=initial_epoch) + "-optimizer.pkl")

                    weight_values = None
                    if os.path.isfile(optimizer_file):
                        with open(optimizer_file, 'rb') as f:
                            weight_values = pickle.load(f)

                    if os.path.isfile(callback_file):
                        #return functional_model, load_optimizer_from_h5(trained_model_file_name, weight_values), pickle.load(open(callback_file, "rb"))
                        return functional_model, weight_values, pickle.load(open(callback_file, "rb"))
                    else:
                        #return functional_model, load_optimizer_from_h5(trained_model_file_name, weight_values), None
                        return functional_model, weight_values, None
                except Exception as e:
                    print_seperator()
                    print("!!!" + str(i + 1) + ". try... No Checkpoint found. !!!")
                    print(str(e))

            FLAGS.initial_epoch = 0
            return None, None, None
        else:
            FLAGS.initial_epoch = 0
            return None, None, None
    else:
        FLAGS.initial_epoch = 0
        return None, None, None


def load_train_and_validation_images(load_all=False, pickleDataset="", hardMultiplier=5):
    ############################################
    # parse annotations of the training set
    ############################################
    if pickleDataset != "":
        try:
            dataset = pickle.load(open(pickleDataset, "rb"))
            print("Loaded Dataset from pickle!")
            train_imgs = dataset["train_imgs"]
            valid_imgs = dataset["valid_imgs"]
            return train_imgs, valid_imgs
        except:
            eprint("Pickle-Dataset", pickleDataset, "was not found!")

    train_imgs = parse_annotation_protobuf(
        img_dir=FLAGS.img_dir,
        labels=FLAGS.label_names,
        search_subfolders=FLAGS.search_subfolders,
        folder_filter_list=FLAGS.img_dir_folder_filter_list,
        load_all=load_all,
        use_multithreading=not DEBUGGING
    )

    if FLAGS.val_dir and FLAGS.val_dir != "":
        valid_imgs = load_validation_images_from_folder(FLAGS.val_dir, load_all=True)
    else:
        print("Use training/validation split with " + str(FLAGS.training_validation_rate) + "!")
        train_valid_split = int(FLAGS.training_validation_rate * len(train_imgs))
        if FLAGS.training_validation_rate < 1.0 and FLAGS.training_validation_rate > 0.0:
            print("Shuffling images for training/validation split!")
            np.random.shuffle(train_imgs)
        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    for img in train_imgs:
        if img['hard'] is True:
            img['hard'] = False
            for i in range(hardMultiplier):
                train_imgs.append(img)

    dataset = {}
    np.random.shuffle(train_imgs)
    dataset["train_imgs"] = train_imgs
    np.random.shuffle(valid_imgs)
    dataset["valid_imgs"] = valid_imgs
    with open(os.path.join(FLAGS.img_dir, "dataset.pkl"), 'wb+') as f:
        pickle.dump(dataset, f)
    return train_imgs, valid_imgs


def load_validation_images_from_folder(img_dir, load_all=True):
    print("Parse validation dir:")
    valid_imgs = parse_annotation_protobuf(
        img_dir=img_dir,
        labels=FLAGS.label_names,
        search_subfolders=FLAGS.search_subfolders,
        folder_filter_list=FLAGS.val_dir_folder_filter_list,
        load_all=load_all,
        use_multithreading=not DEBUGGING
    )
    return valid_imgs


def make_train_and_validation_generator(train_imgs, valid_imgs, generator_config_function, generator_function):
    ############################################
    # Make train and validation generators
    ############################################
    generator_config = generator_config_function(validation=False)
    validation_generator_config = generator_config_function(validation=True)

    train_generator = generator_function(images=train_imgs,
                                         config=generator_config,
                                         shuffle=True,
                                         grayscale=False if FLAGS.color_model == "rgb" else True,
                                         augmentation_threshold=FLAGS.augmentation,
                                         norm=FLAGS.norm,
                                         name="training",
                                         num_cores=FLAGS.num_cores)

    valid_generator = generator_function(images=valid_imgs,
                                         config=validation_generator_config,
                                         shuffle=True,
                                         grayscale=False if FLAGS.color_model == "rgb" else True,
                                         augmentation_threshold=0.0,
                                         norm=FLAGS.norm,
                                         name="validation",
                                         num_cores=FLAGS.num_cores)

    return train_generator, valid_generator


def convert_to_tflite(model, model_name, generator=None, quantization=QUANTIZATION.NONE):
    # Convert the model.
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

    single_model = tf.keras.models.model_from_json(json_model)
    single_model.set_weights(model.get_weights())

    if quantization == QUANTIZATION.MODEL:
        import tensorflow_model_optimization
        quant_aware_model = tensorflow_model_optimization.quantization.keras.quantize_model(single_model)
        quant_aware_model.summary(line_length=150)
        converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(single_model)

    def representative_dataset_gen():
        x_batch, y_true_batch = generator.__getitem__(0)
        for i in range(len(x_batch)):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [np.expand_dims(x_batch[i], 0)]

    if quantization == QUANTIZATION.DYNAMIC_INT8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == QUANTIZATION.INT8_FLOAT_FALLBACK and generator is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
    elif quantization == QUANTIZATION.INT8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    elif quantization == QUANTIZATION.FP16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        pass

    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TF Lite model.
    tflite_name = model_name + "_" + str(quantization.name) + '.tflite'
    with tf.io.gfile.GFile(tflite_name, 'wb') as f:
      f.write(tflite_model)
    print("Converted model to tflite: " + tflite_name)

    conv_interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_details = conv_interpreter.get_input_details()
    print(input_details)


def merge_first_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)