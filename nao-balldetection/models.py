import tensorflow as tf
from tensorflow.keras import layers, models

def mbc_36ksm_finetuned_crop():
    model = models.Sequential(name="mbc_36ksm_finetuned_crop")

    # ==========================================
    # CONVOLUTIONAL FEATURE EXTRACTOR
    # ==========================================

    model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(1, 1), 
        padding='same', activation='relu', input_shape=(16, 16, 1), name="layer_1"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(2, 2), 
        padding='valid', activation='relu', name="layer_2"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', name="layer_3"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', name="layer_4"
    ))

    # ==========================================
    # CLASSIFIER HEAD (DENSE LAYERS)
    # ==========================================
    
    model.add(layers.Flatten(name="flatten"))

    model.add(layers.Dense(units=256, activation='relu', name="layer_5"))

    model.add(layers.Dense(units=32, activation='relu', name="layer_6"))

    model.add(layers.Dense(
        units=2, 
        activation='softmax',
        name="output_probabilities"
    ))

    return model

def rc26_classification_color_32():
    # modified from mbc_36ksm_finetuned_crop
    model = models.Sequential(name="rc26_classification_color_32")

    # ==========================================
    # CONVOLUTIONAL FEATURE EXTRACTOR
    # ==========================================

    model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(1, 1), 
        padding='same', activation='relu', input_shape=(32, 32, 3), name="layer_1"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(2, 2), 
        padding='valid', activation='relu', name="layer_2"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(2, 2), 
        padding='valid', activation='relu', name="layer_3"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', name="layer_4"
    ))

    # ==========================================
    # CLASSIFIER HEAD (DENSE LAYERS)
    # ==========================================
    
    model.add(layers.Flatten(name="flatten"))

    model.add(layers.Dense(units=256, activation='relu', name="layer_5"))

    model.add(layers.Dense(units=32, activation='relu', name="layer_6"))

    model.add(layers.Dense(
        units=2, 
        activation='softmax',
        name="output_probabilities"
    ))

    return model


def mbd_gopen_56k():
    model = models.Sequential(name="mbd_gopen_56k")

    # ==========================================
    # CONVOLUTIONAL FEATURE EXTRACTOR
    # ==========================================
    
    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', input_shape=(16, 16, 1), name="layer_1"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(2, 2), 
        padding='valid', activation='relu', name="layer_2"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', name="layer_3"
    ))

    model.add(layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), 
        padding='valid', activation='relu', name="layer_4"
    ))

    # ==========================================
    # CLASSIFIER HEAD (DENSE LAYERS)
    # ==========================================
    
    model.add(layers.Flatten(name="flatten"))

    model.add(layers.Dense(units=512, activation='relu', name="layer_5"))

    model.add(layers.Dense(units=32, activation='relu', name="layer_6"))

    model.add(layers.Dense(
        units=3, 
        activation='linear',
        name="output_scores"
    ))

    return model


if __name__ == "__main__":
    # Instantiate and look at the beautiful architectural match!
    #detector_model = mbd_gopen_56k()
    #detector_model.summary()

    classifier_model = mbc_36ksm_finetuned_crop()
    classifier_model.summary()

    model = rc26_classification_color_32()
    model.summary()