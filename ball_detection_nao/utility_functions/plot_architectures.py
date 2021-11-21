import tensorflow.keras as keras

from classification_models import *

# plot first model
model = naoth_classification1()
dot_img_file = model._name + '.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model = naoth_classification2()
dot_img_file = model._name + '.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)