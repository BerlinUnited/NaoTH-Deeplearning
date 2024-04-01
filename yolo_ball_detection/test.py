import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from keras.preprocessing.image import  img_to_array

nb_boxes=1
grid_w=2
grid_h=2
cell_w=32
cell_h=32
img_w=grid_w*cell_w
img_h=grid_h*cell_h

def load_image():
    """
    Read input image and output prediction
    """
    #img = cv2.imread('Images/%d.PNG' % j)
    #    img = cv2.resize(img,(64,64))
    
    #x_t = img_to_array(img)

    with open("test.txt", "r") as f:
        y_t = []
        for row in range(grid_w):
            for col in range(grid_h):
                c_t = [float(i) for i in f.readline().split()]
                [x, y, w, h] = [float(i) for i in f.readline().split()]        
                conf_t = [float(i) for i in f.readline().split()]                
                elt = []
                elt += c_t
                for b in range(nb_boxes):
                    elt += [x/cell_w, y/cell_h, w/img_w, h/img_h] + conf_t
                y_t.append(elt)
        assert(f.readline()=="---\n")
        print(y_t)
    return y_t

y_train = []
y_t = load_image()
y_train.append(y_t)
y_train = np.array(y_train)
print(y_train)
print(y_train.shape)