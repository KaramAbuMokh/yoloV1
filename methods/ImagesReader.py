import os
import numpy as np
import cv2
from matplotlib.image import imread

def images_reader(path):
    '''

    :return: all the images values
    '''

    images_arr = []
    images_shape = []

    for image_name in os.listdir(path):
        img = imread(path + '/' + image_name)
        images_shape.append(img.shape)
        dim = (448, 448)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        images_arr.append(resized)


    return images_arr, images_shape