'''
in this file there are functions help with reading
and processing the bounding boxes data
'''
import math
import os
import random

import numpy as np
from methods.Parameters import *


def scale_coords(coordinates, i_shape):
    '''

    :param coordinates: left, top, right, bottom
    :param i_shape: shape of the image : height, width
    :return: scaled coordinates
    '''
    x1, y1, x2, y2 = coordinates[:]

    scaling_height = i_shape[0] / 448
    scaling_width = i_shape[1] / 448

    x1 = x1 / scaling_width
    x2 = x2 / scaling_width
    y1 = y1 / scaling_height
    y2 = y2 / scaling_height
    return [x1, y1, x2, y2]


def get_labels_from_txt(text_file_lines, i_shape):
    '''

    :param text_file_lines: array of the lines of individual text file
    :param i_shape: the shape of the image that this text file represent
    :return: all the boxes in this image
    '''
    labels_in_txt = []
    for line in text_file_lines:
        params = line.split(" ")
        label = []

        if params[0] in labels:
            label = label + [params[0]]

            coordinates = np.array(params[4:8])
            coordinates = coordinates.astype(np.float)
            scaled_coords = scale_coords(coordinates, i_shape)

            label = label + scaled_coords

            labels_in_txt.append(label)

    if len(labels_in_txt) == 0:
        labels_in_txt.append([0., 0., 0., 0., 0., 0., 0.])
    return labels_in_txt


def get_all_boxes_from_text(images_shape):
    '''

    :param images_shape:  the shape of all the images in order to scale the values
    :return: array of images , each image represented by it group of  boxes
    '''
    all_boxes = []
    boxes_path = 'training/label_2/'

    for one_file, i_shape in zip(os.listdir(boxes_path), images_shape):
        f = open(boxes_path + '/' + one_file, "r")
        text_file = f.read()
        text_file_lines = text_file.splitlines()

        # from each text file get the true y
        labels_in_txt = get_labels_from_txt(text_file_lines, i_shape)
        all_boxes.append(labels_in_txt)

    return all_boxes


def get_cell_boxes(x, y, image, i_shape):
    '''

    :param x: line index
    :param y: column index
    :param image:
    :param i_shape:
    :return: label and coordinates of two boxes that this cell responsible of detecting them
    if there is no object in this cell the the confidence is 0
    '''
    two_boxes = []
    for box in image:
        coordinates = np.array(box[1:])

        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2]
        y2 = coordinates[3]

        mid1, mid2 = (x1 + x2) / 2, (y1 + y2) / 2
        categories = [0., 0., 0.]

        if (x * 64) < mid1 < (x + 1) * 64 and y * 64 < mid2 < (y + 1) * 64:
            categories[labels.index(box[0])] = 1 + categories[labels.index(box[0])]
            x1, y1, x2, y2 = x1, y1, x2 - x1, y2 - y1
            two_boxes = categories + [x1, y1, x2, y2] + [1.]
            two_boxes = two_boxes + [x1, y1, x2, y2] + [1.]
            return two_boxes

    if len(two_boxes) == 0:
        a = 0
        b = 447
        two_boxes = [0., 0., 0., float(random.randint(a, b)), float(random.randint(a, b)), float(random.randint(a, b)),
                     float(random.randint(a, b)), 0.
            , float(random.randint(a, b)), float(random.randint(a, b)), float(random.randint(a, b)),
                     float(random.randint(a, b)), 0.]
    return two_boxes









def get_7x7_grid(image, i_shape):
    '''

    :param image: the reaal boxes in the image
    :param i_shape: the image shape
    :return: return 7x7 grid , each cell responsible for two bounding boxes
    '''
    imagei = []
    for x in range(grid_width):
        for y in range(grid_height):
            two_boxes = get_cell_boxes(x, y, image, i_shape)
            imagei.append(two_boxes)

    return imagei


def bounding_boxes_reader(images_shape):
    '''

    :param images_shape: shape of all images
    :return: y_train
    '''
    all_boxes_txt = get_all_boxes_from_text(images_shape)  # all real boxes

    y_to_return = []

    for image, i_shape in zip(all_boxes_txt, images_shape):
        imagei = get_7x7_grid(image, i_shape)
        y_to_return.append(imagei)
    y_to_return = np.array(y_to_return)

    return y_to_return
