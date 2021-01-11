import math

import numpy as np
from  methods.Parameters import *



def get_high_conf_boxes(image_predictions):
    '''

    :param image_predictions: the output of the model , its 7x7
    grid, each cell contains two bounding boxes
    :return: the bounding boxes with confidence 0.7 and above
    '''
    returned_boxes=[]
    for box in image_predictions:
        num_of_labels=len(labels)
        coordinates=box[num_of_labels:]
        coordinates = np.array(coordinates)
        coordinates = np.array_split(coordinates, boxes_in_cell)
        for one in coordinates:
            one=list(one)
            x=[]
            if one[4]>0.7:
                y=np.array(box[:num_of_labels])
                y=np.concatenate([y,one])
                returned_boxes.append(y)

    return returned_boxes

def rescale_coordinates(box,shape):
    '''

    :param box: the box parameters
    :param shape: the image shape (height, width)
    :return: scaled parameters
    '''
    real_height=image_shape[1]
    real_width=image_shape[0]

    scaling_width=real_width/shape[1]
    scaling_height = real_height / shape[0]
    box[3]=box[3]*scaling_width
    box[5] = box[5] * scaling_width
    box[4] = box[4] * scaling_height
    box[6] = box[6] * scaling_height

    return box


