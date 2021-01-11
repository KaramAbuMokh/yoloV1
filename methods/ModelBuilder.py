'''
in this file the function that build the model
'''
import math
import keras
from keras import backend as K
import tensorflow as tf
from pandas import np
from tensorflow.keras.losses import binary_crossentropy

from methods.Parameters import *


def bb_intersection_over_union(boxA, boxB):
    '''
    calculate and return the intersection over union
    this function used in the loss function that i implemented my_loss()
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def my_loss(y_t, y_p):
    '''
    implimenting the loss function
    but this also didnt work
    so i used binary_crossentropy
    '''
    sum=0
    for img_p, img_t in zip(y_p, y_t):
        for i, cell_p, cell_t in zip(range(grid_width*grid_height), img_p, img_t):
            for label_p, label_t in zip(cell_p[:3], cell_t[:3]):
                sum += (label_t - label_p) ** 2

            for box_p, box_t in zip(np.split(cell_p[3:], 5)), zip(np.split(cell_p[3:], 5)):

                sum += 5 * (((box_p[0] - box_t[0]) ** 2) + ((box_p[1] - box_t[1]) ** 2))
                sum += 5 * (((math.sqrt(box_p[2]) - math.sqrt(box_t[2])) ** 2) + ((math.sqrt(box_p[3]) - math.sqrt(box_t[3])) ** 2))
                iou=bb_intersection_over_union(box_p, box_t)
                sum += 5*((iou * box_p[4] - iou * box_t[4]) ** 2)
    return sum


def custom_loss(y_true, y_pred):
    '''
    i got this function from github and made changes on it to fit my model
    but also didnt work

    and above i implimented the loss and aslo didnt work
    '''
    # define a grid of offsets
    # [[[ 0.  0.]]
    # [[ 1.  0.]]
    # [[ 0.  1.]]
    # [[ 1.  1.]]]
    grid = np.array([[[float(x), float(y)]] * boxes_in_cell for y in range(grid_height) for x in range(grid_width)])

    # first three values are classes : cat, rat, and none.
    # However yolo doesn't predict none as a class, none is everything else and is just not predicted
    # so I don't use it in the loss
    y_true_class = y_true[..., 0:3]
    y_pred_class = y_pred[..., 0:3]

    # reshape array as a list of grid / grid cells / boxes / of 5 elements
    pred_boxes = K.reshape(y_pred[..., 3:], (-1, grid_width * grid_height, boxes_in_cell, 5))
    true_boxes = K.reshape(y_true[..., 3:], (-1, grid_width * grid_height, boxes_in_cell, 5))

    # sum coordinates of center of boxes with cell offsets.
    # as pred boxes are limited to 0 to 1 range, pred x,y + offset is limited to predicting elements inside a cell
    y_pred_xy = pred_boxes[..., 0:2]
    # w and h predicted are 0 to 1 with 1 being image size
    y_pred_wh = pred_boxes[..., 2:4]
    # probability that there is something to predict here
    y_pred_conf = pred_boxes[..., 4]

    # same as predicate except that we don't need to add an offset, coordinate are already between 0 and cell count
    y_true_xy = true_boxes[..., 0:2]
    # with and height
    y_true_wh = true_boxes[..., 2:4]
    # probability that there is something in that cell. 0 or 1 here as it's a certitude.
    y_true_conf = true_boxes[..., 4]

    clss_loss = K.sum(K.square(y_true_class - y_pred_class), axis=-1)
    xy_loss = K.sum(K.sum(K.square(y_true_xy - y_pred_xy), axis=-1) * y_true_conf, axis=-1)
    wh_loss = K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1) * y_true_conf, axis=-1)

    # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
    # however the training is a bit unstable

    # compute the intersection of all boxes at once (the IOU)
    intersect_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh) / 2 - K.square(y_pred_xy - y_true_xy))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_area = y_true_wh[..., 0] * y_true_wh[..., 1]
    pred_area = y_pred_wh[..., 0] * y_pred_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    iou = intersect_area / union_area

    conf_loss = K.sum(K.square(y_true_conf * iou - y_pred_conf), axis=-1)

    # final loss function
    d = xy_loss + wh_loss + conf_loss + clss_loss

    return d



def build_model():
    '''
    this function build the model
    :return: model
    '''
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LeakyReLU, Reshape
    # from keras.losses import mean_squared_error,binary_crossentropy

    leaky_activation_function = LeakyReLU(alpha=0.1)
    model = keras.Sequential()
    model.add(Conv2D(filters=64, kernel_size=(7, 7), padding='same', strides=2, input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(MaxPool2D(strides=2, pool_size=(2, 2)))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(MaxPool2D(strides=2, pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(MaxPool2D(strides=2, pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(MaxPool2D(strides=2, pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, strides=2, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)

    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', input_shape=image_shape))
    model.add(leaky_activation_function)

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(leaky_activation_function)
    model.add(Dense(7 * 7 * (3 + 2 * 5),
                    activation="sigmoid"))  # 7x7 grid , 3 labels, 2 num of boxes, 5 parameters (# coordinates + confidence)
    model.add(Reshape((grid_width * grid_height, (3 + boxes_in_cell * 5))))
    model.compile(loss=binary_crossentropy, optimizer='adam')

    # see the summary of the model structure
    print(model.summary())

    return model


