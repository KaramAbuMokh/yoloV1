
from methods.ImagesReader import *
from methods.ffmconvertor import *
from methods.LabelsReader import *
from keras_preprocessing.image import img_to_array, load_img
from matplotlib.image import imread
from methods.model_output import *
from methods.ModelBuilder import *
from methods.paint import *
import numpy as np
import tensorflow as tf


def train_model():

    '''
    here you process the dataset , build the model and
    train the model

    using this function if you have the dataset from the site : https://www.kaggle.com/twaldo/kitti-object-detection
    and have the proper hardware to train the model
    :return:
    '''
    all_images, images_shape = images_reader('training/image_2/')
    bounding_boxes = bounding_boxes_reader(images_shape)
    model = build_model()
    all_images = np.array(all_images)
    bounding_boxes = np.array(bounding_boxes)

    model.fit(all_images[:1800],bounding_boxes[:1800],
              validation_data=(all_images[1800:], bounding_boxes[1800:]),
              batch_size=16, epochs=135,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss')])
    model.save('m.h5')
    pass
def load_image_pixels(filename, shape):
    '''
    this function is to load image and
    scale it in order to fit the network
    input

    it takes the file name and the wanted shape
    the file should be located in the same
    directory with this code

    return the scaled image and the original size
    '''

    # load the image to get its original shape
    # in order to rescale the bounding boxes to
    # fit the original image
    image = load_img(filename)
    width, height = image.size

    # load the image with the required size
    image = load_img(filename, target_size=shape)

    # convert to numpy array
    image = img_to_array(image)

    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)

    # return the scaled image numpy array and the original size
    return image, width, height

if __name__ == '__main__':
    # train the model
    train_model()


    from tensorflow.keras.models import load_model
    model = load_model('m.h5')

    os.mkdir('frames')  # to save frames extracted from video
    os.mkdir("to save video made from frames with boxes")

    # convert video from this
    # directory with the name
    # vid.mp4 to frames
    convert_video_frames()



    '''
    read the inages in order to predict the boxes
    '''
    all_images, images_shape = images_reader('frames')

    '''
    for each image predict the 
    boxes and resize the image 
    and the prediction
    to the original shape 
    '''
    images_from_frames=[]
    resized_images = []
    high_conf_boxes = []
    for img, shape in zip(os.listdir('frames'), images_shape):
        img = imread('frames' + '/' + img)
        images_from_frames.append(img)
        dim = (448, 448)
        img_to_pred = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img_to_pred = np.array(img_to_pred)
        img_to_pred = img_to_pred / 255.0

        img_to_pred=img_to_pred.reshape(1,448,448,3)
        predicted_image = model.predict(img_to_pred)[0]

        high_conf_boxes.append(get_high_conf_boxes(predicted_image))

    # cleaning the directory with
    # the name frames in order
    # to save the images with drawing
    clean_frames_folder()

    '''
    for each image , we scale the
    coordinates for each box, draw 
    the boxes and save the image in 
    the directory with the name frames
    '''
    scaled_image_boxes=[]
    for image, image_boxes_high_cong, shape in zip(images_from_frames, high_conf_boxes, images_shape):
        count = 0
        for box in image_boxes_high_cong:
            scaled_image_boxes.append(rescale_coordinates(box, shape))
        i = paint_boxes(scaled_image_boxes, image)
        cv2.imwrite("frames/%07d.jpg" % count, image)
        count += 1

    # creating video from the images
    # in the directory with the name
    # frames and save the video in the
    # directory with the name stream
    create_mp4()
