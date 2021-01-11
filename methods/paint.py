import cv2


def paint_boxes(scaled_image_boxes,image):
    '''
    this function draw the boxes on image and
    :param scaled_image_boxes: the boxes that should appear on image
    :param image: the image array
    :return:
    '''
    count=0
    for box in scaled_image_boxes:
        image = cv2.rectangle(image, (box[3], box[4]), (box[3]+box[5], box[4]+box[6]), (0, 255, 0), 3)
    return image