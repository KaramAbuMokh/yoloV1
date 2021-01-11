import subprocess
import cv2
import os, shutil


def convert_video_frames():
    '''

    :param input: video path
    :param output: output path
    : divide video into a frames
    '''

    vidcap = cv2.VideoCapture('vid.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/%07d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    pass


def clean_frames_folder():
    '''
    this function will delete the images in the frames directory

    '''
    folder = 'frames/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    pass


clean_frames_folder()
def create_mp4():

    fps, duration = 24, 5
    subprocess.call(
        ["ffmpeg", "-y", "-r"
            , str(fps), "-i"
            , "frames/%07d.jpg"
            , "-vcodec", "mpeg4"
            , "-qscale", "5", "-r"
            , str(fps),"stream/video.mp4"])
    pass



