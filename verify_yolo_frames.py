import enum
import json
import cv2
from yolov3_manager import *

import uuid
import glob
import os

ESCAPE_KEY = 27

SAMPLE_FRAMES_PATH = os.environ.get("SAMPLE_FRAMES_PATH", "./sample_video_frames")

extensions = ("*.png","*.jpg","*.jpeg",)
all_files = []
for extension in extensions:
    all_files.extend(glob.iglob(os.path.join(SAMPLE_FRAMES_PATH,extension)))

def process_framed_test_videos():
    import os

    config = {}
    images_count = 0

    i = 0

    for pathAndFilename in all_files:

        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        image = cv2.imread(pathAndFilename)

        image_with_labels, labels_boxes = predict_image(image)

        if len(labels_boxes) > 0:
            print(labels_boxes)

            save_file = "detected_video_frames/%s.png" % (title)
            cv2.imwrite(save_file, image_with_labels)


process_framed_test_videos()
